// elf.cpp — Santa 2026 engine v1.
//
// Planner: repeatedly searches every remaining target from the current set of built values and
// builds the cheapest one ("cheapest target first"). Per-target search is a backward beam over
// reductions  v = child*f ± atom  /  v = child ± atom, with f/atom taken from a dictionary of all
// values reachable in 0, 1 or 2 instructions from a leaf set (built values). Optionally a Bloom
// filter of 3-instruction values extends the dictionary by one level (exactly verified).
// Allocator: plan is a DAG of values in creation order; it is laid out on 48 registers with a
// Belady policy (evict the value whose next use is farthest) and values evicted too early are
// rebuilt from their recipe. Every program is re-verified before output.
//
// Input  (stdin or file): one puzzle per line:  id L ns s... nt t...
// Output (stdout): one line per puzzle:  id length [[op,i,j,d],...]
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;

// ---------------------------------------------------------------- parameters
struct Params {
    int width = 10;          // beam width
    double alpha = 0.18;     // rank = cost + alpha*log2(residual)
    int n_leaves = 80;       // leaf budget for the dictionary
    int near_k = 4;          // neighbours examined in sorted dictionary
    int bloom_lg = 0;        // 0 = no S3 bloom; else log2(bits)
    int lookahead = 0;       // candidates evaluated with one step lookahead (0 = off, 1 = real-cost greedy)
    double la_weight = 0.4;  // lookahead: own + w * remaining
    int la_alts = 3;         // alternative solutions kept per target
    int la_slack = 1;        // solutions up to best+slack are kept
    int la_width = 4;        // beam width for the remaining-cost estimate
    int seed = 1;
    double noise = 0.0;      // random tie-break noise added to target costs
    int threads = 8;
    int verbose = 0;
    int palette = 1;         // build 1, powers of two and small odd constants up front (as leaves)
    int n_done = 52;         // leaves: previous targets (largest first) up to this many leaves
};

static inline u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

// ---------------------------------------------------------------- dictionary
struct Expr { u64 v; int a, b; char op; uint8_t cost; };

struct Index {  // open addressing value -> id; interleaved slots; resizable between builds
    struct Slot { u64 key; int id; };
    vector<Slot> tab; vector<u64> used; u64 mask = 0; size_t cap = 0; int bits = 0;
    static const int FBITS = 25;              // presence filter (4 MB): rejects most misses in cache
    vector<u64> filt = vector<u64>((1ULL << FBITS) / 64, 0);
    static inline u64 fpos(u64 v) { return (mix64(v) >> 20) & ((1ULL << FBITS) - 1); }
    void init(int b) { bits = b; tab.assign(1ULL << b, {0, -1}); mask = (1ULL << b) - 1; cap = (size_t)((1ULL << b) * 0.6); used.clear(); }
    void ensure(size_t expected) {  // call on an empty index
        int b = 16; while ((1ULL << b) * 0.5 < expected && b < 27) b++;
        if (b != bits) init(b);
    }
    inline int get(u64 v) const {
        u64 f = fpos(v);
        if (!(filt[f >> 6] >> (f & 63) & 1)) return -1;
        u64 i = mix64(v) & mask;
        while (tab[i].key) { if (tab[i].key == v) return tab[i].id; i = (i + 1) & mask; }
        return -1;
    }
    inline bool insert(u64 v, int id) {
        if (used.size() >= cap) return false;
        u64 f = fpos(v);
        bool maybe = filt[f >> 6] >> (f & 63) & 1;
        u64 i = mix64(v) & mask;
        if (maybe) while (tab[i].key) { if (tab[i].key == v) return false; i = (i + 1) & mask; }
        else while (tab[i].key) i = (i + 1) & mask;
        tab[i] = {v, id}; used.push_back(i); filt[f >> 6] |= 1ULL << (f & 63); return true;
    }
    void clear() {
        for (u64 i : used) { u64 f = fpos(tab[i].key); filt[f >> 6] = 0; tab[i].key = 0; }
        used.clear();
    }
};

struct Bloom {
    vector<u64> bits; u64 mask = 0; bool on = false;
    void init(int lg) { if (lg <= 0) { on = false; return; } bits.assign((1ULL << lg) / 64, 0); mask = (1ULL << lg) - 1; on = true; }
    void clear() { if (on) fill(bits.begin(), bits.end(), 0); }
    inline void add(u64 v) {
        u64 h = mix64(v), h2 = mix64(v ^ 0x9e3779b97f4a7c15ULL);
        u64 a = h & mask, b = ((h >> 32) ^ h2) & mask, c = h2 & mask;
        bits[a >> 6] |= 1ULL << (a & 63); bits[b >> 6] |= 1ULL << (b & 63); bits[c >> 6] |= 1ULL << (c & 63);
    }
    inline bool has(u64 v) const {
        u64 h = mix64(v), h2 = mix64(v ^ 0x9e3779b97f4a7c15ULL);
        u64 a = h & mask, b = ((h >> 32) ^ h2) & mask, c = h2 & mask;
        return (bits[a >> 6] >> (a & 63) & 1) && (bits[b >> 6] >> (b & 63) & 1) && (bits[c >> 6] >> (c & 63) & 1);
    }
};

struct Dict {
    u64 L = 0;
    Index index;
    vector<Expr> e;
    int n0 = 0, n1 = 0, n2 = 0;
    array<vector<pair<u64, int>>, 3> sorted;
    Bloom bloom;  // values of tree cost 3 (S2 op S0, S1 op S1)
    Dict() { index.init(20); }
    inline void add(u64 v, int a, int b, char op, int cost) {
        if (v > 0 && v <= L && index.insert(v, (int)e.size())) e.push_back({v, a, b, op, (uint8_t)cost});
    }
    inline void combine(int i, int j, int cost) {
        u64 a = e[i].v, b = e[j].v;
        if (a + b <= L) add(a + b, i, j, '+', cost);
        if (a > b) add(a - b, i, j, '-', cost); else if (b > a) add(b - a, j, i, '-', cost);
        if ((u128)a * b <= L) add(a * b, i, j, '*', cost);
        if (a % b == 0) add(a / b, i, j, '/', cost);
        if (a != b && b % a == 0) add(b / a, j, i, '/', cost);
    }
    static inline int results(u64 a, u64 b, u64 L, u64 *out) {
        int n = 0;
        if (a + b <= L) out[n++] = a + b;
        if (a > b) out[n++] = a - b; else if (b > a) out[n++] = b - a;
        if ((u128)a * b <= L) out[n++] = a * b;
        if (a % b == 0) out[n++] = a / b;
        if (a != b && b % a == 0) out[n++] = b / a;
        return n;
    }
    void build(const vector<u64> &leaves, u64 limit) {
        L = limit;
        index.clear(); e.clear();
        {   // size the table for the expected dictionary
            size_t n = leaves.size(), est = n * n * 3 + n * n * n * 6 + 1024;
            if (est > (1ULL << 26)) est = 1ULL << 26;
            index.ensure(max<size_t>(est / 4, 1 << 16));
        }
        for (auto &s : sorted) s.clear();
        for (u64 v : leaves) add(v, -1, -1, 0, 0);
        n0 = e.size();
        for (int i = 0; i < n0; i++) for (int j = 0; j <= i; j++) combine(i, j, 1);
        n1 = e.size();
        for (int i = n0; i < n1; i++) {
            for (int j = 0; j < n0; j++) combine(i, j, 2);
            combine(i, i, 2);
        }
        n2 = e.size();
        for (int i = 0; i < (int)e.size(); i++) sorted[e[i].cost].emplace_back(e[i].v, i);
        for (auto &s : sorted) sort(s.begin(), s.end());
        if (bloom.on) {
            bloom.clear();
            u64 out[8];
            for (int i = n1; i < n2; i++) {
                for (int j = 0; j < n0; j++) { int k = results(e[i].v, e[j].v, L, out); for (int q = 0; q < k; q++) bloom.add(out[q]); }
                int k = results(e[i].v, e[i].v, L, out); for (int q = 0; q < k; q++) bloom.add(out[q]);
            }
            for (int i = n0; i < n1; i++) for (int j = n0; j <= i; j++) { int k = results(e[i].v, e[j].v, L, out); for (int q = 0; q < k; q++) bloom.add(out[q]); }
        }
    }
    inline int get(u64 v) const { return index.get(v); }
    // exact decomposition of a bloom hit: v = x op y with cost(x)+cost(y) <= 2. Returns {op,x,y} ids or false.
    bool decompose3(u64 v, char &op, int &x, int &y) const {
        // v = A op B with B in S0 and A in S2 (or A in S1 and B in S1)
        auto tryB = [&](int bi, int maxA) -> bool {
            u64 B = e[bi].v; int ai;
            if (v > B && (ai = get(v - B)) >= 0 && e[ai].cost <= maxA) { op = '+'; x = ai; y = bi; return true; }
            if (v + B <= L && (ai = get(v + B)) >= 0 && e[ai].cost <= maxA) { op = '-'; x = ai; y = bi; return true; }
            if (B > v && (ai = get(B - v)) >= 0 && e[ai].cost <= maxA) { op = '-'; x = bi; y = ai; return true; }
            if (v % B == 0 && (ai = get(v / B)) >= 0 && e[ai].cost <= maxA) { op = '*'; x = ai; y = bi; return true; }
            if ((u128)v * B <= L && (ai = get(v * B)) >= 0 && e[ai].cost <= maxA) { op = '/'; x = ai; y = bi; return true; }
            if (B % v == 0 && (ai = get(B / v)) >= 0 && e[ai].cost <= maxA) { op = '/'; x = bi; y = ai; return true; }
            return false;
        };
        for (int bi = 0; bi < n0; bi++) if (tryB(bi, 2)) return true;
        for (int bi = n0; bi < n1; bi++) if (tryB(bi, 1)) return true;
        int ai;
        if (v % 2 == 0 && (ai = get(v / 2)) >= 0) { op = '+'; x = ai; y = ai; return true; }
        u64 r = (u64)sqrtl((long double)v);
        for (u64 s = r > 2 ? r - 2 : 1; s <= r + 2; s++) if (s * s == v && (ai = get(s)) >= 0) { op = '*'; x = ai; y = ai; return true; }
        return false;
    }
};

// ---------------------------------------------------------------- search
// A reduction link encodes parent = child*factor (op) atom, or parent = atom - child, or parent = child (op) atom.
struct Link { int factor = -1, atom = -1; char op = 0; };  // op: '+' child[*f]+atom, '-' child[*f]-atom, 'r' atom-child, '/' child/atom
struct State { u64 v; int cost = 0, parent = -1; Link link; double rank = 0; };

struct Sol { int cost, end, term, kind; };
struct Search {
    const Dict &d; const Params &P;
    vector<Sol> sols; int slack = 0;
    int best = 1 << 28, end = -1;      // end = history index of terminal state
    int term = -1; int term_kind = 0;  // term: dict id (kind 0) or bloom value (kind 3, decomposed later)
    vector<State> history, next;
    unordered_map<u64, int> visited;
    Search(const Dict &d, const Params &P) : d(d), P(P) {}
    inline int ecost(int id) const { return id < 0 ? 0 : 1 + d.e[id].cost; }
    void offer(int p, u64 v, int f, int a, char op) {
        if (v == 0 || v > d.L) return;
        int cost = history[p].cost + ecost(f) + ecost(a);
        if (op == '/') cost = history[p].cost + 1 + d.e[a].cost;  // child / atom: atom cost only
        if (cost > best + slack) return;
        int t = d.get(v);
        if (t >= 0) {
            int tc = cost + d.e[t].cost;
            if (tc <= best + slack) {
                sols.push_back({tc, (int)history.size(), t, 0});
                history.push_back({v, cost, p, {f, a, op}});
                if (tc < best) { best = tc; end = history.size() - 1; term = t; term_kind = 0; }
            }
            return;
        }
        if (cost >= best) return;
        if (d.bloom.on && cost + 3 <= best + slack && d.bloom.has(v)) {
            char o; int x, y;
            if (d.decompose3(v, o, x, y)) {
                sols.push_back({cost + 3, (int)history.size(), -1, 3});
                history.push_back({v, cost, p, {f, a, op}});
                if (cost + 3 < best) { best = cost + 3; end = history.size() - 1; term_kind = 3; term = -1; }
                return;
            }
        }
        if (cost + (d.bloom.on ? 4 : 3) >= best || v >= history[p].v) return;
        double rank = cost + P.alpha * log2((double)v);
        if ((int)next.size() == P.width && rank >= next.back().rank) return;
        auto pv = visited.find(v);
        if (pv != visited.end() && pv->second <= cost) return;
        for (auto it = next.begin(); it != next.end(); ++it)
            if (it->v == v) { if (it->cost <= cost) return; next.erase(it); break; }
        State s{v, cost, p, {f, a, op}, rank};
        auto it = upper_bound(next.begin(), next.end(), rank, [](double k, const State &x) { return k < x.rank; });
        next.insert(it, s);
        if ((int)next.size() > P.width) next.pop_back();
    }
    void expand(int p) {
        u64 v = history[p].v;
        for (auto &arr : d.sorted) {
            auto it = lower_bound(arr.begin(), arr.end(), make_pair(v, -1));
            for (int k = -P.near_k; k < P.near_k; k++) {
                i64 pos = (it - arr.begin()) + k;
                if (pos < 0 || pos >= (i64)arr.size()) continue;
                auto [a, id] = arr[pos];
                if (a == v) continue;
                offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r');
            }
        }
        for (int i = 0; i < d.n1; i++) {
            u64 a = d.e[i].v;
            if (a == 1) continue;
            u64 diff = v > a ? v - a : a - v;
            if (diff) {
                if (d.get(diff) >= 0) offer(p, diff, -1, i, v > a ? '+' : 'r');
                if ((diff & 15) == 0) shifted(p, diff, i, v > a ? '+' : 'r');
            }
            if (a + v <= d.L) {
                if (d.get(a + v) >= 0) offer(p, a + v, -1, i, '-');
                if (((a + v) & 15) == 0) shifted(p, a + v, i, '-');
            }
            if (a >= v) continue;
            u64 q = v / a, r = v % a;
            if (!r) { offer(p, q, i, -1, '+'); continue; }
            int rid = d.get(r);
            if (rid >= 0) offer(p, q, i, rid, '+');
            if (a - r <= d.L - v) { rid = d.get(a - r); if (rid >= 0) offer(p, q + 1, i, rid, '-'); }
            if (i < d.n0 && q > 64) {  // approximate the quotient by a dictionary value, then fix the error
                for (auto &arr : d.sorted) {
                    auto it = lower_bound(arr.begin(), arr.end(), make_pair(q, -1));
                    for (int k = -1; k <= 0; k++) {
                        i64 pos = (it - arr.begin()) + k;
                        if (pos < 0 || pos >= (i64)arr.size()) continue;
                        auto [b, bi] = arr[pos];
                        if ((u128)a * b > d.L) continue;
                        u64 mul = a * b;
                        if (mul == v) continue;
                        int id = d.get(mul);
                        if (id >= 0) offer(p, mul > v ? mul - v : v - mul, -1, id, mul < v ? '+' : 'r');
                    }
                }
            }
        }
    }
    void shifted(int p, u64 diff, int atom, char op) {
        int k = __builtin_ctzll(diff);
        while (k >= 4) {
            int f = d.get(1ULL << k);
            if (f >= 0) { offer(p, diff >> k, f, atom, op); break; }
            k--;
        }
    }
    void run(u64 target) {
        history.clear(); next.clear(); visited.clear(); best = 1 << 28; end = -1; sols.clear();
        history.push_back({target, 0, -1, {}});
        int id = d.get(target);
        if (id >= 0) { end = 0; term = id; term_kind = 0; best = d.e[id].cost; sols.push_back({best, 0, id, 0}); return; }
        if (d.bloom.on && d.bloom.has(target)) {
            char o; int x, y;
            if (d.decompose3(target, o, x, y)) { end = 0; term_kind = 3; best = 3; sols.push_back({3, 0, -1, 3}); return; }
        }
        vector<int> front{0};
        for (int depth = 0; depth < 24 && !front.empty(); depth++) {
            next.clear();
            for (int p : front) if (history[p].cost + 3 < best) expand(p);
            front.clear();
            for (State s : next) {
                if (s.cost + 3 >= best) continue;
                visited[s.v] = s.cost;
                front.push_back(history.size());
                history.push_back(s);
            }
        }
    }
};

// ---------------------------------------------------------------- puzzle / planner
struct Node { u64 v, a, b; char op; };  // value = a op b

struct Planner {
    u64 L; vector<u64> starts, targets; Params P;
    map<u64, Node> defs;     // built non-start values
    set<u64> have;           // starts + built
    vector<u64> order;       // creation order of built values
    mt19937_64 rng;
    Planner(u64 L, vector<u64> s, vector<u64> t, Params p) : L(L), starts(s), targets(t), P(p), rng(p.seed) {
        for (u64 v : starts) have.insert(v);
    }
    u64 make(char op, u64 a, u64 b) {
        u64 v;
        if (op == '+') v = a + b; else if (op == '-') v = a - b; else if (op == '*') v = a * b; else v = a / b;
        if (!have.count(v)) { have.insert(v); defs[v] = {v, a, b, op}; order.push_back(v); }
        return v;
    }
    u64 mat(const Dict &d, int id) {
        const Expr &e = d.e[id];
        if (have.count(e.v)) return e.v;
        u64 a = mat(d, e.a); u64 b = mat(d, e.b);
        return make(e.op, a, b);
    }
    u64 mat_bloom(const Dict &d, u64 v) {
        if (have.count(v)) return v;
        char o; int x, y;
        if (!d.decompose3(v, o, x, y)) throw runtime_error("bloom decompose failed");
        u64 a = mat(d, x), b = mat(d, y);
        return make(o, a, b);
    }
    vector<u64> palette;
    set<u64> done;
    double t_build = 0, t_search = 0;
    u64 power(int k) {
        u64 v = 1ULL << k;
        if (have.count(v)) return v;
        if (k == 0) return one();
        if (k == 1) return make('+', one(), one());
        u64 r = power(k - k / 2), l = power(k / 2);
        return make('*', l, r);
    }
    void setup() {
        if (!P.palette) return;
        palette.push_back(one());
        for (int k : {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48})
            if ((1ULL << k) <= L) palette.push_back(power(k));
        for (u64 i : {3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 31})
            if (i <= L) {
                if (!have.count(i)) {
                    if (i == 11 || i == 13 || i == 19 || i == 23) make('+', i - (i == 23 ? 16 : 8), i == 23 ? 16 : 8);
                    else if ((i & (i + 1)) == 0 && i + 1 <= L) make('-', i + 1, one());
                    else binary(i);
                }
                palette.push_back(i);
            }
    }
    vector<u64> leaves() {
        vector<u64> lv; unordered_set<u64> used;
        auto add = [&](u64 v) { if (used.insert(v).second) lv.push_back(v); };
        for (u64 v : palette) add(v);
        for (u64 v : starts) add(v);
        for (auto it = done.rbegin(); it != done.rend() && (int)lv.size() < P.n_done; ++it) add(*it);
        for (auto it = order.rbegin(); it != order.rend() && (int)lv.size() < P.n_leaves; ++it) add(*it);
        return lv;
    }
    // materialize the search result for target
    void build_from(const Dict &d, Search &s, u64 target) { build_sol(d, s, {s.best, s.end, s.term, s.term_kind}, target); }
    void build_sol(const Dict &d, Search &s, Sol sol, u64 target) {
        u64 v;
        if (sol.kind == 3) v = mat_bloom(d, s.history[sol.end].v);
        else v = mat(d, sol.term);
        for (int p = sol.end; s.history[p].parent >= 0; p = s.history[p].parent) {
            Link l = s.history[p].link;
            if (l.op == '/') { v = make('/', v, mat(d, l.atom)); continue; }
            if (l.factor >= 0) v = make('*', v, mat(d, l.factor));
            if (l.atom >= 0) {
                u64 a = mat(d, l.atom);
                v = l.op == 'r' ? make('-', a, v) : make(l.op, v, a);
            }
        }
        if (v != target) throw runtime_error("reconstruction mismatch");
    }
    void rollback(size_t n) {
        while (order.size() > n) { u64 v = order.back(); order.pop_back(); have.erase(v); defs.erase(v); }
    }
    struct Cand { u64 t; vector<Node> nodes; };
    void apply(const vector<Node> &nodes) {
        for (auto &n : nodes) if (!have.count(n.v)) { have.insert(n.v); defs[n.v] = n; order.push_back(n.v); }
    }
    // sum of best costs of the given targets from the current state (searching with width w)
    double remaining_cost(Dict &d, const vector<u64> &rem, int w) {
        Params Q = P; Q.width = w;
        d.build(leaves(), L);
        Search s(d, Q);
        double tot = 0;
        for (u64 t : rem) {
            if (have.count(t)) continue;
            s.run(t);
            tot += s.end < 0 ? 60 : s.best;
        }
        return tot;
    }
    void solve_lookahead(Dict &d) {
        setup();
        vector<u64> remaining;
        for (u64 t : targets) if (!have.count(t)) remaining.push_back(t); else done.insert(t);
        while (!remaining.empty()) {
            d.build(leaves(), L);
            Search s(d, P); s.slack = P.la_slack;
            vector<Cand> cands;
            for (u64 t : remaining) {
                s.run(t);
                if (s.end < 0) continue;
                vector<Sol> so = s.sols;
                sort(so.begin(), so.end(), [](const Sol &a, const Sol &b) { return a.cost < b.cost; });
                int kept = 0; set<vector<u64>> seen;
                for (auto &sol : so) {
                    if (kept >= P.la_alts) break;
                    size_t n0 = order.size();
                    build_sol(d, s, sol, t);
                    Cand c{t, {}};
                    for (size_t k = n0; k < order.size(); k++) c.nodes.push_back(defs[order[k]]);
                    rollback(n0);
                    vector<u64> key; for (auto &n : c.nodes) key.push_back(n.v);
                    if (!seen.insert(key).second) continue;
                    cands.push_back(move(c)); kept++;
                }
            }
            if (cands.empty()) { binary(remaining[0]); }
            else {
                stable_sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.nodes.size() < b.nodes.size(); });
                int K = min<int>(P.lookahead, cands.size());
                int bi = 0; double bs = 1e18;
                if (K > 1) {
                    for (int k = 0; k < K; k++) {
                        size_t n0 = order.size();
                        apply(cands[k].nodes);
                        vector<u64> rest;
                        for (u64 x : remaining) if (!have.count(x)) rest.push_back(x);
                        double rc = remaining_cost(d, rest, P.la_width);
                        rollback(n0);
                        double sc = cands[k].nodes.size() + P.la_weight * rc;
                        if (sc < bs) { bs = sc; bi = k; }
                    }
                }
                apply(cands[bi].nodes);
                if (P.verbose > 1) fprintf(stderr, "  target %llu built=%zu total=%zu remaining=%zu cands=%zu pick=%d\n", (unsigned long long)cands[bi].t, cands[bi].nodes.size(), order.size(), remaining.size(), cands.size(), bi);
            }
            vector<u64> rem2;
            for (u64 x : remaining) if (!have.count(x)) rem2.push_back(x); else done.insert(x);
            remaining.swap(rem2);
        }
    }
    void solve(Dict &d) {
        if (P.lookahead > 0) { solve_lookahead(d); return; }
        setup();
        vector<u64> remaining;
        for (u64 t : targets) if (!have.count(t)) remaining.push_back(t); else done.insert(t);
        uniform_real_distribution<double> U(0, 1);
        while (!remaining.empty()) {
            auto tb = chrono::steady_clock::now();
            d.build(leaves(), L);
            t_build += chrono::duration<double>(chrono::steady_clock::now() - tb).count();
            tb = chrono::steady_clock::now();
            Search s(d, P);
            int bi = -1; double bc = 1e18;
            for (int i = 0; i < (int)remaining.size(); i++) {
                s.run(remaining[i]);
                double c = s.end < 0 ? 1e9 : s.best + (P.noise > 0 ? P.noise * U(rng) : 0.0);
                if (c < bc) { bc = c; bi = i; }
            }
            t_search += chrono::duration<double>(chrono::steady_clock::now() - tb).count();
            u64 t = remaining[bi];
            s.run(t);
            size_t before = order.size();
            if (s.end < 0) binary(t); else build_from(d, s, t);
            if (P.verbose > 1) fprintf(stderr, "  target %llu est=%d built=%zu total=%zu remaining=%zu\n", (unsigned long long)t, s.best, order.size() - before, order.size(), remaining.size());
            // any other targets that appeared as by-products are done too
            vector<u64> rem2;
            for (u64 x : remaining) if (!have.count(x)) rem2.push_back(x); else done.insert(x);
            remaining.swap(rem2);
        }
    }
    u64 one() {
        for (u64 v : have) if (v == 1) return 1;
        return make('/', starts[0], starts[0]);
    }
    u64 binary(u64 v) {  // fallback: v from 1 by doubling
        if (have.count(v)) return v;
        if (v == 1) return one();
        u64 h = binary(v / 2);
        u64 tw = make('+', h, h);
        return v % 2 ? make('+', tw, one()) : tw;
    }
};

// ---------------------------------------------------------------- allocator
struct Instr { char op; int i, j, d; };

static string verify_prog(u64 L, const vector<u64> &starts, const vector<u64> &targets, const vector<Instr> &prog) {
    array<u64, 48> reg{}; array<bool, 48> def{};
    set<u64> seen(starts.begin(), starts.end());
    for (size_t k = 0; k < starts.size(); k++) { reg[k] = starts[k]; def[k] = true; }
    int step = 0;
    for (auto &in : prog) {
        string at = " at step " + to_string(step++);
        if (in.i < 0 || in.i > 47 || in.j < 0 || in.j > 47 || in.d < 0 || in.d > 47) return "bad register" + at;
        if (!def[in.i] || !def[in.j]) return "undefined source" + at;
        u64 a = reg[in.i], b = reg[in.j], r;
        if (in.op == '+') { r = a + b; }
        else if (in.op == '-') { if (a <= b) return "nonpositive sub" + at; r = a - b; }
        else if (in.op == '*') { if ((u128)a * b > L) return "mul over limit" + at; r = a * b; }
        else { if (a % b) return "inexact div" + at; r = a / b; }
        if (r == 0 || r > L) return "over limit" + at;
        reg[in.d] = r; def[in.d] = true; seen.insert(r);
    }
    for (u64 t : targets) if (!seen.count(t)) return "missing target " + to_string(t);
    if (prog.size() > 4096) return "too long";
    return "";
}

static vector<Instr> compile(const vector<u64> &starts, const vector<u64> &targets, const map<u64, Node> &defs,
                             const vector<u64> &order, int *plan_size = nullptr) {
    // keep the planner's creation order (a topological order), dropping values no target needs
    set<u64> need, startset(starts.begin(), starts.end());
    function<void(u64)> mark = [&](u64 v) {
        if (startset.count(v) || !need.insert(v).second) return;
        const Node &n = defs.at(v);
        mark(n.a); mark(n.b);
    };
    for (u64 t : targets) mark(t);
    vector<u64> seq;
    if (getenv("ELF_PLANORDER")) { for (u64 v : order) if (need.count(v)) seq.push_back(v); }
    else {  // demand-driven: targets in planner order, each cone emitted depth-first just before use
        set<u64> placed(startset), tset(targets.begin(), targets.end());
        function<void(u64)> visit = [&](u64 v) {
            if (!placed.insert(v).second) return;
            const Node &n = defs.at(v);
            visit(n.a); visit(n.b);
            seq.push_back(v);
        };
        for (u64 v : order) if (tset.count(v)) visit(v);
    }
    if (plan_size) *plan_size = seq.size();
    if (getenv("ELF_LIVESTAT")) {  // max simultaneously live values in plan order
        unordered_map<u64, int> last;
        for (int i = 0; i < (int)seq.size(); i++) { const Node &n = defs.at(seq[i]); last[n.a] = i; last[n.b] = i; }
        int live = 0, mx = 0; vector<int> ends(seq.size() + 1, 0);
        for (u64 s0 : starts) if (last.count(s0)) { live++; ends[last[s0]]++; }
        for (int i = 0; i < (int)seq.size(); i++) {
            live++; mx = max(mx, live);
            auto it = last.find(seq[i]); if (it == last.end()) live--; else ends[it->second]++;
            live -= ends[i];
        }
        fprintf(stderr, "LIVESTAT maxlive=%d plan=%zu\n", mx, seq.size());
    }
    // next-use lists
    unordered_map<u64, vector<int>> uses;  // positions (descending order stored; back() = next)
    for (int i = (int)seq.size() - 1; i >= 0; --i) {
        const Node &n = defs.at(seq[i]);
        uses[n.a].push_back(i);
        if (n.b != n.a) uses[n.b].push_back(i);
    }
    array<u64, 48> bank{}; array<bool, 48> occ{};
    // seeds cannot be rebuilt: pin every seed that any planned value reads (rebuilds may need it late)
    set<u64> seeds;
    for (u64 v : seq) { const Node &n = defs.at(v); for (u64 s : starts) if (n.a == s || n.b == s) seeds.insert(s); }
    for (size_t k = 0; k < starts.size(); k++) { bank[k] = starts[k]; occ[k] = true; }
    vector<Instr> prog;
    int pos = 0;
    function<int(u64, u64)> emit = [&](u64 v, u64 locked) -> int {
        for (int k = 0; k < 48; k++) if (occ[k] && bank[k] == v) return k;
        const Node &n = defs.at(v);
        int ra = emit(n.a, locked);
        int rb = emit(n.b, locked | (1ULL << ra));
        u64 outer = locked;
        locked |= (1ULL << ra) | (1ULL << rb);
        int rd = -1; i64 latest = -1;
        // a source register whose value dies here may receive the result (operands are read first)
        for (int k : {ra, rb}) {
            if (outer >> k & 1) continue;
            u64 old = bank[k];
            auto it = uses.find(old);
            bool dead = (it == uses.end() || it->second.empty()) && !seeds.count(old);
            if (dead) { rd = k; latest = 1LL << 41; break; }
        }
        // rebuild cost of a value given current registers (instructions), capped
        function<int(u64, int)> rcost = [&](u64 x, int depth) -> int {
            for (int k = 0; k < 48; k++) if (occ[k] && bank[k] == x) return 0;
            if (depth > 6 || !defs.count(x)) return 50;
            const Node &m = defs.at(x);
            int c = 1 + rcost(m.a, depth + 1);
            if (m.b != m.a) c += rcost(m.b, depth + 1);
            return min(c, 50);
        };
        double bestsc = -1;
        if (rd < 0) for (int k = 0; k < 48; k++) {
            if (locked >> k & 1) continue;
            if (!occ[k]) { rd = k; break; }
            u64 old = bank[k];
            auto it = uses.find(old);
            i64 nu = (it == uses.end() || it->second.empty()) ? (1LL << 40) : it->second.back();
            if (seeds.count(old)) continue;  // seeds cannot be rebuilt
            double sc;
            if (nu >= (1LL << 40)) sc = 1e18;
            else {
                bank[k] ^= 0x8000000000000000ULL;  // pretend evicted while costing its rebuild
                int rc = max(1, rcost(old, 0));
                bank[k] ^= 0x8000000000000000ULL;
                sc = double(nu - pos) / rc;
            }
            if (sc > bestsc) { bestsc = sc; rd = k; }
        }
        if (rd < 0) throw runtime_error("no register");
        prog.push_back({n.op, ra, rb, rd});
        bank[rd] = v; occ[rd] = true;
        return rd;
    };
    for (; pos < (int)seq.size(); pos++) {
        u64 v = seq[pos];
        const Node &n = defs.at(v);
        // consume this use
        auto &ua = uses[n.a]; if (!ua.empty() && ua.back() == pos) ua.pop_back();
        if (n.b != n.a) { auto &ub = uses[n.b]; if (!ub.empty() && ub.back() == pos) ub.pop_back(); }
        bool present = false;
        for (int k = 0; k < 48; k++) if (occ[k] && bank[k] == v) present = true;
        if (!present) emit(v, 0);
    }
    return prog;
}

// ---------------------------------------------------------------- driver
struct PuzzleIn { string id; u64 L; vector<u64> s, t; };

static string to_json(const vector<Instr> &p) {
    string s = "[";
    for (size_t k = 0; k < p.size(); k++) {
        if (k) s += ",";
        s += "[\""; s += p[k].op; s += "\"," + to_string(p[k].i) + "," + to_string(p[k].j) + "," + to_string(p[k].d) + "]";
    }
    return s + "]";
}

int main(int argc, char **argv) {
    Params P; string inpath;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nx = [&]() { return string(argv[++i]); };
        if (a == "-w") P.width = stoi(nx());
        else if (a == "-a") P.alpha = stod(nx());
        else if (a == "-n") P.n_leaves = stoi(nx());
        else if (a == "-k") P.near_k = stoi(nx());
        else if (a == "-b") P.bloom_lg = stoi(nx());
        else if (a == "-s") P.seed = stoi(nx());
        else if (a == "-z") P.noise = stod(nx());
        else if (a == "-t") P.threads = stoi(nx());
        else if (a == "-v") P.verbose = 1;
        else if (a == "-vv") P.verbose = 2;
        else if (a == "-p") P.palette = stoi(nx());
        else if (a == "-la") P.lookahead = stoi(nx());
        else if (a == "-lw") P.la_weight = stod(nx());
        else if (a == "-lalt") P.la_alts = stoi(nx());
        else if (a == "-lslack") P.la_slack = stoi(nx());
        else if (a == "-lwidth") P.la_width = stoi(nx());
        else if (a == "-nd") P.n_done = stoi(nx());
        else inpath = a;
    }
    vector<PuzzleIn> puzzles;
    {
        ifstream f(inpath);
        string line;
        while (getline(f, line)) {
            istringstream is(line); PuzzleIn p; int ns, nt;
            is >> p.id >> p.L >> ns; p.s.resize(ns); for (auto &x : p.s) is >> x;
            is >> nt; p.t.resize(nt); for (auto &x : p.t) is >> x;
            if (!p.id.empty()) puzzles.push_back(p);
        }
    }
    vector<string> out(puzzles.size());
    atomic<int> nextp{0}; mutex mu; long total = 0;
    auto t0 = chrono::steady_clock::now();
    auto worker = [&]() {
        Dict d; d.bloom.init(P.bloom_lg);
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            auto &pz = puzzles[k];
            string res;
            try {
                Planner pl(pz.L, pz.s, pz.t, P);
                pl.solve(d);
                if (P.verbose) fprintf(stderr, "  time build=%.2fs search=%.2fs dict=%zu n1=%d\n", pl.t_build, pl.t_search, d.e.size(), d.n1);
                int psz = 0;
                auto prog = compile(pz.s, pz.t, pl.defs, pl.order, &psz);
                string why = verify_prog(pz.L, pz.s, pz.t, prog);
                if (!why.empty()) throw runtime_error("verify failed: " + why);
                res = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);
                lock_guard<mutex> g(mu); total += prog.size();
                if (P.verbose) fprintf(stderr, "%s %zu plan=%d (%.1fs)\n", pz.id.c_str(), prog.size(), psz, chrono::duration<double>(chrono::steady_clock::now() - t0).count());
            } catch (exception &ex) {
                lock_guard<mutex> g(mu);
                fprintf(stderr, "%s FAILED: %s\n", pz.id.c_str(), ex.what());
            }
            out[k] = res;
        }
    };
    vector<thread> th;
    for (int i = 0; i < P.threads; i++) th.emplace_back(worker);
    for (auto &t : th) t.join();
    for (auto &s : out) if (!s.empty()) printf("%s\n", s.c_str());
    fprintf(stderr, "TOTAL %ld  time %.1fs\n", total, chrono::duration<double>(chrono::steady_clock::now() - t0).count());
}
