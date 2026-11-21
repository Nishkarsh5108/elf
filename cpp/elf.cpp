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
    int la_inc = 1;          // evaluate lookahead candidates with an incremental dictionary
    int lns_kmax = 4;        // LNS: max targets removed per move
    int mitm = 0;            // root meet-in-the-middle with the S3 bloom
    int rootfull = 0;        // root: use every dictionary value as a factor
    long long dict_cap = 0;  // cap on dictionary entries (0 = none)
    int beam = 1;            // program-level beam width (>1 enables beam search; uses -la as expansions per state)
    int seed = 1;
    double noise = 0.0;      // random tie-break noise added to target costs
    int threads = 8;
    int verbose = 0;
    int palette = 2;         // 2 = eager constants (free leaves), 1 = virtual (charged when used), 0 = none
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
    // returns existing id, or -1 after inserting (v,id), or -2 if full
    inline int find_or_insert(u64 v, int id) {
        if (used.size() >= cap) { int g = get(v); return g >= 0 ? g : -2; }
        u64 f = fpos(v);
        bool maybe = filt[f >> 6] >> (f & 63) & 1;
        u64 i = mix64(v) & mask;
        if (maybe) while (tab[i].key) { if (tab[i].key == v) return tab[i].id; i = (i + 1) & mask; }
        else while (tab[i].key) i = (i + 1) & mask;
        tab[i] = {v, id}; used.push_back(i); filt[f >> 6] |= 1ULL << (f & 63); return -1;
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
    static const int MAXC = 7;
    array<vector<pair<u64, int>>, MAXC + 2> sorted;   // by cost; last bucket = incremental extras
    vector<int> low, leaves_all, s1_all;               // ids of leaves and level-1 entries
    vector<int> xs_ins;                                // incremental extras in insertion order
    bool incremental = false;
    size_t cap_entries = 0;  // optional dictionary size cap (0 = none)
    struct Mark { size_t e, used, low, leaves, s1, xs, saved; };
    vector<pair<int, Expr>> saved;   // entries promoted to leaves by add_leaves (restored on rollback)
    Mark mark() const { return {e.size(), index.used.size(), low.size(), leaves_all.size(), s1_all.size(), xs_ins.size(), saved.size()}; }
    void resort_extras() {
        auto &x = sorted[MAXC + 1]; x.clear();
        for (int id : xs_ins) x.emplace_back(e[id].v, id);
        sort(x.begin(), x.end());
    }
    void add_leaves(const vector<u64> &vals) {
        incremental = true;
        size_t e0 = e.size();
        vector<int> nl;
        for (u64 v : vals) {
            int id = get(v);
            if (id >= 0) {
                if (e[id].a < 0) continue;  // already a leaf
                saved.push_back({id, e[id]});
                e[id].a = -1; e[id].b = -1; e[id].op = 0; e[id].cost = 0;
                nl.push_back(id);
                continue;
            }
            size_t b = e.size(); add(v, -1, -1, 0, 0); if (e.size() > b) nl.push_back((int)b);
        }
        touched.clear();
        for (size_t x = 0; x < nl.size(); x++) {
            for (int j : leaves_all) combine(nl[x], j, 1);
            for (size_t y = 0; y <= x; y++) combine(nl[x], nl[y], 1);
        }
        vector<int> news1 = touched;   // new or cheaper level-1 entries
        for (int i : nl) { leaves_all.push_back(i); low.push_back(i); }
        for (int k : news1) low.push_back(k);
        touched.clear();
        for (int k : news1) { for (int j : leaves_all) combine(k, j, 2); combine(k, k, 2); }
        for (int k : s1_all) for (int j : nl) combine(k, j, 2);
        for (int k : news1) s1_all.push_back(k);
        for (int k : news1) xs_ins.push_back(k);
        for (int k : touched) xs_ins.push_back(k);
        for (int i : nl) xs_ins.push_back(i);
        resort_extras();
        incremental = false;
    }
    void rollback(const Mark &m) {
        for (size_t k = m.used; k < index.used.size(); k++) index.tab[index.used[k]].key = 0;
        index.used.resize(m.used);
        while (saved.size() > m.saved) { auto &pr = saved.back(); if (pr.first < (int)m.e) e[pr.first] = pr.second; saved.pop_back(); }
        e.resize(m.e); low.resize(m.low); leaves_all.resize(m.leaves); s1_all.resize(m.s1); xs_ins.resize(m.xs);
        resort_extras();
    }
    Bloom bloom;  // values of tree cost 3 (S2 op S0, S1 op S1)
    Dict() { index.init(20); }
    inline void add(u64 v, int a, int b, char op, int cost) {
        if (v == 0 || v > L || cost > MAXC || (cap_entries && e.size() >= cap_entries)) return;
        int id = index.find_or_insert(v, (int)e.size());
        if (id == -1) { e.push_back({v, a, b, op, (uint8_t)cost}); if (incremental) touched.push_back((int)e.size() - 1); }
        else if (id >= 0 && cost < e[id].cost && e[id].a >= 0) {
            if (incremental) { saved.push_back({id, e[id]}); touched.push_back(id); }
            e[id] = {v, a, b, op, (uint8_t)cost};
        }
    }
    vector<int> touched;  // ids inserted or improved during an incremental phase
    inline void combine(int i, int j, int /*level*/) {
        u64 a = e[i].v, b = e[j].v;
        int cost = 1 + e[i].cost + (i == j ? 0 : e[j].cost);
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
    void build(const vector<u64> &leaves, u64 limit) { build(leaves, vector<int>(leaves.size(), 0), limit); }
    void build(const vector<u64> &leaves, const vector<int> &lcost, u64 limit) {
        L = limit;
        index.clear(); e.clear();
        {   // size the table for the expected dictionary
            size_t n = leaves.size(), est = n * n * 3 + n * n * n * 6 + 1024;
            if (est > (1ULL << 26)) est = 1ULL << 26;
            index.ensure(max<size_t>(est / 2, 1 << 17));
        }
        for (auto &s : sorted) s.clear();
        for (size_t k = 0; k < leaves.size(); k++) add(leaves[k], -1, -1, 0, lcost[k]);
        n0 = e.size();
        for (int i = 0; i < n0; i++) for (int j = 0; j <= i; j++) combine(i, j, 1);
        n1 = e.size();
        for (int i = n0; i < n1; i++) {
            for (int j = 0; j < n0; j++) combine(i, j, 2);
            combine(i, i, 2);
        }
        n2 = e.size();
        low.clear(); leaves_all.clear(); s1_all.clear(); xs_ins.clear(); saved.clear();
        for (int i = 0; i < n1; i++) low.push_back(i);
        for (int i = 0; i < n0; i++) leaves_all.push_back(i);
        for (int i = n0; i < n1; i++) s1_all.push_back(i);
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
        if (p == 0 && P.rootfull) {
            // root only: every dictionary value (incl. 2-op values) as a factor: v = q*A + r / (q+1)*A - (A-r)
            for (int i = 0; i < (int)d.e.size(); i++) {
                u64 a = d.e[i].v;
                if (a < 2 || a >= v || d.e[i].cost < 2) continue;  // cost<2 entries are handled below
                u64 q = v / a, r = v % a;
                if (!r) { offer(p, q, i, -1, '+'); continue; }
                int rid = d.get(r);
                if (rid >= 0) offer(p, q, i, rid, '+');
                if (a - r <= d.L - v) { rid = d.get(a - r); if (rid >= 0) offer(p, q + 1, i, rid, '-'); }
            }
        }
        for (int i : d.low) {
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
            if (d.e[i].a < 0 && q > 64) {  // approximate the quotient by a dictionary value, then fix the error
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
        if (d.bloom.on && P.mitm) {
            // meet in the middle at the root: target = A (op) B, B from the dictionary, A in the S3 bloom
            for (int i = 0; i < (int)d.e.size(); i++) {
                const Expr &B = d.e[i];
                int c = 1 + B.cost + 3;
                if (c >= best) continue;
                u64 bv = B.v, A[3]; char ops[3]; int na = 0;
                if (target > bv) { A[na] = target - bv; ops[na++] = '+'; }       // t = A + B
                if (target + bv <= d.L) { A[na] = target + bv; ops[na++] = '-'; } // t = A - B
                if (bv > target) { A[na] = bv - target; ops[na++] = 'r'; }       // t = B - A
                for (int q = 0; q < na; q++) {
                    if (!d.bloom.has(A[q])) continue;
                    char o; int x, y;
                    if (!d.decompose3(A[q], o, x, y)) continue;
                    sols.push_back({c, (int)history.size(), -1, 3});
                    history.push_back({A[q], 1 + B.cost, 0, {-1, i, ops[q]}});
                    if (c < best) { best = c; end = history.size() - 1; term_kind = 3; term = -1; }
                    break;
                }
            }
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
        if (e.a < 0) return build_palette(e.v);
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
    u64 build_palette(u64 v) {
        if (have.count(v)) return v;
        if (v == 1) return one();
        if ((v & (v - 1)) == 0) return power(__builtin_ctzll(v));
        if (v == 11 || v == 13 || v == 19 || v == 23) return make('+', build_palette(v - (v == 23 ? 16 : 8)), build_palette(v == 23 ? 16 : 8));
        if ((v & (v + 1)) == 0 && v + 1 <= L) return make('-', build_palette(v + 1), one());
        return binary(v);
    }
    vector<int> palcost;  // virtual cost of each palette value (0 once built)
    void setup() {
        palette.clear();
        if (!P.palette) return;
        if (P.palette == 2) {  // eager (legacy): build everything now, as free leaves
            palette.push_back(one());
            for (int k : {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48})
                if ((1ULL << k) <= L) palette.push_back(power(k));
            for (u64 i : {3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 31}) if (i <= L) palette.push_back(build_palette(i));
            return;
        }
        palette.push_back(1);
        for (int k : {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48}) if ((1ULL << k) <= L) palette.push_back(1ULL << k);
        for (u64 i : {3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 31}) if (i <= L) palette.push_back(i);
    }
    int virtual_cost(u64 v) {
        if (have.count(v)) return 0;
        size_t n0 = order.size();
        build_palette(v);
        int c = order.size() - n0;
        rollback(n0);
        return c;
    }
    vector<int> leaf_costs;
    vector<u64> pref;  // preferred leaves (LNS neighbourhood)
    vector<u64> leaves() {
        vector<u64> lv; unordered_set<u64> used; leaf_costs.clear();
        auto add = [&](u64 v) { if (used.insert(v).second) { lv.push_back(v); leaf_costs.push_back(0); } };
        for (u64 v : palette) if (used.insert(v).second) { lv.push_back(v); leaf_costs.push_back(virtual_cost(v)); }
        for (u64 v : starts) add(v);
        for (u64 v : pref) if ((int)lv.size() < P.n_leaves && have.count(v)) add(v);
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
    double remaining_cost_inc(Dict &d, const vector<Node> &nodes, const vector<u64> &rem, int w) {
        Params Q = P; Q.width = w;
        auto m = d.mark();
        vector<u64> vals; for (auto &n : nodes) vals.push_back(n.v);
        d.add_leaves(vals);
        Search s(d, Q);
        double tot = 0;
        for (u64 t : rem) {
            if (have.count(t)) continue;
            s.run(t);
            tot += s.end < 0 ? 60 : s.best;
        }
        d.rollback(m);
        return tot;
    }
    double remaining_cost(Dict &d, const vector<u64> &rem, int w) {
        Params Q = P; Q.width = w;
        { auto lv_ = leaves(); d.build(lv_, leaf_costs, L); }
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
            { auto lv_ = leaves(); d.build(lv_, leaf_costs, L); }
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
                        vector<u64> rest;
                        set<u64> made; for (auto &n : cands[k].nodes) made.insert(n.v);
                        for (u64 x : remaining) if (!have.count(x) && !made.count(x)) rest.push_back(x);
                        double rc;
                        if (getenv("ELF_DBGINC")) {
                            double r1 = remaining_cost_inc(d, cands[k].nodes, rest, P.la_width);
                            size_t n0 = order.size();
                            apply(cands[k].nodes);
                            double r2 = remaining_cost(d, rest, P.la_width);
                            rollback(n0);
                            { auto lv_ = leaves(); d.build(lv_, leaf_costs, L); }
                            fprintf(stderr, "DBGINC cand=%d inc=%.0f full=%.0f rest=%zu\n", k, r1, r2, rest.size());
                            rc = r2;
                        } else if (P.la_inc) rc = remaining_cost_inc(d, cands[k].nodes, rest, P.la_width);
                        else {
                            size_t n0 = order.size();
                            apply(cands[k].nodes);
                            rc = remaining_cost(d, rest, P.la_width);
                            rollback(n0);
                        }
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
    // candidate derivations (with alternatives) for every remaining target from the current state
    vector<Cand> candidates(Dict &d, const vector<u64> &remaining) {
        { auto lv_ = leaves(); d.build(lv_, leaf_costs, L); }
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
                try { build_sol(d, s, sol, t); } catch (exception &) { rollback(n0); continue; }
                Cand c{t, {}};
                for (size_t k = n0; k < order.size(); k++) c.nodes.push_back(defs[order[k]]);
                rollback(n0);
                vector<u64> key; for (auto &n : c.nodes) key.push_back(n.v);
                if (!seen.insert(key).second) continue;
                cands.push_back(move(c)); kept++;
            }
        }
        stable_sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.nodes.size() < b.nodes.size(); });
        return cands;
    }
    void refresh_done() { done.clear(); for (u64 t : targets) if (have.count(t)) done.insert(t); }
    vector<u64> still_needed() { vector<u64> r; for (u64 t : targets) if (!have.count(t)) r.push_back(t); return r; }
    // program-level beam search over planner states
    void solve_beam(Dict &d) {
        setup();
        size_t base = order.size();
        set<u64> base_have = have; map<u64, Node> base_defs = defs;
        struct BE { vector<Node> nodes; double score; bool fin; };
        vector<BE> beam{{{}, 0.0, false}};
        auto restore = [&](const vector<Node> &nodes) {
            rollback(base); apply(nodes); refresh_done();
        };
        int guard = 0;
        while (guard++ < 200) {
            vector<BE> kids; bool any = false;
            for (auto &be : beam) {
                if (be.fin) { kids.push_back(be); continue; }
                restore(be.nodes);
                vector<u64> rem = still_needed();
                if (rem.empty()) { kids.push_back({be.nodes, (double)be.nodes.size(), true}); continue; }
                any = true;
                auto cands = candidates(d, rem);
                if (cands.empty()) {  // fallback: binary build of one target
                    size_t n0 = order.size(); binary(rem[0]);
                    vector<Node> nn = be.nodes; for (size_t k = n0; k < order.size(); k++) nn.push_back(defs[order[k]]);
                    kids.push_back({nn, (double)nn.size(), false});
                    continue;
                }
                int M = min<int>(P.lookahead, cands.size());
                for (int k = 0; k < M; k++) {
                    set<u64> made; for (auto &n : cands[k].nodes) made.insert(n.v);
                    vector<u64> rest; for (u64 x : rem) if (!made.count(x)) rest.push_back(x);
                    double h = rest.empty() ? 0 : remaining_cost_inc(d, cands[k].nodes, rest, P.la_width);
                    vector<Node> nn = be.nodes; nn.insert(nn.end(), cands[k].nodes.begin(), cands[k].nodes.end());
                    kids.push_back({nn, nn.size() + P.la_weight * h, rest.empty()});
                }
            }
            sort(kids.begin(), kids.end(), [](const BE &a, const BE &b) { return a.score < b.score; });
            vector<BE> nb; set<vector<u64>> seen;
            for (auto &k : kids) {
                vector<u64> key; for (auto &n : k.nodes) key.push_back(n.v);
                sort(key.begin(), key.end());
                if (!seen.insert(key).second) continue;
                nb.push_back(k);
                if ((int)nb.size() >= P.beam) break;
            }
            beam.swap(nb);
            if (!any) break;
            if (P.verbose > 1) fprintf(stderr, "  beam step: best score %.1f size %zu fin=%d\n", beam[0].score, beam[0].nodes.size(), (int)beam[0].fin);
        }
        // best finished state
        int bi = -1;
        for (int i = 0; i < (int)beam.size(); i++) if (beam[i].fin && (bi < 0 || beam[i].nodes.size() < beam[bi].nodes.size())) bi = i;
        if (bi < 0) bi = 0;
        restore(beam[bi].nodes);
        for (u64 t : still_needed()) binary(t);
    }
    void solve(Dict &d) {
        if (P.beam > 1) { solve_beam(d); return; }
        if (P.lookahead > 0) { solve_lookahead(d); return; }
        setup();
        vector<u64> remaining;
        for (u64 t : targets) if (!have.count(t)) remaining.push_back(t); else done.insert(t);
        uniform_real_distribution<double> U(0, 1);
        while (!remaining.empty()) {
            auto tb = chrono::steady_clock::now();
            { auto lv_ = leaves(); d.build(lv_, leaf_costs, L); }
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

static vector<Instr> compile_mode(const vector<u64> &starts, const vector<u64> &targets, const map<u64, Node> &defs,
                             const vector<u64> &order, int *plan_size, bool planorder, bool seed_evict) {
    // keep the planner's creation order (a topological order), dropping values no target needs
    set<u64> need, startset(starts.begin(), starts.end());
    function<void(u64)> mark = [&](u64 v) {
        if (startset.count(v) || !need.insert(v).second) return;
        const Node &n = defs.at(v);
        mark(n.a); mark(n.b);
    };
    for (u64 t : targets) mark(t);
    vector<u64> seq;
    if (planorder) { for (u64 v : order) if (need.count(v)) seq.push_back(v); }
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
    // next-use lists: uses[x] sorted descending (back() = soonest); entries < pos are stale.
    // Pseudo-uses are added for the arguments of an evicted live value (they will be needed to rebuild it).
    const i64 INF = 1LL << 40;
    unordered_map<u64, vector<int>> uses;
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
    // first use >= pos (strict: > pos)
    auto nextuse = [&](u64 x, bool strict) -> i64 {
        auto it = uses.find(x);
        if (it == uses.end()) return INF;
        auto &u = it->second;
        while (!u.empty() && u.back() < pos) u.pop_back();
        for (int k = (int)u.size() - 1; k >= 0; k--) if (!strict || u[k] > pos) return u[k];
        return INF;
    };
    auto add_use = [&](u64 x, int t) {
        if (seeds.count(x) && !seed_evict) return;
        auto &u = uses[x];
        auto it = lower_bound(u.begin(), u.end(), t, greater<int>());
        if (it != u.end() && *it == t) return;
        u.insert(it, t);
    };
    auto inreg = [&](u64 x) -> int { for (int k = 0; k < 48; k++) if (occ[k] && bank[k] == x) return k; return -1; };
    // rebuild cost assuming only values with a future use (or seeds) survive in registers
    function<int(u64, int)> rcost = [&](u64 x, int depth) -> int {
        int rx = inreg(x);
        if (rx >= 0 && (seeds.count(x) || nextuse(x, false) < INF)) return 0;
        if (!defs.count(x)) return rx >= 0 ? 0 : 1000;  // an evicted starting value cannot be rebuilt
        if (depth > 8) return 60;
        const Node &m = defs.at(x);
        int c = 1 + rcost(m.a, depth + 1);
        if (m.b != m.a) c += rcost(m.b, depth + 1);
        return c >= 1000 ? 1000 : min(c, 60);
    };
    static const bool trace = getenv("ELF_TRACE") != nullptr;
    set<u64> built_once;
    function<int(u64, u64)> emit = [&](u64 v, u64 locked) -> int {
        int r0 = inreg(v);
        if (r0 >= 0) return r0;
        const Node &n = defs.at(v);
        int ra = emit(n.a, locked);
        int rb = emit(n.b, locked | (1ULL << ra));
        u64 outer = locked;
        locked |= (1ULL << ra) | (1ULL << rb);
        int rd = -1;
        // a source register whose value dies here may receive the result (operands are read first)
        for (int k : {ra, rb}) {
            if (outer >> k & 1) continue;
            u64 old = bank[k];
            if (nextuse(old, true) >= INF && (!seeds.count(old) || seed_evict)) { rd = k; break; }
        }
        if (rd < 0) for (int k = 0; k < 48; k++) if (!(locked >> k & 1) && !occ[k]) { rd = k; break; }
        if (rd < 0) {
            double bestsc = -1e18;
            for (int k = 0; k < 48; k++) {
                if (locked >> k & 1) continue;
                u64 old = bank[k];
                i64 nu = nextuse(old, false);
                if (seeds.count(old) && (!seed_evict || nu < INF)) continue;
                double sc;
                if (nu >= INF) sc = 1e18;
                else if (nu <= pos) sc = -1e17;  // needed right now
                else {
                    u64 save = bank[k]; bank[k] = 0; occ[k] = false;
                    int rc = max(1, rcost(old, 0));
                    bank[k] = save; occ[k] = true;
                    if (rc >= 1000) continue;
                    sc = double(nu - pos) / rc;
                }
                if (sc > bestsc) { bestsc = sc; rd = k; }
            }
            if (rd < 0) throw runtime_error("no register");
            u64 old = bank[rd];
            i64 nu = nextuse(old, false);
            if (nu < INF && defs.count(old)) {  // evicting a live value: its inputs will be needed to rebuild it
                const Node &m = defs.at(old);
                add_use(m.a, (int)nu); add_use(m.b, (int)nu);
            }
        }
        prog.push_back({n.op, ra, rb, rd});
        if (trace && !built_once.insert(v).second)
            fprintf(stderr, "REBUILD pos=%d v=%llu\n", pos, (unsigned long long)v);
        bank[rd] = v; occ[rd] = true;
        return rd;
    };
    for (; pos < (int)seq.size(); pos++) {
        u64 v = seq[pos];
        if (inreg(v) < 0) emit(v, 0);
    }
    return prog;
}

static string verify_prog(u64 L, const vector<u64> &starts, const vector<u64> &targets, const vector<Instr> &prog);
static int g_reorder_tries = 0;
// try several schedules / seed policies, return the shortest program that verifies
static vector<Instr> compile(const vector<u64> &starts, const vector<u64> &targets, const map<u64, Node> &defs,
                             const vector<u64> &order, int *plan_size = nullptr, u64 L = 0) {
    vector<Instr> best; bool have = false;
    int psz = 0;
    for (int mode = 0; mode < 4; mode++) {
        bool po = mode & 1, se = mode & 2;
        try {
            auto p = compile_mode(starts, targets, defs, order, &psz, po, se);
            if (L && !verify_prog(L, starts, targets, p).empty()) continue;
            if (!have || p.size() < best.size()) { best = p; have = true; }
        } catch (exception &) {}
    }
    if (!have) throw runtime_error("compile failed");
    if (plan_size) *plan_size = psz;
    // spill overhead left: hill-climb over the order in which target cones are emitted
    if (g_reorder_tries > 0 && (int)best.size() > psz) {
        set<u64> tset(targets.begin(), targets.end()), sset(starts.begin(), starts.end());
        vector<u64> tord;
        for (u64 v : order) if (tset.count(v) && !sset.count(v)) tord.push_back(v);
        mt19937_64 rng(tord.size() * 7919 + best.size());
        size_t cur = best.size();
        for (int it = 0; it < g_reorder_tries && (int)cur > psz && tord.size() > 2; it++) {
            auto cand = tord;
            size_t i = rng() % cand.size(), j = rng() % cand.size();
            u64 x = cand[i]; cand.erase(cand.begin() + i); cand.insert(cand.begin() + min(j, cand.size()), x);
            for (bool se : {true, false}) {
                try {
                    auto p = compile_mode(starts, targets, defs, cand, nullptr, false, se);
                    if (p.size() > cur) continue;
                    if (L && !verify_prog(L, starts, targets, p).empty()) continue;
                    tord = cand; cur = p.size();
                    if (p.size() < best.size()) best = p;
                    break;
                } catch (exception &) {}
            }
        }
    }
    return best;
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

// parse a program JSON [["+",0,1,2],...] (no validation; verified later)
static vector<Instr> parse_prog(const string &js) {
    vector<Instr> p; size_t i = 0;
    while ((i = js.find("[\"", i)) != string::npos) {
        Instr in; in.op = js[i + 2];
        i = js.find(',', i) + 1; in.i = atoi(js.c_str() + i);
        i = js.find(',', i) + 1; in.j = atoi(js.c_str() + i);
        i = js.find(',', i) + 1; in.d = atoi(js.c_str() + i);
        p.push_back(in);
    }
    return p;
}

// LNS: remove a window of targets (with every value only they need), re-plan them, recompile.
static vector<Instr> lns_improve(const PuzzleIn &pz, vector<Instr> best, Params P, Dict &d, double seconds, int *n_acc, int *iters) {
    mt19937_64 rng(P.seed * 1000003ULL + std::hash<string>()(pz.id));
    set<u64> tset(pz.t.begin(), pz.t.end()), sset(pz.s.begin(), pz.s.end());
    // DAG of the current best program
    auto to_dag = [&](const vector<Instr> &prog, map<u64, Node> &defs, vector<u64> &order) {
        array<u64, 48> reg{}; set<u64> have(sset);
        for (size_t k = 0; k < pz.s.size(); k++) reg[k] = pz.s[k];
        for (auto &in : prog) {
            u64 a = reg[in.i], b = reg[in.j], r;
            if (in.op == '+') r = a + b; else if (in.op == '-') r = a - b; else if (in.op == '*') r = a * b; else r = a / b;
            reg[in.d] = r;
            if (!have.count(r)) { have.insert(r); defs[r] = {r, a, b, in.op}; order.push_back(r); }
        }
    };
    map<u64, Node> cdefs; vector<u64> corder;
    to_dag(best, cdefs, corder);
    size_t bestlen = best.size();
    if (getenv("ELF_RECOMPILE")) {
        auto rp = compile(pz.s, pz.t, cdefs, corder, nullptr, pz.L);
        fprintf(stderr, "RECOMPILE %s orig=%zu dag=%zu recompiled=%zu\n", pz.id.c_str(), best.size(), corder.size(), rp.size());
    }
    auto t0 = chrono::steady_clock::now();
    int acc = 0, it = 0;
    uniform_real_distribution<double> U(0, 1);
    while (chrono::duration<double>(chrono::steady_clock::now() - t0).count() < seconds) {
        it++;
        // targets in production order
        vector<u64> prod; for (u64 v : corder) if (tset.count(v)) prod.push_back(v);
        if (prod.empty()) break;
        int kmax = min<int>(P.lns_kmax, prod.size());
        int k = 1 + rng() % kmax;
        int st = rng() % (prod.size() - k + 1);
        set<u64> S(prod.begin() + st, prod.begin() + st + k);
        // keep everything the other targets need
        set<u64> need;
        function<void(u64)> mark = [&](u64 v) {
            if (sset.count(v) || !need.insert(v).second) return;
            const Node &n = cdefs.at(v); mark(n.a); mark(n.b);
        };
        for (u64 t : pz.t) if (!S.count(t) && !sset.count(t)) mark(t);
        Planner pl(pz.L, pz.s, pz.t, P);
        size_t wpos = 0, cnt = 0;
        for (u64 v : corder) {
            cnt++;
            if (need.count(v)) { pl.have.insert(v); pl.defs[v] = cdefs.at(v); pl.order.push_back(v); }
            if (v == *S.begin() || (S.count(v) && wpos == 0)) wpos = cnt;
        }
        // preferred leaves: kept values used by the removed cone, then by distance to the window
        {
            set<u64> pf;
            for (u64 v : corder) if (!need.count(v)) { const Node &n = cdefs.at(v); for (u64 x : {n.a, n.b}) if (need.count(x) && pf.insert(x).second) pl.pref.push_back(x); }
            vector<pair<long long, u64>> byd; long long idx = 0;
            for (u64 v : corder) { idx++; if (need.count(v) && !pf.count(v)) byd.push_back({llabs(idx - (long long)wpos), v}); }
            sort(byd.begin(), byd.end());
            for (auto &q : byd) pl.pref.push_back(q.second);
        }
        if (P.noise <= 0) pl.P.noise = 0.5;
        pl.rng.seed(rng());
        try {
            pl.solve(d);
            auto prog = compile(pz.s, pz.t, pl.defs, pl.order, nullptr, pz.L);
            if (prog.size() < bestlen && verify_prog(pz.L, pz.s, pz.t, prog).empty()) {
                best = prog; bestlen = prog.size(); acc++;
                cdefs.clear(); corder.clear(); to_dag(best, cdefs, corder);
                if (P.verbose) fprintf(stderr, "  %s iter %d k=%d -> %zu\n", pz.id.c_str(), it, k, bestlen);
            }
        } catch (exception &) {}
    }
    *n_acc = acc; *iters = it;
    return best;
}

int main(int argc, char **argv) {
    Params P; string inpath, lnspath, dagpath; double lns_sec = 10;
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
        else if (a == "-linc") P.la_inc = stoi(nx());
        else if (a == "-nd") P.n_done = stoi(nx());
        else if (a == "-lns") lnspath = nx();
        else if (a == "-dag") dagpath = nx();
        else if (a == "-lsec") lns_sec = stod(nx());
        else if (a == "-lk") P.lns_kmax = stoi(nx());
        else if (a == "-mitm") P.mitm = stoi(nx());
        else if (a == "-rf") P.rootfull = stoi(nx());
        else if (a == "-B") P.beam = stoi(nx());
        else if (a == "-cap") P.dict_cap = stoll(nx());
        else if (a == "-reorder") g_reorder_tries = stoi(nx());
        else inpath = a;
    }
    map<string, pair<map<u64, Node>, vector<u64>>> dagin;
    if (!dagpath.empty()) {  // DAG file: "id n" then n lines "v op a b" in topological order
        ifstream f(dagpath); string id; int n;
        while (f >> id >> n) {
            auto &e = dagin[id];
            for (int i = 0; i < n; i++) {
                unsigned long long v, a, b; string op; f >> v >> op >> a >> b;
                e.first[v] = {v, a, b, op[0]}; e.second.push_back(v);
            }
        }
    }
    map<string, vector<Instr>> lnsin;
    if (!lnspath.empty()) {
        ifstream f(lnspath); string line;
        while (getline(f, line)) { size_t sp = line.find(' '); if (sp == string::npos) continue; lnsin[line.substr(0, sp)] = parse_prog(line.substr(sp + 1)); }
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
        Dict d; d.bloom.init(P.bloom_lg); d.cap_entries = P.dict_cap;
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            auto &pz = puzzles[k];
            string res;
            try {
                if (!dagpath.empty()) {
                    if (!dagin.count(pz.id)) continue;
                    auto &dg = dagin[pz.id];
                    int psz = 0;
                    auto prog = compile(pz.s, pz.t, dg.first, dg.second, &psz, pz.L);
                    string why = verify_prog(pz.L, pz.s, pz.t, prog);
                    if (!why.empty()) throw runtime_error("verify failed: " + why);
                    res = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);
                    lock_guard<mutex> g(mu); total += prog.size();
                    fprintf(stderr, "%s dag=%d prog=%zu\n", pz.id.c_str(), psz, prog.size());
                    out[k] = res;
                    continue;
                }
                if (!lnspath.empty()) {
                    if (!lnsin.count(pz.id)) continue;
                    Params Q = P; Q.palette = 1;
                    int acc = 0, iters = 0;
                    auto &b0 = lnsin[pz.id];
                    size_t L0 = b0.size();
                    auto prog = lns_improve(pz, b0, Q, d, lns_sec, &acc, &iters);
                    string why = verify_prog(pz.L, pz.s, pz.t, prog);
                    if (!why.empty()) throw runtime_error("verify failed: " + why);
                    res = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);
                    lock_guard<mutex> g(mu); total += prog.size();
                    fprintf(stderr, "%s %zu -> %zu (acc %d / %d iters)\n", pz.id.c_str(), L0, prog.size(), acc, iters);
                    out[k] = res;
                    continue;
                }
                Planner pl(pz.L, pz.s, pz.t, P);
                pl.solve(d);
                if (P.verbose) fprintf(stderr, "  time build=%.2fs search=%.2fs dict=%zu n1=%d\n", pl.t_build, pl.t_search, d.e.size(), d.n1);
                int psz = 0;
                auto prog = compile(pz.s, pz.t, pl.defs, pl.order, &psz, pz.L);
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
