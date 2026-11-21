// elf2.cpp — Santa 2026 next-generation deep-search + register-aware DAG-LNS engine.
//
// Five search engines per target (with exact DAG-node deduplication):
//   1. Exact depth <= 4 MITM over S0/S1/S2
//   2. Product-difference T = A*B +- C: S01 x S012 (binary search) + S2 x S2 (two-pointer scan)
//   3. 64-bit Pollard-Rho factoring of T +- C (with 3-factor + S3-cofactor checks)
//   4. Wrappers: target-difference T = U +- D (over done_targets) and exact-division T = N / k
//   5. Product-residual multi-step beam (depth <= 6)
//
// Two modes:
//   -lns progs.txt  : Deep DAG-LNS (k=1, k=2..4 joint w/ branching lookahead, helper-cut, pair-hub)
//                     + STOKE-style neutral moves & periodic kicks + peak-liveness register scheduler
//   -construct 1    : Scale-aware Hcub-style constructive planner (optional -lsec LNS polish)
//
// Build (Windows): .venv/Scripts/python.exe -m ziglang c++ -O3 -march=native -std=c++20 cpp/elf2.cpp -o elf2.exe
// Build (Linux)  : g++ -O3 -march=native -std=c++20 -pthread cpp/elf2.cpp -o elf2
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
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using u64 = uint64_t;
using i64 = int64_t;
using u128 = unsigned __int128;

// ---------------------------------------------------------------- Parameters & Timing
struct Params {
    int threads = 8;
    int n_leaves = 72;
    int s2_leaves = 40;
    size_t s2_cap = 300000;
    int fac_k = 48;
    int beam_w = 24;
    double alpha = 0.18;
    int construct = 0;
    int lookahead = 10;
    double la_weight = 0.35;
    int la_alts = 4;
    double lns_sec = -1.0;
    int lns_kmax = 4;
    double eq_prob = 0.20;
    int kick_stall = 60;
    int reorder_tries = 32;
    int seed = 1;
    int verbose = 0;
    string lns_path;
    string out_path;
};

static double now_sec() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

static inline u64 mix64(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline u64 isqrt64(u64 n) {
    u64 s = (u64)sqrtl((long double)n);
    while ((u128)(s + 1) * (s + 1) <= n) s++;
    while ((u128)s * s > n) s--;
    return s;
}

// ---------------------------------------------------------------- Safe Arithmetic & Verification
struct Step { char op; u64 a, b, r; };
struct Instr { char op; int i, j, d; };

static inline bool apply_op(char op, u64 a, u64 b, u64 L, u64 &r) {
    if (a == 0 || b == 0) return false;
    switch (op) {
        case '+':
            if (a > L || b > L - a) return false;
            r = a + b;
            return r > 0 && r <= L;
        case '-':
            if (a <= b) return false;
            r = a - b;
            return r <= L;
        case '*': {
            u128 p = (u128)a * b;
            if (p == 0 || p > (u128)L) return false;
            r = (u64)p;
            return true;
        }
        case '/':
            if (a % b != 0) return false;
            r = a / b;
            return r > 0 && r <= L;
    }
    return false;
}

struct Inv { u64 w; char op; int side; };  // side 0: T = w op x; side 1: T = x op w
static inline int inverses(u64 T, u64 x, u64 L, Inv *out) {
    int n = 0;
    if (T > x && T - x <= L) out[n++] = {T - x, '+', 0};
    if (x <= L - T) out[n++] = {T + x, '-', 0};
    if (x > T && x - T <= L) out[n++] = {x - T, '-', 1};
    if (x > 1 && T % x == 0 && T / x <= L) out[n++] = {T / x, '*', 0};
    if (x > 1 && (u128)T * x <= (u128)L) out[n++] = {T * x, '/', 0};
    if (x > T && x % T == 0 && x / T <= L) out[n++] = {x / T, '/', 1};
    return n;
}

static inline Step mk_inv_step(const Inv &iv, u64 x, u64 T) {
    return iv.side == 0 ? Step{iv.op, iv.w, x, T} : Step{iv.op, x, iv.w, T};
}

static string to_json(const vector<Instr> &p) {
    string s;
    s.reserve(p.size() * 14 + 4);
    s.push_back('[');
    for (size_t k = 0; k < p.size(); k++) {
        if (k) s.push_back(',');
        s += "[\"";
        s.push_back(p[k].op);
        s += "\"," + to_string(p[k].i) + "," + to_string(p[k].j) + "," + to_string(p[k].d) + "]";
    }
    s.push_back(']');
    return s;
}

static string verify_prog(u64 L, const vector<u64> &starts, const vector<u64> &targets, const vector<Instr> &prog) {
    if (prog.size() > 4096) return "too many instructions";
    array<u64, 48> reg{};
    array<bool, 48> def{};
    unordered_set<u64> seen(starts.begin(), starts.end());
    for (size_t k = 0; k < starts.size(); k++) {
        if (k >= 48) return "too many starts";
        reg[k] = starts[k];
        def[k] = true;
    }
    for (size_t step = 0; step < prog.size(); step++) {
        const auto &in = prog[step];
        if (in.i < 0 || in.i > 47 || in.j < 0 || in.j > 47 || in.d < 0 || in.d > 47)
            return "bad register at step " + to_string(step);
        if (!def[in.i] || !def[in.j])
            return "undefined source at step " + to_string(step);
        u64 r = 0;
        if (!apply_op(in.op, reg[in.i], reg[in.j], L, r))
            return "illegal op at step " + to_string(step);
        reg[in.d] = r;
        def[in.d] = true;
        seen.insert(r);
    }
    for (u64 t : targets) {
        if (!seen.count(t)) return "missing target " + to_string(t);
    }
    if (to_json(prog).size() > 131072) return "json > 131072 bytes";
    return "";
}

// ---------------------------------------------------------------- Fast 64-bit Factoring
static inline u64 bgcd(u64 a, u64 b) {
    if (!a) return b;
    if (!b) return a;
    int sh = __builtin_ctzll(a | b);
    a >>= __builtin_ctzll(a);
    do {
        b >>= __builtin_ctzll(b);
        if (a > b) swap(a, b);
        b -= a;
    } while (b);
    return a << sh;
}

static inline u64 mul_mod(u64 a, u64 b, u64 m) {
    return (u64)((u128)a * b % m);
}

static u64 pow_mod(u64 a, u64 d, u64 m) {
    u64 r = 1;
    a %= m;
    while (d) {
        if (d & 1) r = mul_mod(r, a, m);
        a = mul_mod(a, a, m);
        d >>= 1;
    }
    return r;
}

static bool is_prime64(u64 n) {
    if (n < 2) return false;
    static const u64 sprimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 p : sprimes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    u64 d = n - 1;
    int s = __builtin_ctzll(d);
    d >>= s;
    static const u64 bases[] = {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};
    for (u64 a : bases) {
        if (a % n == 0) continue;
        u64 x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s; r++) {
            x = mul_mod(x, x, n);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

static u64 pollard_brent(u64 n, mt19937_64 &rng) {
    if ((n & 1ULL) == 0) return 2;
    u64 c = 1 + (rng() % (n - 1));
    u64 y = 1 + (rng() % (n - 1));
    u64 m = 64, g = 1, r = 1, q = 1;
    u64 x = y, ys = y;
    int steps = 0;
    const int MAX_STEPS = 2048;
    while (g == 1 && steps < MAX_STEPS) {
        x = y;
        for (u64 i = 0; i < r; i++) {
            y = (mul_mod(y, y, n) + c) % n;
            steps++;
        }
        u64 k = 0;
        while (k < r && g == 1 && steps < MAX_STEPS) {
            ys = y;
            u64 lim = min(m, r - k);
            for (u64 i = 0; i < lim; i++) {
                y = (mul_mod(y, y, n) + c) % n;
                u64 diff = x > y ? x - y : y - x;
                q = mul_mod(q, diff, n);
                steps++;
            }
            g = bgcd(q, n);
            k += m;
        }
        r <<= 1;
    }
    if (g == n) {
        do {
            ys = (mul_mod(ys, ys, n) + c) % n;
            u64 diff = x > ys ? x - ys : ys - x;
            g = bgcd(diff, n);
            steps++;
        } while (g == 1 && steps < MAX_STEPS + 256);
    }
    if (g > 1 && g < n) return g;
    return 0;  // iteration cap reached
}

static void factor_rec(u64 n, vector<u64> &primes, mt19937_64 &rng) {
    if (n <= 1) return;
    if (is_prime64(n)) { primes.push_back(n); return; }
    u64 d = pollard_brent(n, rng);
    if (d == 0 || d == n) { primes.push_back(n); return; }  // unsplit fallback
    factor_rec(d, primes, rng);
    factor_rec(n / d, primes, rng);
}

static void factor64(u64 n, vector<pair<u64, int>> &pf, mt19937_64 &rng) {
    pf.clear();
    if (n <= 1) return;
    static const u64 small_p[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    };
    for (u64 p : small_p) {
        if (p * p > n) break;
        if (n % p == 0) {
            int e = 0;
            while (n % p == 0) { n /= p; e++; }
            pf.push_back({p, e});
        }
    }
    if (n > 1) {
        vector<u64> rem;
        factor_rec(n, rem, rng);
        sort(rem.begin(), rem.end());
        for (u64 p : rem) {
            if (!pf.empty() && pf.back().first == p) pf.back().second++;
            else pf.push_back({p, 1});
        }
    }
    sort(pf.begin(), pf.end());
}

static void divisors_bounded(const vector<pair<u64, int>> &pf, u64 max_d, vector<u64> &out, size_t max_count = 512) {
    out.clear();
    out.push_back(1);
    for (auto [p, e] : pf) {
        if (p > max_d) break;
        size_t cur_sz = out.size();
        for (size_t idx = 0; idx < cur_sz; idx++) {
            u64 base = out[idx];
            u128 v = base;
            for (int k = 1; k <= e; k++) {
                v *= p;
                if (v > (u128)max_d) break;
                out.push_back((u64)v);
                if (out.size() >= max_count) return;
            }
        }
    }
}

// ---------------------------------------------------------------- Hash Index & PoolDict
struct Index {
    struct Slot { u64 key; int id; };
    vector<Slot> tab;
    vector<u64> used;
    u64 mask = 0;
    size_t cap = 0;
    int bits = 0;
    static const int FBITS = 25;  // 4 MB bitfilter
    vector<u64> filt = vector<u64>((1ULL << FBITS) / 64, 0);

    static inline u64 fpos(u64 v) { return (mix64(v) >> 20) & ((1ULL << FBITS) - 1); }

    void init(int b) {
        bits = b;
        tab.assign(1ULL << b, {0, -1});
        mask = (1ULL << b) - 1;
        cap = (size_t)((1ULL << b) * 0.6);
        used.clear();
        fill(filt.begin(), filt.end(), 0);
    }

    void ensure(size_t expected) {
        int b = 16;
        while ((1ULL << b) * 0.5 < expected && b < 22) b++;
        if (b != bits) init(b);
    }

    inline int get(u64 v) const {
        if (v == 0) return -1;
        u64 f = fpos(v);
        if (!((filt[f >> 6] >> (f & 63)) & 1)) return -1;
        u64 i = mix64(v) & mask;
        while (tab[i].key) {
            if (tab[i].key == v) return tab[i].id;
            i = (i + 1) & mask;
        }
        return -1;
    }

    inline int find_or_insert(u64 v, int id) {
        if (used.size() >= cap) {
            int g = get(v);
            return g >= 0 ? g : -2;
        }
        u64 f = fpos(v);
        bool maybe = (filt[f >> 6] >> (f & 63)) & 1;
        u64 i = mix64(v) & mask;
        if (maybe) {
            while (tab[i].key) {
                if (tab[i].key == v) return tab[i].id;
                i = (i + 1) & mask;
            }
        } else {
            while (tab[i].key) i = (i + 1) & mask;
        }
        tab[i] = {v, id};
        used.push_back(i);
        filt[f >> 6] |= 1ULL << (f & 63);
        return -1;
    }

    inline void clear() {
        for (u64 i : used) {
            u64 f = fpos(tab[i].key);
            filt[f >> 6] = 0;
            tab[i].key = 0;
        }
        used.clear();
    }
};

struct Expr { u64 v; int a, b; char op; uint8_t cost; };

struct PoolDict {
    u64 L = 0;
    vector<u64> starts;
    Index index;      // value -> id in e
    Index pool_idx;   // value -> 0 for all actually-built values in current pool (incl. done_targets)
    vector<Expr> e;
    int n0 = 0, n1 = 0, n2 = 0;
    size_t s2_cap = 450000;
    vector<pair<u64, int>> sorted01, sorted012;
    vector<pair<u64, int>> extra01, extra012;  // incremental entries during branching (O(1) rollback!)
    vector<u64> done_targets;
    u64 max_v = 0;

    // Incremental add/rollback state for k>=2 branching lookahead
    bool inc = false;
    vector<int> touched;
    vector<int> leaves_all, s1_all;
    vector<pair<int, Expr>> saved;
    struct Mark { size_t e, used, leaves, s1, saved; u64 max_v; };

    PoolDict() {
        index.init(20);
        pool_idx.init(14);
    }

    inline int get(u64 v) const { return index.get(v); }
    inline bool in_pool(u64 v) const { return pool_idx.get(v) >= 0; }

    inline void add_expr(u64 v, int a, int b, char op, int cost) {
        if (v == 0 || v > L || cost > 2 || (s2_cap && e.size() >= s2_cap)) return;
        int id = index.find_or_insert(v, (int)e.size());
        if (id == -1) {
            e.push_back({v, a, b, op, (uint8_t)cost});
            if (inc) touched.push_back((int)e.size() - 1);
        } else if (id >= 0 && cost < e[id].cost && e[id].a >= 0) {
            if (inc) { saved.push_back({id, e[id]}); touched.push_back(id); }
            e[id] = {v, a, b, op, (uint8_t)cost};
        }
    }

    inline void combine(int i, int j) {
        u64 a = e[i].v, b = e[j].v;
        int cost = 1 + e[i].cost + (i == j ? 0 : e[j].cost);
        if (cost > 2) return;
        if (a <= L - b) add_expr(a + b, i, j, '+', cost);
        if (a > b) add_expr(a - b, i, j, '-', cost);
        else if (b > a) add_expr(b - a, j, i, '-', cost);
        if ((u128)a * b <= (u128)L) add_expr(a * b, i, j, '*', cost);
        if (a % b == 0) add_expr(a / b, i, j, '/', cost);
        if (a != b && b % a == 0) add_expr(b / a, j, i, '/', cost);
    }

    void build(u64 limit, const vector<u64> &st, const vector<u64> &all_built,
               const vector<u64> &leaves, const vector<int> &lcost,
               const vector<u64> &done_t, int s2_leaves, size_t cap_entries) {
        L = limit;
        starts = st;
        s2_cap = cap_entries;
        index.clear();
        pool_idx.clear();
        e.clear();
        sorted01.clear();
        sorted012.clear();
        extra01.clear();
        extra012.clear();
        leaves_all.clear();
        s1_all.clear();
        saved.clear();
        inc = false;

        pool_idx.ensure(all_built.size() + 64);
        for (u64 v : all_built) pool_idx.find_or_insert(v, 0);

        done_targets = done_t;
        sort(done_targets.begin(), done_targets.end());
        done_targets.erase(unique(done_targets.begin(), done_targets.end()), done_targets.end());

        size_t est = leaves.size() * leaves.size() * 4 + 4096;
        if (s2_cap && est < s2_cap + 16384) est = s2_cap + 16384;
        index.ensure(est);

        for (size_t k = 0; k < leaves.size(); k++) {
            if (leaves[k] == 0 || leaves[k] > L) continue;
            int id = index.find_or_insert(leaves[k], (int)e.size());
            if (id == -1) e.push_back({leaves[k], -1, -1, 0, (uint8_t)lcost[k]});
        }
        n0 = (int)e.size();
        for (int i = 0; i < n0; i++) leaves_all.push_back(i);

        for (int i = 0; i < n0; i++)
            for (int j = 0; j <= i; j++)
                combine(i, j);
        n1 = (int)e.size();
        for (int i = n0; i < n1; i++) s1_all.push_back(i);

        // S1[i] op S1[i] (2*x, x*x)
        for (int i = n0; i < n1; i++) {
            if (s2_cap && e.size() >= s2_cap) break;
            combine(i, i);
        }
        // Prioritized leaves outer loop j=0..k2-1 so every S1 combines with top leaves before cap
        int k2 = min(n0, s2_leaves);
        for (int j = 0; j < k2; j++) {
            if (s2_cap && e.size() >= s2_cap) break;
            for (int i = n0; i < n1; i++) {
                if (s2_cap && e.size() >= s2_cap) break;
                combine(i, j);
            }
        }
        n2 = (int)e.size();

        sorted01.reserve(n1);
        for (int i = 0; i < n1; i++) sorted01.emplace_back(e[i].v, i);
        sort(sorted01.begin(), sorted01.end());

        sorted012.reserve(n2);
        for (int i = 0; i < n2; i++) sorted012.emplace_back(e[i].v, i);
        sort(sorted012.begin(), sorted012.end());
        max_v = sorted012.empty() ? 0 : sorted012.back().first;
    }

    Mark mark() const {
        return {e.size(), index.used.size(), leaves_all.size(), s1_all.size(), saved.size(), max_v};
    }

    void add_leaves_inc(const vector<u64> &vals) {
        inc = true;
        vector<int> nl;
        for (u64 v : vals) {
            if (v == 0 || v > L) continue;
            int id = get(v);
            if (id >= 0) {
                if (e[id].a < 0 && e[id].cost == 0) continue;
                saved.push_back({id, e[id]});
                e[id] = {v, -1, -1, 0, 0};
                nl.push_back(id);
            } else if (!s2_cap || e.size() < s2_cap + 16384) {
                int nid = (int)e.size();
                if (index.find_or_insert(v, nid) == -1) {
                    e.push_back({v, -1, -1, 0, 0});
                    nl.push_back(nid);
                }
            }
        }
        touched.clear();
        for (size_t x = 0; x < nl.size(); x++) {
            for (int j : leaves_all) combine(nl[x], j);
            for (size_t y = 0; y <= x; y++) combine(nl[x], nl[y]);
        }
        vector<int> news1 = touched;
        for (int id : nl) leaves_all.push_back(id);
        touched.clear();
        for (int k : news1) {
            for (int j : leaves_all) {
                if (s2_cap && e.size() >= s2_cap + 16384) break;
                combine(k, j);
            }
            combine(k, k);
        }
        for (int k : s1_all) {
            for (int j : nl) {
                if (s2_cap && e.size() >= s2_cap + 16384) break;
                combine(k, j);
            }
        }
        for (int k : news1) s1_all.push_back(k);

        for (int id : nl) {
            extra01.emplace_back(e[id].v, id);
            extra012.emplace_back(e[id].v, id);
            max_v = max(max_v, e[id].v);
        }
        for (int id : news1) {
            if (e[id].cost <= 1) extra01.emplace_back(e[id].v, id);
            extra012.emplace_back(e[id].v, id);
            max_v = max(max_v, e[id].v);
        }
        for (int id : touched) {
            extra012.emplace_back(e[id].v, id);
            max_v = max(max_v, e[id].v);
        }
        sort(extra01.begin(), extra01.end());
        sort(extra012.begin(), extra012.end());
        inc = false;
    }

    void rollback(const Mark &m) {
        // Leave filt bits set (false positives harmless; avoids clearing shared bits!)
        for (size_t k = m.used; k < index.used.size(); k++) {
            index.tab[index.used[k]].key = 0;
        }
        index.used.resize(m.used);
        while (saved.size() > m.saved) {
            auto pr = saved.back();
            saved.pop_back();
            if (pr.first < (int)m.e) e[pr.first] = pr.second;
        }
        e.resize(m.e);
        leaves_all.resize(m.leaves);
        s1_all.resize(m.s1);
        extra01.clear();
        extra012.clear();
        max_v = m.max_v;
    }
};

// Select up to n_leaves prioritizing starts, 1, window-adjacent, fan-out, and mid-scale [L^0.2, L^0.65].
// Everything else (especially large > L^0.78 targets) goes to done_t for Engine 4 target-diff.
static void select_leaves(
    u64 L, const vector<u64> &starts, const vector<u64> &pool,
    const unordered_set<u64> &target_set, const unordered_map<u64, int> &fanout,
    const unordered_set<u64> &near_vals, int n_leaves,
    vector<u64> &out_leaves, vector<int> &out_costs, vector<u64> &out_done_t,
    bool virtual_one) {
    out_leaves.clear();
    out_costs.clear();
    out_done_t.clear();
    unordered_set<u64> seen;
    double logL = log((double)max<u64>(L, 2));
    u64 hi_cut = (u64)pow((double)L, 0.78);
    u64 tgt_hi = (u64)pow((double)L, 0.65);

    struct Scored { double sc; u64 v; };
    vector<Scored> cand;
    for (u64 s : starts) {
        if (seen.insert(s).second) cand.push_back({1e9 + (double)(100000 - min<u64>(s, 99999)), s});
    }
    bool have_one = seen.count(1);
    for (u64 v : pool) {
        if (!seen.insert(v).second) continue;
        if (v == 1) { have_one = true; cand.push_back({5e8, 1}); continue; }
        bool is_tgt = target_set.count(v) > 0;
        if (v > hi_cut || (is_tgt && v > tgt_hi)) {
            out_done_t.push_back(v);
            continue;
        }
        double frac = log((double)max<u64>(v, 1)) / logL;
        double sc = 0.0;
        if (near_vals.count(v)) sc += 2000.0;
        auto it = fanout.find(v);
        if (it != fanout.end()) sc += 120.0 * min(8, it->second);
        if (frac >= 0.20 && frac <= 0.62) sc += 300.0 * (1.0 - fabs(frac - 0.44) / 0.28);
        else if (frac < 0.20) sc += 100.0;
        else sc += 20.0;
        if (is_tgt) sc -= 30.0;
        cand.push_back({sc, v});
    }
    sort(cand.begin(), cand.end(), [](const Scored &a, const Scored &b) {
        return a.sc != b.sc ? a.sc > b.sc : a.v < b.v;
    });
    if (!have_one && virtual_one) {
        // Place virtual 1 right after starts
        size_t pos = min(starts.size(), cand.size());
        cand.insert(cand.begin() + pos, {5e8, 1});
    }
    for (size_t i = 0; i < cand.size(); i++) {
        if ((int)out_leaves.size() < n_leaves) {
            out_leaves.push_back(cand[i].v);
            out_costs.push_back((cand[i].v == 1 && !have_one && virtual_one) ? 1 : 0);
        } else {
            out_done_t.push_back(cand[i].v);
        }
    }
}

// ---------------------------------------------------------------- Step Reconstruction & Dedup
struct LocalBuf {
    u64 vals[64];
    int n = 0;
    inline bool has(u64 v) const {
        for (int i = 0; i < n; i++) if (vals[i] == v) return true;
        return false;
    }
    inline void add(u64 v) {
        if (n < 64 && !has(v)) vals[n++] = v;
    }
};

static bool emit_dict_rec(const PoolDict &d, int id, const unordered_set<u64> &extra_pool,
                          LocalBuf &loc, vector<Step> &out) {
    u64 v = d.e[id].v;
    if (d.in_pool(v) || extra_pool.count(v) || loc.has(v)) return true;
    if (d.e[id].a < 0) {
        if (v == 1 && !d.starts.empty()) {
            loc.add(1);
            out.push_back({'/', d.starts[0], d.starts[0], 1});
            return true;
        }
        return false;
    }
    if (!emit_dict_rec(d, d.e[id].a, extra_pool, loc, out)) return false;
    if (!emit_dict_rec(d, d.e[id].b, extra_pool, loc, out)) return false;
    if (!loc.has(v)) {
        loc.add(v);
        out.push_back({d.e[id].op, d.e[d.e[id].a].v, d.e[d.e[id].b].v, v});
    }
    return true;
}

static bool finalize_steps(const PoolDict &d, const unordered_set<u64> &extra_pool,
                           const int *dict_ids, int ndids,
                           const Step *tail, int ntail,
                           u64 target, vector<Step> &out) {
    out.clear();
    LocalBuf loc;
    for (int i = 0; i < ndids; i++) {
        if (dict_ids[i] < 0) continue;
        if (!emit_dict_rec(d, dict_ids[i], extra_pool, loc, out)) return false;
    }
    for (int i = 0; i < ntail; i++) {
        const Step &s = tail[i];
        if (d.in_pool(s.r) || extra_pool.count(s.r) || loc.has(s.r)) continue;
        u64 chk = 0;
        if (!apply_op(s.op, s.a, s.b, d.L, chk) || chk != s.r) return false;
        loc.add(s.r);
        out.push_back(s);
    }
    if (!d.in_pool(target) && !extra_pool.count(target)) {
        if (out.empty() || out.back().r != target) {
            if (!loc.has(target)) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------- Five-Engine DeepSearch
struct SearchStats {
    u64 calls = 0;
    array<u64, 6> hits{};      // hits[1..5] per engine
    array<u64, 6> wins{};      // winning improvements in LNS per engine
    array<double, 6> sec{};    // time spent in each engine (and sec[0] = PoolDict build)
};

struct CandSol {
    int cost;
    int engine;
    vector<Step> steps;
};

struct DeepSearch {
    const PoolDict &d;
    const Params &P;
    mt19937_64 &rng;
    unordered_set<u64> extra_pool;  // temporary values added by earlier targets in a joint k>=2 move

    // Scratch buffers
    vector<pair<u64, int>> pf;
    vector<u64> divs;
    vector<pair<u64, int>> h_divs;

    DeepSearch(const PoolDict &dict, const Params &par, mt19937_64 &r) : d(dict), P(par), rng(r) {}

    struct Ctx {
        u64 T;
        int hard_cap;
        int max_alts;
        int slack;
        int best_found = 1 << 20;
        vector<CandSol> sols;
        inline int eff_cap() const {
            if (best_found > hard_cap) return hard_cap;
            if (max_alts == 1 && slack == 0) return best_found - 1;
            return min(hard_cap, best_found + slack);
        }
    };

    void offer_steps(Ctx &ctx, int engine, vector<Step> &steps) {
        int c = (int)steps.size();
        if (c > ctx.eff_cap() || c > ctx.hard_cap) return;
        // Dedup by final sequence values
        for (const auto &ex : ctx.sols) {
            if (ex.cost == c && ex.steps.size() == steps.size()) {
                bool same = true;
                for (size_t i = 0; i < steps.size(); i++) {
                    if (ex.steps[i].r != steps[i].r) { same = false; break; }
                }
                if (same) return;
            }
        }
        ctx.sols.push_back({c, engine, move(steps)});
        if (c < ctx.best_found) ctx.best_found = c;
    }

    void offer_combo(Ctx &ctx, int engine, const int *dids, int ndids, const Step *tail, int ntail) {
        vector<Step> steps;
        if (!finalize_steps(d, extra_pool, dids, ndids, tail, ntail, ctx.T, steps)) return;
        offer_steps(ctx, engine, steps);
    }

    // Engine 1: Exact depth <= 4 MITM
    void run_exact_le4(Ctx &ctx, bool depth4) {
        u64 T = ctx.T;
        if (d.in_pool(T) || extra_pool.count(T)) {
            vector<Step> empty;
            offer_steps(ctx, 1, empty);
            return;
        }
        int id = d.get(T);
        if (id >= 0 && d.e[id].cost <= ctx.eff_cap()) {
            offer_combo(ctx, 1, &id, 1, nullptr, 0);
            if (ctx.eff_cap() <= 1) return;
        }
        if (ctx.eff_cap() < 2) return;

        Inv inv[6], inv2[6];
        // Cost 2..3: T = A op B with B in S0 u S1, A in S0..S2
        for (int bi = 0; bi < d.n1; bi++) {
            int cb = d.e[bi].cost;
            if (cb + 1 > ctx.eff_cap()) continue;
            u64 B = d.e[bi].v;
            int k = inverses(T, B, d.L, inv);
            int maxA = min(2, ctx.eff_cap() - 1);
            for (int q = 0; q < k; q++) {
                int ai = d.get(inv[q].w);
                if (ai >= 0 && d.e[ai].cost <= maxA) {
                    if (d.e[ai].cost + cb + 1 - 2 <= ctx.eff_cap()) {
                        int ids[2] = {ai, bi};
                        Step st = mk_inv_step(inv[q], B, T);
                        offer_combo(ctx, 1, ids, 2, &st, 1);
                    }
                }
            }
        }
        // T = x + x or x * x for x in S0..S2
        if (T % 2 == 0) {
            int ai = d.get(T / 2);
            if (ai >= 0 && d.e[ai].cost + 1 <= ctx.eff_cap()) {
                Step st{'+', T / 2, T / 2, T};
                offer_combo(ctx, 1, &ai, 1, &st, 1);
            }
        }
        {
            u64 s = isqrt64(T);
            if (s > 1 && s * s == T) {
                int ai = d.get(s);
                if (ai >= 0 && d.e[ai].cost + 1 <= ctx.eff_cap()) {
                    Step st{'*', s, s, T};
                    offer_combo(ctx, 1, &ai, 1, &st, 1);
                }
            }
        }
        if (!depth4 || ctx.eff_cap() < 4) return;

        // Cost 4: T = (A op1 B) op2 C with B, C in S0, A in S0..S2
        for (int ci = 0; ci < d.n0; ci++) {
            if (ctx.eff_cap() < 4) break;
            u64 C = d.e[ci].v;
            int k1 = inverses(T, C, d.L, inv);
            for (int q1 = 0; q1 < k1; q1++) {
                u64 w1 = inv[q1].w;
                for (int bi = 0; bi < d.n0; bi++) {
                    u64 B = d.e[bi].v;
                    int k2 = inverses(w1, B, d.L, inv2);
                    for (int q2 = 0; q2 < k2; q2++) {
                        int ai = d.get(inv2[q2].w);
                        if (ai >= 0) {
                            int ids[3] = {ai, bi, ci};
                            Step tail[2] = {mk_inv_step(inv2[q2], B, w1), mk_inv_step(inv[q1], C, T)};
                            offer_combo(ctx, 1, ids, 3, tail, 2);
                        }
                    }
                }
                if (w1 % 2 == 0) {
                    int ai = d.get(w1 / 2);
                    if (ai >= 0) {
                        int ids[2] = {ai, ci};
                        Step tail[2] = {{'+', w1 / 2, w1 / 2, w1}, mk_inv_step(inv[q1], C, T)};
                        offer_combo(ctx, 1, ids, 2, tail, 2);
                    }
                }
                u64 s1 = isqrt64(w1);
                if (s1 > 1 && s1 * s1 == w1) {
                    int ai = d.get(s1);
                    if (ai >= 0) {
                        int ids[2] = {ai, ci};
                        Step tail[2] = {{'*', s1, s1, w1}, mk_inv_step(inv[q1], C, T)};
                        offer_combo(ctx, 1, ids, 2, tail, 2);
                    }
                }
            }
        }
    }

    // Engine 2: Product-difference T = A*B +- C (no C < A restriction!)
    void run_prod_diff(Ctx &ctx, bool do_s2_twoptr) {
        u64 T = ctx.T;
        if (ctx.eff_cap() < 2 || d.sorted012.empty()) return;

        // 2a: A in S01 (and extra01), B in S012 (and extra012)
        const auto &S012 = d.sorted012;
        const auto &E012 = d.extra012;
        int N = (int)S012.size(), NE = (int)E012.size();
        auto scan_arr_for_A = [&](u64 A, int aid, int ca, const vector<pair<u64, int>> &Bvec, int bsz) {
            if (bsz <= 0) return;
            u64 q = T / A;
            auto it = lower_bound(Bvec.begin(), Bvec.end(), make_pair(q, -1));
            int mid = (int)(it - Bvec.begin());
            int lo = max(0, mid - 6), hi = min(bsz - 1, mid + 6);
            for (int idx = lo; idx <= hi; idx++) {
                u64 B = Bvec[idx].first;
                int bid = Bvec[idx].second;
                if (B < 2 || (u128)A * B > (u128)d.L) continue;
                int cb = d.e[bid].cost;
                u64 P = A * B;
                if (P == T) {
                    if (ca + cb + 1 - 1 <= ctx.eff_cap()) {
                        int ids[2] = {aid, bid};
                        Step st{'*', A, B, T};
                        offer_combo(ctx, 2, ids, 2, &st, 1);
                    }
                } else {
                    u64 R = P > T ? P - T : T - P;
                    if (R <= d.max_v) {
                        int rid = d.get(R);
                        if (rid >= 0) {
                            int cr = d.e[rid].cost;
                            if (ca + cb + cr + 2 - 3 <= ctx.eff_cap()) {
                                int ids[3] = {aid, bid, rid};
                                Step tail[2] = {{'*', A, B, P},
                                                P > T ? Step{'-', P, R, T} : Step{'+', P, R, T}};
                                offer_combo(ctx, 2, ids, 3, tail, 2);
                            }
                        }
                    }
                }
            }
        };
        for (const auto &Sarr : {ref(d.sorted01), ref(d.extra01)}) {
            for (const auto &pa : Sarr.get()) {
                u64 A = pa.first;
                int aid = pa.second;
                if (A < 2) continue;
                int ca = d.e[aid].cost;
                if (ca + 1 > ctx.eff_cap()) continue;
                scan_arr_for_A(A, aid, ca, S012, N);
                if (NE > 0) scan_arr_for_A(A, aid, ca, E012, NE);
            }
        }
        // Also scan extra012 entries as A against S012
        for (const auto &pa : E012) {
            u64 A = pa.first; int aid = pa.second;
            if (A < 2) continue;
            int ca = d.e[aid].cost;
            if (ca >= 2 && ca + 2 <= ctx.eff_cap() + 2) {
                scan_arr_for_A(A, aid, ca, S012, N);
                scan_arr_for_A(A, aid, ca, E012, NE);
            }
        }

        // 2b: S2 x S2 +- S012 two-pointer scan
        if (!do_s2_twoptr || ctx.eff_cap() < 5) return;
        int j = N - 1;
        for (int i = 0; i < N; i++) {
            u64 A = S012[i].first;
            int aid = S012[i].second;
            if (A < 2) continue;
            if ((u128)A * A > (u128)T + d.max_v) break;
            int ca = d.e[aid].cost;
            if (ca < 2) continue;  // ca <= 1 already handled in 2a
            if (ca + 1 > ctx.eff_cap()) continue;
            while (j > i && (u128)A * S012[j].first > (u128)T) j--;
            int lo = max(i, j - 1), hi = min(N - 1, j + 2);
            for (int idx = lo; idx <= hi; idx++) {
                u64 B = S012[idx].first;
                int bid = S012[idx].second;
                if ((u128)A * B > (u128)d.L) continue;
                int cb = d.e[bid].cost;
                u64 P = A * B;
                u64 R = P > T ? P - T : T - P;
                if (R == 0) {
                    if (ca + (aid == bid ? 0 : cb) + 1 - 1 <= ctx.eff_cap()) {
                        int ids[2] = {aid, bid};
                        Step st{'*', A, B, T};
                        offer_combo(ctx, 2, ids, 2, &st, 1);
                    }
                } else if (R <= d.max_v) {
                    if (ca + cb > ctx.eff_cap() + 1) continue;
                    int rid = d.get(R);
                    if (rid >= 0) {
                        int cr = d.e[rid].cost;
                        if (ca + cb + cr + 2 - 3 <= ctx.eff_cap()) {
                            int ids[3] = {aid, bid, rid};
                            Step tail[2] = {{'*', A, B, P},
                                            P > T ? Step{'-', P, R, T} : Step{'+', P, R, T}};
                            offer_combo(ctx, 2, ids, 3, tail, 2);
                        }
                    }
                }
            }
        }
    }

    // Engine 3: T +- C 64-bit Factoring (+ 3-factor and S3-cofactor checks)
    void run_factor64(Ctx &ctx, int max_c, bool allow_s3_cof) {
        if (ctx.eff_cap() < 3 || max_c <= 0) return;
        u64 T = ctx.T;

        struct Off { u64 v; int id; int c; };
        vector<Off> offs;
        offs.push_back({0, -1, 0});
        for (int i = 0; i < d.n0; i++) offs.push_back({d.e[i].v, i, d.e[i].cost});
        if ((int)offs.size() < max_c) {
            vector<Off> s1c;
            s1c.reserve(d.n1 - d.n0);
            for (int i = d.n0; i < d.n1; i++) s1c.push_back({d.e[i].v, i, d.e[i].cost});
            sort(s1c.begin(), s1c.end(), [](const Off &a, const Off &b) {
                return a.v < b.v;
            });
            for (size_t k = 0; k < s1c.size() && (int)offs.size() < max_c; k++) offs.push_back(s1c[k]);
        }

        Inv inv[6];
        for (const auto &off : offs) {
            int base_c = off.c + (off.v > 0 ? 1 : 0);
            if (base_c + 2 > ctx.eff_cap()) continue;
            for (int sg = 0; sg < 2; sg++) {
                if (off.v == 0 && sg == 1) continue;
                u64 N = 0;
                if (sg == 0) {
                    if (off.v > d.L - T) continue;
                    N = T + off.v;  // T = N - off
                } else {
                    if (T <= off.v) continue;
                    N = T - off.v;  // T = N + off
                }
                factor64(N, pf, rng);
                u64 bound = min(isqrt64(N), d.max_v);
                if (bound < 2) continue;
                divisors_bounded(pf, bound, divs, 512);
                h_divs.clear();
                for (u64 dv : divs) {
                    if (dv < 2) continue;
                    int did = d.get(dv);
                    if (did >= 0) h_divs.emplace_back(dv, did);
                }
                if (h_divs.empty()) continue;

                Step add_off = (sg == 0) ? Step{'-', N, off.v, T} : Step{'+', N, off.v, T};
                for (size_t i = 0; i < h_divs.size(); i++) {
                    u64 dv = h_divs[i].first;
                    int did = h_divs[i].second;
                    u64 q = N / dv;
                    if (q < 2) continue;
                    int qid = d.get(q);
                    if (qid >= 0) {
                        int ids[3] = {did, qid, off.id};
                        if (off.v == 0) {
                            Step st{'*', dv, q, T};
                            offer_combo(ctx, 3, ids, 2, &st, 1);
                        } else {
                            Step tail[2] = {{'*', dv, q, N}, add_off};
                            offer_combo(ctx, 3, ids, 3, tail, 2);
                        }
                    }
                    // 3-factor check: N = dv * d2 * d3
                    if (base_c + d.e[did].cost + 2 <= ctx.eff_cap() + 2) {
                        for (size_t j = i; j < h_divs.size(); j++) {
                            u64 d2 = h_divs[j].first;
                            if (d2 > q) break;
                            if (q % d2 != 0) continue;
                            u64 d3 = q / d2;
                            if (d3 < 2) continue;
                            int d3id = d.get(d3);
                            if (d3id >= 0) {
                                int d2id = h_divs[j].second;
                                u64 p12 = dv * d2;
                                int ids[4] = {did, d2id, d3id, off.id};
                                if (off.v == 0) {
                                    Step tail[2] = {{'*', dv, d2, p12}, {'*', p12, d3, T}};
                                    offer_combo(ctx, 3, ids, 3, tail, 2);
                                } else {
                                    Step tail[3] = {{'*', dv, d2, p12}, {'*', p12, d3, N}, add_off};
                                    offer_combo(ctx, 3, ids, 4, tail, 3);
                                }
                            }
                        }
                    }
                    // S3 cofactor check: q = A op B with B in S0, A in H
                    if (allow_s3_cof && qid < 0 && base_c + d.e[did].cost + 3 <= ctx.eff_cap()) {
                        for (int bi = 0; bi < d.n0; bi++) {
                            u64 B = d.e[bi].v;
                            int k = inverses(q, B, d.L, inv);
                            for (int z = 0; z < k; z++) {
                                int aid = d.get(inv[z].w);
                                if (aid >= 0 && d.e[aid].cost + d.e[did].cost + base_c + 2 - 2 <= ctx.eff_cap()) {
                                    int ids[4] = {aid, bi, did, off.id};
                                    if (off.v == 0) {
                                        Step tail[2] = {mk_inv_step(inv[z], B, q), {'*', dv, q, T}};
                                        offer_combo(ctx, 3, ids, 3, tail, 2);
                                    } else {
                                        Step tail[3] = {mk_inv_step(inv[z], B, q), {'*', dv, q, N}, add_off};
                                        offer_combo(ctx, 3, ids, 4, tail, 3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Fast sub-search helper for Engine 4 & Engine 5 tails (uses S0 leaves only -> ~0.04ms!)
    bool fast_sub_solve(u64 val, int sub_budget, bool allow_fac, vector<Step> &out_steps) {
        if (sub_budget < 0) return false;
        Ctx sub{val, sub_budget, 1, 0, 1 << 20, {}};
        int eid = d.get(val);
        if (eid >= 0 && d.e[eid].cost <= sub_budget) {
            offer_combo(sub, 1, &eid, 1, nullptr, 0);
            if (sub.best_found <= 2) { out_steps = move(sub.sols.back().steps); return true; }
        }
        if (sub_budget >= 2) {
            Inv inv[6];
            for (int bi = 0; bi < d.n0; bi++) {
                u64 B = d.e[bi].v;
                int cb = d.e[bi].cost;
                int k = inverses(val, B, d.L, inv);
                for (int q = 0; q < k; q++) {
                    int ai = d.get(inv[q].w);
                    if (ai >= 0 && d.e[ai].cost + cb + 1 - 2 <= sub.eff_cap()) {
                        int ids[2] = {ai, bi};
                        Step st = mk_inv_step(inv[q], B, val);
                        offer_combo(sub, 1, ids, 2, &st, 1);
                    }
                }
            }
            if (val % 2 == 0) {
                int ai = d.get(val / 2);
                if (ai >= 0 && d.e[ai].cost + 1 <= sub.eff_cap()) {
                    Step st{'+', val / 2, val / 2, val};
                    offer_combo(sub, 1, &ai, 1, &st, 1);
                }
            }
            // Fast S0 x S012 +- S012 scan (<= 96 binary searches instead of 11,000!)
            const auto &S012 = d.sorted012;
            int N = (int)S012.size();
            if (N > 0) {
                for (int aid = 0; aid < d.n0; aid++) {
                    u64 A = d.e[aid].v;
                    int ca = d.e[aid].cost;
                    if (A < 2 || ca + 1 > sub.eff_cap()) continue;
                    u64 q = val / A;
                    auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                    int mid = (int)(it - S012.begin());
                    for (int idx = max(0, mid - 3); idx <= min(N - 1, mid + 3); idx++) {
                        u64 B = S012[idx].first;
                        int bid = S012[idx].second;
                        if (B < 2 || (u128)A * B > (u128)d.L) continue;
                        int cb = d.e[bid].cost;
                        u64 P = A * B;
                        if (P == val) {
                            if (ca + cb + 1 <= sub.eff_cap()) {
                                int ids[2] = {aid, bid};
                                Step st{'*', A, B, val};
                                offer_combo(sub, 2, ids, 2, &st, 1);
                            }
                        } else {
                            u64 R = P > val ? P - val : val - P;
                            if (R <= d.max_v) {
                                int rid = d.get(R);
                                if (rid >= 0 && ca + cb + d.e[rid].cost + 2 - 2 <= sub.eff_cap()) {
                                    int ids[3] = {aid, bid, rid};
                                    Step tail[2] = {{'*', A, B, P},
                                                    P > val ? Step{'-', P, R, val} : Step{'+', P, R, val}};
                                    offer_combo(sub, 2, ids, 3, tail, 2);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (allow_fac && sub.eff_cap() >= 3) run_factor64(sub, 8, false);
        if (sub.best_found <= sub_budget) {
            auto best_it = min_element(sub.sols.begin(), sub.sols.end(),
                                       [](const CandSol &a, const CandSol &b) { return a.cost < b.cost; });
            out_steps = move(best_it->steps);
            return true;
        }
        return false;
    }

    // Engine 4: Target-difference T = U +- D and Exact-division T = N / k
    void run_wrappers(Ctx &ctx, bool /*allow_fac*/) {
        if (ctx.eff_cap() < 2) return;
        u64 T = ctx.T;

        // 4a: Target difference over done_targets (and extra_pool)
        vector<u64> Ucand;
        const auto &DT = d.done_targets;
        if (!DT.empty()) {
            auto it = lower_bound(DT.begin(), DT.end(), T);
            int mid = (int)(it - DT.begin());
            int lo = max(0, mid - 6), hi = min((int)DT.size() - 1, mid + 6);
            for (int i = lo; i <= hi; i++) if (DT[i] != T) Ucand.push_back(DT[i]);
        }
        for (u64 v : extra_pool) if (v != T) Ucand.push_back(v);
        sort(Ucand.begin(), Ucand.end(), [&](u64 a, u64 b) {
            u64 da = a > T ? a - T : T - a;
            u64 db = b > T ? b - T : T - b;
            return da < db;
        });
        if (Ucand.size() > 12) Ucand.resize(12);

        vector<Step> sub_steps;
        for (u64 U : Ucand) {
            if (ctx.eff_cap() < 2) break;
            u64 D = U > T ? U - T : T - U;
            if (fast_sub_solve(D, ctx.eff_cap() - 1, false, sub_steps)) {
                Step fin = U > T ? Step{'-', U, D, T} : Step{'+', U, D, T};
                sub_steps.push_back(fin);
                // Re-dedup through finalize_steps
                offer_combo(ctx, 4, nullptr, 0, sub_steps.data(), (int)sub_steps.size());
            }
        }

        // 4b: Exact division T = N / k for small k in S0 (2 <= k <= 64)
        int tried = 0;
        vector<u64> ks;
        for (int i = 0; i < d.n0; i++) {
            u64 k = d.e[i].v;
            if (k >= 2 && k <= 64 && (u128)k * T <= (u128)d.L) ks.push_back(k);
        }
        sort(ks.begin(), ks.end());
        for (u64 k : ks) {
            if (tried++ >= 8 || ctx.eff_cap() < 2) break;
            u64 N = k * T;
            if (fast_sub_solve(N, ctx.eff_cap() - 1, false, sub_steps)) {
                sub_steps.push_back({'/', N, k, T});
                offer_combo(ctx, 4, nullptr, 0, sub_steps.data(), (int)sub_steps.size());
            }
        }
    }

    // Engine 5: Product-residual multi-step beam
    void run_beam(Ctx &ctx, int max_depth = 6) {
        if (ctx.eff_cap() < 4 || d.sorted012.empty()) return;
        struct BNode {
            u64 val;
            int cost;
            int parent;
            array<int, 3> dids;
            int ndids;
            array<Step, 2> tail;
            int ntail;
            double rank;
        };
        vector<BNode> hist;
        hist.push_back({ctx.T, 0, -1, {-1, -1, -1}, 0, {}, 0, 0.0});
        unordered_map<u64, int> visited;
        visited[ctx.T] = 0;
        vector<int> front{0};

        auto emit_beam_sol = [&](int node_idx, const int *tail_dids, int ntd, const Step *tail_steps, int nts) {
            vector<int> chain;
            for (int p = node_idx; p >= 0; p = hist[p].parent) chain.push_back(p);
            // Emit leaves/tail first (reverse chain from node_idx back to root)
            vector<Step> acc;
            LocalBuf loc;
            for (int i = 0; i < ntd; i++) {
                if (!emit_dict_rec(d, tail_dids[i], extra_pool, loc, acc)) return;
            }
            for (int i = 0; i < nts; i++) {
                const Step &s = tail_steps[i];
                if (!d.in_pool(s.r) && !extra_pool.count(s.r) && !loc.has(s.r)) {
                    u64 chk = 0;
                    if (!apply_op(s.op, s.a, s.b, d.L, chk) || chk != s.r) return;
                    loc.add(s.r);
                    acc.push_back(s);
                }
            }
            for (int idx : chain) {
                const BNode &bn = hist[idx];
                if (bn.parent < 0) break;
                for (int i = 0; i < bn.ndids; i++) {
                    if (!emit_dict_rec(d, bn.dids[i], extra_pool, loc, acc)) return;
                }
                for (int i = 0; i < bn.ntail; i++) {
                    const Step &s = bn.tail[i];
                    if (!d.in_pool(s.r) && !extra_pool.count(s.r) && !loc.has(s.r)) {
                        u64 chk = 0;
                        if (!apply_op(s.op, s.a, s.b, d.L, chk) || chk != s.r) return;
                        loc.add(s.r);
                        acc.push_back(s);
                    }
                }
            }
            if (acc.empty() || acc.back().r == ctx.T) offer_steps(ctx, 5, acc);
        };

        vector<BNode> next;
        const auto &S01 = d.sorted01;
        const auto &S012 = d.sorted012;
        int N = (int)S012.size();

        for (int depth = 0; depth < max_depth && !front.empty(); depth++) {
            next.clear();
            auto push_child = [&](int parent, u64 nv, int add_c, const int *dids, int nd, const Step *ts, int nt) {
                if (nv == 0 || nv >= hist[parent].val) return;
                int nc = hist[parent].cost + add_c;
                if (nc > ctx.eff_cap()) return;
                int eid = d.get(nv);
                if (eid >= 0) {
                    if (nc + d.e[eid].cost <= ctx.eff_cap()) {
                        // Temporary node push to emit complete sol
                        BNode tmp{nv, nc, parent, {-1, -1, -1}, nd, {}, nt, 0.0};
                        for (int i = 0; i < nd; i++) tmp.dids[i] = dids[i];
                        for (int i = 0; i < nt; i++) tmp.tail[i] = ts[i];
                        int tidx = (int)hist.size();
                        hist.push_back(tmp);
                        emit_beam_sol(tidx, &eid, 1, nullptr, 0);
                        hist.pop_back();
                    }
                    return;
                }
                if (nc + 2 > ctx.eff_cap()) return;
                double rk = nc + P.alpha * log2((double)nv);
                if ((int)next.size() == P.beam_w && rk >= next.back().rank) return;
                auto vit = visited.find(nv);
                if (vit != visited.end() && vit->second <= nc) return;
                for (auto it = next.begin(); it != next.end(); ++it) {
                    if (it->val == nv) {
                        if (it->cost <= nc) return;
                        next.erase(it);
                        break;
                    }
                }
                BNode bn{nv, nc, parent, {-1, -1, -1}, nd, {}, nt, rk};
                for (int i = 0; i < nd; i++) bn.dids[i] = dids[i];
                for (int i = 0; i < nt; i++) bn.tail[i] = ts[i];
                auto it = upper_bound(next.begin(), next.end(), rk,
                                      [](double k, const BNode &x) { return k < x.rank; });
                next.insert(it, bn);
                if ((int)next.size() > P.beam_w) next.pop_back();
            };

            for (int p : front) {
                if (hist[p].cost + 2 > ctx.eff_cap()) continue;
                u64 val = hist[p].val;

                // At depth > 0, check fast S012 tail completions (cost <= 5 from S0/S1/S2)
                if (depth > 0) {
                    int rem = ctx.eff_cap() - hist[p].cost;
                    vector<Step> sub;
                    if (fast_sub_solve(val, min(5, rem), false, sub)) {
                        emit_beam_sol(p, nullptr, 0, sub.data(), (int)sub.size());
                    }
                    if (hist[p].cost + 2 > ctx.eff_cap()) continue;
                }

                // 1. Product-residual: depth 0 scans all S01; depth >= 1 scans S0 only (~137x faster!)
                auto scan_prod_res = [&](u64 A, int aid, int ca) {
                    if (A < 2 || A >= val) return;
                    if (hist[p].cost + ca + 2 > ctx.eff_cap()) return;
                    u64 q = val / A;
                    auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                    int mid = (int)(it - S012.begin());
                    for (int idx = max(0, mid - 2); idx <= min(N - 1, mid + 2); idx++) {
                        u64 B = S012[idx].first;
                        int bid = S012[idx].second;
                        if (B < 2 || (u128)A * B > (u128)d.L) continue;
                        u64 P = A * B;
                        u64 R = P > val ? P - val : val - P;
                        if (R > 0 && R < val) {
                            int dids[2] = {aid, bid};
                            Step ts[2] = {{'*', A, B, P}, P > val ? Step{'-', P, R, val} : Step{'+', P, R, val}};
                            push_child(p, R, ca + d.e[bid].cost + 2, dids, 2, ts, 2);
                        }
                    }
                };
                if (depth == 0) {
                    for (const auto &pa : S01) scan_prod_res(pa.first, pa.second, d.e[pa.second].cost);
                } else {
                    for (int aid = 0; aid < d.n0; aid++) scan_prod_res(d.e[aid].v, aid, d.e[aid].cost);
                }

                // 2. Quotient/remainder over S0 u S1 (depth 0) or S0 (depth >= 1)
                int q_lim = (depth == 0) ? d.n1 : d.n0;
                for (int i = 0; i < q_lim; i++) {
                    u64 a = d.e[i].v;
                    if (a < 2 || a >= val) continue;
                    int ca = d.e[i].cost;
                    u64 q = val / a, r = val % a;
                    if (r == 0) {
                        Step st{'*', q, a, val};
                        push_child(p, q, ca + 1, &i, 1, &st, 1);
                    } else {
                        int rid = d.get(r);
                        if (rid >= 0) {
                            int dids[2] = {i, rid};
                            Step ts[2] = {{'*', q, a, q * a}, {'+', q * a, r, val}};
                            push_child(p, q, ca + d.e[rid].cost + 2, dids, 2, ts, 2);
                        }
                        if (a - r <= d.L - val && (u128)(q + 1) * a <= (u128)d.L) {
                            int rid2 = d.get(a - r);
                            if (rid2 >= 0) {
                                int dids[2] = {i, rid2};
                                Step ts[2] = {{'*', q + 1, a, (q + 1) * a}, {'-', (q + 1) * a, a - r, val}};
                                push_child(p, q + 1, ca + d.e[rid2].cost + 2, dids, 2, ts, 2);
                            }
                        }
                    }
                }

                // 3. Near-neighbor difference in sorted012
                {
                    auto it = lower_bound(S012.begin(), S012.end(), make_pair(val, -1));
                    int mid = (int)(it - S012.begin());
                    for (int idx = max(0, mid - 3); idx <= min(N - 1, mid + 3); idx++) {
                        u64 a = S012[idx].first;
                        int aid = S012[idx].second;
                        if (a == val) continue;
                        u64 nv = val > a ? val - a : a - val;
                        Step st = val > a ? Step{'+', nv, a, val} : Step{'-', a, nv, val};
                        push_child(p, nv, d.e[aid].cost + 1, &aid, 1, &st, 1);
                    }
                }

                // 4. Depth 0 factor cofactors for small C in {0} u S0
                if (depth == 0) {
                    int n_off = min(d.n0, 20);
                    for (int oi = -1; oi < n_off; oi++) {
                        u64 C = oi < 0 ? 0 : d.e[oi].v;
                        int cc = oi < 0 ? 0 : d.e[oi].cost;
                        for (int sg = 0; sg < 2; sg++) {
                            if (C == 0 && sg == 1) continue;
                            u64 Nval = sg == 0 ? (C <= d.L - val ? val + C : 0) : (val > C ? val - C : 0);
                            if (!Nval) continue;
                            factor64(Nval, pf, rng);
                            divisors_bounded(pf, min(isqrt64(Nval), d.max_v), divs, 128);
                            for (u64 dv : divs) {
                                if (dv < 2) continue;
                                int did = d.get(dv);
                                if (did < 0) continue;
                                u64 q = Nval / dv;
                                if (q >= val || q < 2) continue;
                                if (C == 0) {
                                    Step st{'*', dv, q, val};
                                    push_child(p, q, d.e[did].cost + 1, &did, 1, &st, 1);
                                } else {
                                    int dids[2] = {did, oi};
                                    Step ts[2] = {{'*', dv, q, Nval},
                                                  sg == 0 ? Step{'-', Nval, C, val} : Step{'+', Nval, C, val}};
                                    push_child(p, q, d.e[did].cost + cc + 2, dids, 2, ts, 2);
                                }
                            }
                        }
                    }
                }
            }

            front.clear();
            for (auto &bn : next) {
                if (bn.cost + 1 > ctx.eff_cap()) continue;
                visited[bn.val] = bn.cost;
                front.push_back((int)hist.size());
                hist.push_back(move(bn));
            }
        }
    }

    // Main search entry point (mode: 0=fast L<=5, 1=medium no S2xS2/beam, 2=full 5 engines)
    vector<CandSol> search(u64 T, int budget, int max_alts, int slack, SearchStats &stats, int mode = 2) {
        stats.calls++;
        Ctx ctx{T, budget, max_alts, slack, 1 << 20, {}};
        double ts0 = now_sec();
        run_exact_le4(ctx, budget >= 4);
        double ts1 = now_sec(); stats.sec[1] += ts1 - ts0;
        if (!(max_alts == 1 && slack == 0 && ctx.best_found <= 2)) {
            run_prod_diff(ctx, mode >= 2 && ctx.eff_cap() >= 5);
        }
        double ts2 = now_sec(); stats.sec[2] += ts2 - ts1;
        if (mode >= 1 && ctx.eff_cap() >= 3) {
            run_factor64(ctx, mode >= 2 ? P.fac_k : min(P.fac_k, 16), mode >= 2 && ctx.eff_cap() >= 5);
        }
        double ts3 = now_sec(); stats.sec[3] += ts3 - ts2;
        if (mode >= 1 && ctx.eff_cap() >= 2) {
            run_wrappers(ctx, false);
        }
        double ts4 = now_sec(); stats.sec[4] += ts4 - ts3;
        if (mode >= 2 && ctx.eff_cap() >= 4 && (ctx.best_found > 5 || max_alts > 1)) {
            run_beam(ctx, 6);
        }
        double ts5 = now_sec(); stats.sec[5] += ts5 - ts4;
        if (ctx.sols.empty()) return {};

        sort(ctx.sols.begin(), ctx.sols.end(), [](const CandSol &a, const CandSol &b) {
            if (a.cost != b.cost) return a.cost < b.cost;
            return a.engine < b.engine;
        });
        int best_c = ctx.sols[0].cost;
        stats.hits[ctx.sols[0].engine]++;

        // Deduplicate by final step sequence, keep up to max_alts within best_c + slack
        vector<CandSol> out;
        set<vector<u64>> seen_vals;
        for (auto &s : ctx.sols) {
            if (s.cost > best_c + slack || s.cost > budget) break;
            vector<u64> key;
            for (const auto &st : s.steps) key.push_back(st.r);
            if (!seen_vals.insert(key).second) continue;
            out.push_back(move(s));
            if ((int)out.size() >= max_alts) break;
        }
        return out;
    }
};

// ---------------------------------------------------------------- DAG & Register Allocator / Scheduler
struct DagNode { u64 v; char op; int a, b; };

struct Dag {
    u64 L = 0;
    vector<u64> starts;
    vector<u64> targets;  // non-start targets
    vector<DagNode> nodes;
    int ks = 0;
    unordered_map<u64, int> val2idx;
    vector<int> tnode;
    vector<vector<int>> users;
    vector<u64> owners;

    void init(u64 limit, const vector<u64> &st, const vector<u64> &tg) {
        L = limit;
        starts = st;
        unordered_set<u64> sset(st.begin(), st.end());
        targets.clear();
        for (u64 t : tg) if (!sset.count(t)) targets.push_back(t);
    }

    bool from_defs(const map<u64, Step> &defs, const vector<u64> &pref_order = {}) {
        nodes.clear();
        val2idx.clear();
        for (u64 s : starts) {
            if (val2idx.count(s)) continue;
            val2idx[s] = (int)nodes.size();
            nodes.push_back({s, 0, -1, -1});
        }
        ks = (int)nodes.size();

        // Collect all nodes reachable from targets via DFS
        unordered_set<u64> needed;
        function<bool(u64)> mark = [&](u64 v) -> bool {
            if (val2idx.count(v)) return true;
            if (!needed.insert(v).second) return true;
            auto it = defs.find(v);
            if (it == defs.end()) return false;
            return mark(it->second.a) && mark(it->second.b);
        };
        for (u64 t : targets) if (!mark(t)) return false;

        // Topological sort preserving pref_order when valid
        unordered_map<u64, int> indeg;
        unordered_map<u64, vector<u64>> adj;
        unordered_map<u64, int> prio;
        for (size_t i = 0; i < pref_order.size(); i++) prio[pref_order[i]] = (int)i;
        for (u64 v : needed) {
            const Step &s = defs.at(v);
            int d = 0;
            if (needed.count(s.a)) { d++; adj[s.a].push_back(v); }
            if (s.b != s.a && needed.count(s.b)) { d++; adj[s.b].push_back(v); }
            indeg[v] = d;
        }
        // Min-heap by priority (or original position)
        using PQItem = pair<int, u64>;
        vector<PQItem> ready;
        for (auto &[v, d] : indeg) {
            if (d == 0) {
                int pr = prio.count(v) ? prio[v] : 1000000;
                ready.push_back({pr, v});
            }
        }
        make_heap(ready.begin(), ready.end(), greater<PQItem>());
        size_t cnt = 0;
        while (!ready.empty()) {
            pop_heap(ready.begin(), ready.end(), greater<PQItem>());
            u64 v = ready.back().second;
            ready.pop_back();
            const Step &s = defs.at(v);
            int ia = val2idx.at(s.a), ib = val2idx.at(s.b);
            val2idx[v] = (int)nodes.size();
            nodes.push_back({v, s.op, ia, ib});
            cnt++;
            for (u64 w : adj[v]) {
                if (--indeg[w] == 0) {
                    int pr = prio.count(w) ? prio[w] : 1000000 + (int)cnt;
                    ready.push_back({pr, w});
                    push_heap(ready.begin(), ready.end(), greater<PQItem>());
                }
            }
        }
        if (cnt != needed.size()) return false;  // cycle detected!
        tnode.resize(targets.size());
        for (size_t i = 0; i < targets.size(); i++) tnode[i] = val2idx.at(targets[i]);
        return true;
    }

    bool from_prog(const vector<Instr> &prog) {
        array<u64, 48> reg{};
        array<bool, 48> def{};
        for (size_t k = 0; k < starts.size(); k++) { reg[k] = starts[k]; def[k] = true; }
        map<u64, Step> defs;
        vector<u64> ord;
        unordered_set<u64> have(starts.begin(), starts.end());
        for (const auto &in : prog) {
            if (!def[in.i] || !def[in.j]) return false;
            u64 r = 0;
            if (!apply_op(in.op, reg[in.i], reg[in.j], L, r)) return false;
            if (!have.count(r)) {
                have.insert(r);
                defs[r] = {in.op, reg[in.i], reg[in.j], r};
                ord.push_back(r);
            }
            reg[in.d] = r;
            def[in.d] = true;
        }
        return from_defs(defs, ord);
    }

    void analyse() {
        int n = (int)nodes.size();
        users.assign(n, {});
        for (int u = ks; u < n; u++) {
            users[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) users[nodes[u].b].push_back(u);
        }
        owners.assign(n, 0);
        for (size_t ti = 0; ti < tnode.size(); ti++) {
            if (ti < 64) owners[tnode[ti]] |= (1ULL << ti);
        }
        for (int u = n - 1; u >= ks; u--) {
            u64 m = owners[u];
            if (!m) continue;
            owners[nodes[u].a] |= m;
            owners[nodes[u].b] |= m;
        }
    }

    // Peak-liveness topological scheduler. Returns true and fills prog if max_peak <= 48!
    bool schedule_min_live(vector<Instr> &prog_out, int restarts, mt19937_64 &rng) const {
        int n = (int)nodes.size();
        if (n - ks > 4096) return false;
        vector<int> orig_users(n, 0), orig_indeg(n, 0);
        for (int u = ks; u < n; u++) {
            orig_users[nodes[u].a]++;
            if (nodes[u].b != nodes[u].a) orig_users[nodes[u].b]++;
            if (nodes[u].a >= ks) orig_indeg[u]++;
            if (nodes[u].b >= ks && nodes[u].b != nodes[u].a) orig_indeg[u]++;
        }
        // Adjacency list: node -> dependent nodes
        vector<vector<int>> deps(n);
        for (int u = ks; u < n; u++) {
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
        }
        vector<int> ready0;
        for (int u = ks; u < n; u++) if (orig_indeg[u] == 0) ready0.push_back(u);

        int live0 = 0;
        for (int s = 0; s < ks; s++) if (orig_users[s] > 0) live0++;

        for (int rep = 0; rep < restarts; rep++) {
            vector<int> rem = orig_users, indeg = orig_indeg, ready = ready0;
            vector<int> seq;
            seq.reserve(n - ks);
            int live = live0;
            int max_peak = live;
            bool ok = true;

            while (!ready.empty()) {
                int best_pos = -1;
                double best_sc = 1e18;
                for (int idx = 0; idx < (int)ready.size(); idx++) {
                    int u = ready[idx];
                    int a = nodes[u].a, b = nodes[u].b;
                    int freed = (rem[a] == 1 ? 1 : 0) + (b != a && rem[b] == 1 ? 1 : 0);
                    int peak = live + (freed == 0 ? 1 : 0);
                    int delta = (rem[u] > 0 ? 1 : 0) - freed;
                    double sc = (peak > 48 ? 1e6 : 0.0) + delta * 100.0 + (rem[u] > 0 ? 15.0 : 0.0) - freed * 4.0;
                    // Prefer nodes freeing inputs with low remaining uses
                    sc += 0.2 * (rem[a] + (b != a ? rem[b] : 0));
                    if (rep > 0) sc += 2.5 * ((double)(rng() % 1000) / 1000.0);
                    if (sc < best_sc) { best_sc = sc; best_pos = idx; }
                }
                int u = ready[best_pos];
                ready[best_pos] = ready.back();
                ready.pop_back();

                int a = nodes[u].a, b = nodes[u].b;
                int freed = (rem[a] == 1 ? 1 : 0) + (b != a && rem[b] == 1 ? 1 : 0);
                int peak = live + (freed == 0 ? 1 : 0);
                max_peak = max(max_peak, peak);
                if (max_peak > 48) { ok = false; break; }

                rem[a]--;
                if (b != a) rem[b]--;
                live += (rem[u] > 0 ? 1 : 0) - freed;
                seq.push_back(u);
                for (int w : deps[u]) {
                    if (--indeg[w] == 0) ready.push_back(w);
                }
            }
            if (!ok || (int)seq.size() != n - ks) continue;

            // Emit exact physical 48-register program
            vector<int> rem2 = orig_users;
            vector<int> reg_of(n, -1);
            array<bool, 48> occ{};
            for (int s = 0; s < ks; s++) {
                reg_of[s] = s;
                occ[s] = (rem2[s] > 0);
            }
            vector<Instr> prog;
            prog.reserve(seq.size());
            for (int u : seq) {
                int a = nodes[u].a, b = nodes[u].b;
                int ra = reg_of[a], rb = reg_of[b];
                if (ra < 0 || rb < 0) { ok = false; break; }
                rem2[a]--;
                if (b != a) rem2[b]--;
                if (rem2[a] == 0) occ[ra] = false;
                if (b != a && rem2[b] == 0) occ[rb] = false;
                int rd = -1;
                if (!occ[ra]) rd = ra;
                else if (!occ[rb]) rd = rb;
                else {
                    for (int r = 0; r < 48; r++) if (!occ[r]) { rd = r; break; }
                }
                if (rd < 0) { ok = false; break; }
                prog.push_back({nodes[u].op, ra, rb, rd});
                reg_of[u] = rd;
                occ[rd] = (rem2[u] > 0);
            }
            if (ok && verify_prog(L, starts, targets, prog).empty()) {
                prog_out = move(prog);
                return true;
            }
        }
        return false;
    }

    // Rematerialisation-aware Belady fallback (ported from elf.cpp)
    vector<Instr> compile_belady_mode(const vector<u64> &tord, bool planorder, bool seed_evict) const {
        map<u64, Step> defs;
        vector<u64> order;
        for (int u = ks; u < (int)nodes.size(); u++) {
            const auto &nd = nodes[u];
            defs[nd.v] = {nd.op, nodes[nd.a].v, nodes[nd.b].v, nd.v};
            order.push_back(nd.v);
        }
        set<u64> startset(starts.begin(), starts.end());
        vector<u64> seq;
        if (planorder) {
            seq = order;
        } else {
            set<u64> placed(startset);
            function<void(u64)> visit = [&](u64 v) {
                if (!placed.insert(v).second) return;
                const Step &s = defs.at(v);
                visit(s.a);
                visit(s.b);
                seq.push_back(v);
            };
            for (u64 t : tord) visit(t);
        }

        const i64 INF = 1LL << 40;
        unordered_map<u64, vector<int>> uses;
        for (int i = (int)seq.size() - 1; i >= 0; --i) {
            const Step &s = defs.at(seq[i]);
            uses[s.a].push_back(i);
            if (s.b != s.a) uses[s.b].push_back(i);
        }
        array<u64, 48> bank{};
        array<bool, 48> occ{};
        set<u64> seeds;
        for (u64 v : seq) {
            const Step &s = defs.at(v);
            if (startset.count(s.a)) seeds.insert(s.a);
            if (startset.count(s.b)) seeds.insert(s.b);
        }
        for (size_t k = 0; k < starts.size(); k++) { bank[k] = starts[k]; occ[k] = true; }

        vector<Instr> prog;
        int pos = 0;
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
        auto inreg = [&](u64 x) -> int {
            for (int k = 0; k < 48; k++) if (occ[k] && bank[k] == x) return k;
            return -1;
        };
        function<int(u64, int)> rcost = [&](u64 x, int depth) -> int {
            int rx = inreg(x);
            if (rx >= 0 && (seeds.count(x) || nextuse(x, false) < INF)) return 0;
            if (!defs.count(x)) return rx >= 0 ? 0 : 1000;
            if (depth > 8) return 60;
            const Step &m = defs.at(x);
            int c = 1 + rcost(m.a, depth + 1);
            if (m.b != m.a) c += rcost(m.b, depth + 1);
            return c >= 1000 ? 1000 : min(c, 60);
        };
        function<int(u64, u64)> emit = [&](u64 v, u64 locked) -> int {
            int r0 = inreg(v);
            if (r0 >= 0) return r0;
            if (!defs.count(v)) throw runtime_error("evicted start");
            const Step &n = defs.at(v);
            int ra = emit(n.a, locked);
            int rb = emit(n.b, locked | (1ULL << ra));
            u64 outer = locked;
            locked |= (1ULL << ra) | (1ULL << rb);
            int rd = -1;
            for (int k : {ra, rb}) {
                if ((outer >> k) & 1) continue;
                u64 old = bank[k];
                if (nextuse(old, true) >= INF && (!seeds.count(old) || seed_evict)) { rd = k; break; }
            }
            if (rd < 0) for (int k = 0; k < 48; k++) if (!((locked >> k) & 1) && !occ[k]) { rd = k; break; }
            if (rd < 0) {
                double bestsc = -1e18;
                for (int k = 0; k < 48; k++) {
                    if ((locked >> k) & 1) continue;
                    u64 old = bank[k];
                    i64 nu = nextuse(old, false);
                    if (seeds.count(old) && (!seed_evict || nu < INF)) continue;
                    double sc;
                    if (nu >= INF) sc = 1e18;
                    else if (nu <= pos) sc = -1e17;
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
                if (nu < INF && defs.count(old)) {
                    const Step &m = defs.at(old);
                    add_use(m.a, (int)nu);
                    add_use(m.b, (int)nu);
                }
            }
            prog.push_back({n.op, ra, rb, rd});
            bank[rd] = v; occ[rd] = true;
            return rd;
        };
        for (; pos < (int)seq.size(); pos++) {
            u64 v = seq[pos];
            if (inreg(v) < 0) emit(v, 0);
        }
        return prog;
    }

    bool compile(vector<Instr> &prog_out, int reorder_tries, mt19937_64 &rng) const {
        if (schedule_min_live(prog_out, 16, rng)) return true;

        vector<u64> tord;
        unordered_set<u64> tset(targets.begin(), targets.end());
        for (int u = ks; u < (int)nodes.size(); u++) {
            if (tset.count(nodes[u].v)) tord.push_back(nodes[u].v);
        }

        vector<Instr> best;
        bool have = false;
        int dag_len = (int)nodes.size() - ks;
        for (int mode = 0; mode < 4; mode++) {
            bool po = mode & 1, se = mode & 2;
            try {
                auto p = compile_belady_mode(tord, po, se);
                if (!verify_prog(L, starts, targets, p).empty()) continue;
                if (!have || p.size() < best.size()) { best = move(p); have = true; }
            } catch (...) {}
        }
        if (have && (int)best.size() > dag_len && reorder_tries > 0 && tord.size() > 2) {
            size_t cur = best.size();
            for (int it = 0; it < reorder_tries && (int)cur > dag_len; it++) {
                auto cand = tord;
                size_t i = rng() % cand.size(), j = rng() % cand.size();
                u64 x = cand[i]; cand.erase(cand.begin() + i);
                cand.insert(cand.begin() + min(j, cand.size()), x);
                for (bool se : {true, false}) {
                    try {
                        auto p = compile_belady_mode(cand, false, se);
                        if (p.size() > cur) continue;
                        if (!verify_prog(L, starts, targets, p).empty()) continue;
                        tord = move(cand); cur = p.size();
                        if (p.size() < best.size()) best = move(p);
                        break;
                    } catch (...) {}
                }
            }
        }
        if (have) prog_out = move(best);
        return have;
    }
};

// ---------------------------------------------------------------- Mode 1: Deep DAG-LNS
struct LnsResult {
    vector<Instr> prog;
    int strict_acc = 0;
    int eq_acc = 0;
    int iters = 0;
    int rej_pressure = 0;
    SearchStats stats;
};

static LnsResult lns_optimize(u64 L, const vector<u64> &starts, const vector<u64> &all_targets,
                              const vector<Instr> &init_prog, const Params &P,
                              const string &pid, mt19937_64 &rng) {
    LnsResult res;
    res.prog = init_prog;
    Dag best_dag, cur_dag;
    best_dag.init(L, starts, all_targets);
    if (!best_dag.from_prog(init_prog)) return res;
    best_dag.analyse();
    cur_dag = best_dag;

    // Try compiling initial DAG cleanly (can immediately shave spills if schedule_min_live succeeds)
    {
        vector<Instr> init_opt;
        if (best_dag.compile(init_opt, P.reorder_tries, rng) && init_opt.size() < res.prog.size()) {
            res.prog = move(init_opt);
            res.strict_acc++;
        }
    }

    int nt = (int)cur_dag.targets.size();
    if (nt == 0) return res;

    PoolDict dict;
    double t0 = now_sec();
    int stall = 0;
    int cur_prog_len = (int)res.prog.size();
    bool analysed = true;

    // Helper: assemble & evaluate a candidate replacement
    auto try_commit = [&](const vector<char> &inR, const vector<char> &excl,
                          const vector<Step> &new_steps, int engine_id, int slack_budget) -> int {
        // Return: 1 = strict improvement over global best, 0 = accepted neutral/kick, -1 = rejected
        map<u64, Step> defs;
        vector<u64> pref_ord;
        for (int u = cur_dag.ks; u < (int)cur_dag.nodes.size(); u++) {
            if (inR[u]) continue;
            const auto &nd = cur_dag.nodes[u];
            defs[nd.v] = {nd.op, cur_dag.nodes[nd.a].v, cur_dag.nodes[nd.b].v, nd.v};
            pref_ord.push_back(nd.v);
        }
        // Insert new steps near removed window
        size_t ins_pos = pref_ord.size();
        for (int u = cur_dag.ks; u < (int)cur_dag.nodes.size(); u++) {
            if (inR[u]) {
                // find first kept node after u
                for (int w = u + 1; w < (int)cur_dag.nodes.size(); w++) {
                    if (!inR[w]) {
                        auto it = find(pref_ord.begin(), pref_ord.end(), cur_dag.nodes[w].v);
                        if (it != pref_ord.end()) ins_pos = min<size_t>(ins_pos, it - pref_ord.begin());
                        break;
                    }
                }
                break;
            }
        }
        vector<u64> new_vals;
        for (const auto &s : new_steps) {
            if (!defs.count(s.r)) {
                defs[s.r] = s;
                new_vals.push_back(s.r);
            }
        }
        pref_ord.insert(pref_ord.begin() + ins_pos, new_vals.begin(), new_vals.end());

        Dag ndag;
        ndag.init(L, starts, all_targets);
        if (!ndag.from_defs(defs, pref_ord)) return -1;

        int new_dag_len = (int)ndag.nodes.size() - ndag.ks;
        if (new_dag_len > cur_prog_len + slack_budget) return -1;

        vector<Instr> nprog;
        if (!ndag.compile(nprog, P.reorder_tries, rng)) return -1;
        int nlen = (int)nprog.size();
        if (nlen > cur_prog_len + slack_budget) {
            if (new_dag_len <= cur_prog_len + slack_budget) res.rej_pressure++;
            return -1;
        }
        if (nlen < (int)res.prog.size()) {
            res.prog = nprog;
            best_dag = ndag;
            best_dag.analyse();
            cur_dag = best_dag;
            cur_prog_len = nlen;
            analysed = true;
            res.strict_acc++;
            if (engine_id >= 1 && engine_id <= 5) res.stats.wins[engine_id]++;
            stall = 0;
            if (P.verbose > 1) {
                fprintf(stderr, "  [%s] iter=%d STRICT %d (eng=%d dag=%d)\n",
                        pid.c_str(), res.iters, nlen, engine_id, new_dag_len);
            }
            return 1;
        }
        // Neutral / kick acceptance on cur_dag
        if (nlen <= cur_prog_len + slack_budget) {
            cur_dag = move(ndag);
            cur_prog_len = nlen;
            analysed = false;
            res.eq_acc++;
            return 0;
        }
        return -1;
    };

    // Helper: build PoolDict for an excluded mask
    auto build_pool_for = [&](const vector<char> &excl, const vector<char> &inR) {
        int n = (int)cur_dag.nodes.size();
        vector<u64> all_built;
        unordered_set<u64> tset(cur_dag.targets.begin(), cur_dag.targets.end());
        unordered_map<u64, int> fanout;
        unordered_set<u64> near_vals;
        for (int u = 0; u < n; u++) {
            if (!excl[u]) {
                all_built.push_back(cur_dag.nodes[u].v);
                fanout[cur_dag.nodes[u].v] = (int)cur_dag.users[u].size();
            } else if (inR[u] && u >= cur_dag.ks) {
                int a = cur_dag.nodes[u].a, b = cur_dag.nodes[u].b;
                if (!excl[a]) near_vals.insert(cur_dag.nodes[a].v);
                if (!excl[b]) near_vals.insert(cur_dag.nodes[b].v);
            }
        }
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        double tb = now_sec();
        select_leaves(L, starts, all_built, tset, fanout, near_vals, P.n_leaves, leaves, lcosts, done_t, false);
        dict.build(L, starts, all_built, leaves, lcosts, done_t, P.s2_leaves, P.s2_cap);
        res.stats.sec[0] += now_sec() - tb;
    };

    // Helper: compute downstream excl from Q nodes + inR
    auto compute_excl = [&](const vector<char> &inR, const vector<int> &Q_nodes) {
        vector<char> excl = inR;
        vector<int> st = Q_nodes;
        while (!st.empty()) {
            int x = st.back(); st.pop_back();
            for (int w : cur_dag.users[x]) {
                if (!excl[w]) { excl[w] = 1; st.push_back(w); }
            }
        }
        return excl;
    };

    while (now_sec() - t0 < P.lns_sec) {
        if (!analysed) { cur_dag.analyse(); analysed = true; }
        res.iters++;
        int n = (int)cur_dag.nodes.size();

        int kick_slack = 0;
        bool accept_eq = ((double)(rng() % 1000) / 1000.0) < P.eq_prob;
        if (P.kick_stall > 0 && stall >= P.kick_stall) {
            if (stall >= P.kick_stall * 2) {
                cur_dag = best_dag;
                cur_prog_len = (int)res.prog.size();
                analysed = true;
                stall = 0;
            } else {
                kick_slack = 1;
            }
        }

        int roll = (int)(rng() % 100);

        if (roll < 35 || nt == 1) {
            // Move Type A: k=1 single-target deep resynthesis (weighted by exclusive cone size)
            vector<int> wts(nt, 1);
            int wsum = 0;
            for (int ti = 0; ti < nt; ti++) {
                u64 bit = 1ULL << ti;
                int c = 0;
                for (int u = cur_dag.ks; u < n; u++)
                    if (cur_dag.owners[u] && (cur_dag.owners[u] & ~bit) == 0) c++;
                wts[ti] = max(1, c);
                wsum += wts[ti];
            }
            int pick = (int)(rng() % max(1, wsum)), ti = 0;
            for (; ti + 1 < nt; ti++) { if (pick < wts[ti]) break; pick -= wts[ti]; }
            u64 qm = 1ULL << ti;
            if ((cur_dag.owners[cur_dag.tnode[ti]] & ~qm) != 0) continue;

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            if (rs <= 1 && !accept_eq && !kick_slack) continue;

            int budget = rs - ((accept_eq || kick_slack) ? 0 : 1) + kick_slack;
            if (budget < 1) continue;

            vector<char> excl = compute_excl(inR, {cur_dag.tnode[ti]});
            build_pool_for(excl, inR);
            DeepSearch ds(dict, P, rng);
            auto sols = ds.search(cur_dag.targets[ti], budget, 1, 0, res.stats, 2);
            if (!sols.empty()) {
                int r = try_commit(inR, excl, sols[0].steps, sols[0].engine, kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else if (roll < 75) {
            // Move Type B: k = 2..lns_kmax joint resynthesis with A=4 branching lookahead
            u64 qm = 0;
            if (rng() % 2 == 0) {
                // Sample a shared helper footprint
                vector<u64> masks;
                for (int u = cur_dag.ks; u < n; u++) {
                    u64 m = cur_dag.owners[u];
                    int pc = __builtin_popcountll(m);
                    if (pc >= 2 && pc <= P.lns_kmax) masks.push_back(m);
                }
                if (!masks.empty()) qm = masks[rng() % masks.size()];
            }
            if (qm == 0) {
                int k = 2 + (int)(rng() % max(1, min(P.lns_kmax, nt) - 1));
                if (rng() % 2 == 0 && nt >= k) {
                    // Close-in-value window
                    int st = (int)(rng() % (nt - k + 1));
                    for (int j = 0; j < k; j++) qm |= (1ULL << (st + j));
                } else {
                    for (int j = 0; j < k; j++) qm |= (1ULL << (rng() % nt));
                }
            }
            vector<int> Q;
            for (int ti = 0; ti < nt; ti++) {
                if (((qm >> ti) & 1) && (cur_dag.owners[cur_dag.tnode[ti]] & ~qm) == 0) Q.push_back(ti);
            }
            if (Q.size() < 2) { stall++; continue; }

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            int budget = rs - ((accept_eq || kick_slack) ? 0 : 1) + kick_slack;
            if (budget < (int)Q.size()) { stall++; continue; }

            vector<int> Q_nodes;
            for (int ti : Q) Q_nodes.push_back(cur_dag.tnode[ti]);
            vector<char> excl = compute_excl(inR, Q_nodes);
            build_pool_for(excl, inR);
            DeepSearch ds(dict, P, rng);

            int best_used = budget + 1;
            vector<Step> best_steps;
            int best_eng = 0;

            int n_ord = Q.size() == 2 ? 2 : min(3, (int)Q.size());
            for (int ord = 0; ord < n_ord; ord++) {
                if (ord == 1 && Q.size() == 2) swap(Q[0], Q[1]);
                else if (ord > 0) shuffle(Q.begin(), Q.end(), rng);

                int rem_first = best_used - 1 - ((int)Q.size() - 1);
                if (rem_first < 1) break;
                // Alternative branching on Q[0] (full mode on ord=0, medium on ord>0)
                int alts0 = (ord == 0 && Q.size() <= 3) ? min(3, P.la_alts) : 1;
                auto sols0 = ds.search(cur_dag.targets[Q[0]], rem_first, alts0, 1, res.stats, ord == 0 ? 2 : 1);
                if (sols0.empty()) continue;

                for (size_t sidx = 0; sidx < sols0.size(); sidx++) {
                    const auto &s0 = sols0[sidx];
                    if ((int)s0.steps.size() + ((int)Q.size() - 1) >= best_used) continue;
                    vector<Step> acc = s0.steps;
                    vector<u64> new_v;
                    for (const auto &st : s0.steps) new_v.push_back(st.r);
                    auto mk = dict.mark();
                    dict.add_leaves_inc(new_v);
                    ds.extra_pool.clear();
                    for (u64 v : new_v) ds.extra_pool.insert(v);

                    bool ok = true;
                    int sub_mode = (ord == 0 && sidx == 0) ? 2 : 1;
                    for (size_t qi = 1; qi < Q.size(); qi++) {
                        u64 tq = cur_dag.targets[Q[qi]];
                        if (dict.in_pool(tq) || ds.extra_pool.count(tq)) continue;
                        int left = (int)(Q.size() - 1 - qi);
                        int rem_b = best_used - 1 - (int)acc.size() - left;
                        if (rem_b < 1) { ok = false; break; }
                        auto sq = ds.search(tq, rem_b, 1, 0, res.stats, sub_mode);
                        if (sq.empty()) { ok = false; break; }
                        vector<u64> add_v;
                        for (const auto &st : sq[0].steps) {
                            acc.push_back(st);
                            if (ds.extra_pool.insert(st.r).second) add_v.push_back(st.r);
                        }
                        if (qi + 1 < Q.size() && !add_v.empty()) dict.add_leaves_inc(add_v);
                    }
                    ds.extra_pool.clear();
                    dict.rollback(mk);

                    if (ok && (int)acc.size() < best_used) {
                        best_used = (int)acc.size();
                        best_steps = move(acc);
                        best_eng = s0.engine;
                    }
                }
            }
            if (!best_steps.empty()) {
                int r = try_commit(inR, excl, best_steps, best_eng, kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else if (roll < 90) {
            // Move Type C: Helper-dominated cone replacement
            unordered_set<int> isT(cur_dag.tnode.begin(), cur_dag.tnode.end());
            vector<int> helpers;
            for (int u = cur_dag.ks; u < n; u++) {
                if (!isT.count(u) && __builtin_popcountll(cur_dag.owners[u]) >= 2) helpers.push_back(u);
            }
            if (helpers.empty()) { stall++; continue; }
            int hv = helpers[rng() % helpers.size()];

            // Find nodes dominated by hv: alive[x] = reaches target without hv
            vector<char> alive(n, 0);
            for (int x = n - 1; x >= cur_dag.ks; x--) {
                if (x == hv) continue;
                bool a = isT.count(x) > 0;
                if (!a) {
                    for (int w : cur_dag.users[x]) {
                        if (w != hv && alive[w]) { a = true; break; }
                    }
                }
                alive[x] = a;
            }
            vector<char> inR(n, 0);
            int rs = 0;
            for (int x = cur_dag.ks; x < hv; x++) {
                if (!alive[x]) { inR[x] = 1; rs++; }
            }
            inR[hv] = 1; rs++;
            if (rs < 2) { stall++; continue; }

            int budget = rs - ((accept_eq || kick_slack) ? 0 : 1) + kick_slack;
            if (budget < 1) { stall++; continue; }

            vector<char> excl = compute_excl(inR, {hv});
            build_pool_for(excl, inR);
            DeepSearch ds(dict, P, rng);
            auto sols = ds.search(cur_dag.nodes[hv].v, budget, 1, 0, res.stats, 2);
            if (!sols.empty()) {
                int r = try_commit(inR, excl, sols[0].steps, sols[0].engine, kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else {
            // Move Type D: Cross-target Pair-Hub move (k=2 explicit bridge target B)
            if (nt < 2) { stall++; continue; }
            int ti1 = (int)(rng() % nt), ti2 = (int)(rng() % nt);
            if (ti1 == ti2) { ti2 = (ti1 + 1) % nt; }
            u64 qm = (1ULL << ti1) | (1ULL << ti2);
            if ((cur_dag.owners[cur_dag.tnode[ti1]] & ~qm) != 0 ||
                (cur_dag.owners[cur_dag.tnode[ti2]] & ~qm) != 0) { stall++; continue; }

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            int budget = rs - ((accept_eq || kick_slack) ? 0 : 1) + kick_slack;
            if (budget < 3) { stall++; continue; }

            vector<char> excl = compute_excl(inR, {cur_dag.tnode[ti1], cur_dag.tnode[ti2]});
            build_pool_for(excl, inR);
            DeepSearch ds(dict, P, rng);

            u64 t1 = cur_dag.targets[ti1], t2 = cur_dag.targets[ti2];
            vector<u64> bridges;
            u64 diff = t1 > t2 ? t1 - t2 : t2 - t1;
            if (diff > 0) bridges.push_back(diff);
            if (t1 <= L - t2) bridges.push_back(t1 + t2);
            for (int d1 = -2; d1 <= 2; d1++) {
                for (int d2 = -2; d2 <= 2; d2++) {
                    if ((i64)t1 + d1 <= 0 || (i64)t2 + d2 <= 0) continue;
                    u64 g = bgcd((u64)((i64)t1 + d1), (u64)((i64)t2 + d2));
                    if (g >= 64 && g < min(t1, t2)) bridges.push_back(g);
                }
            }
            sort(bridges.begin(), bridges.end());
            bridges.erase(unique(bridges.begin(), bridges.end()), bridges.end());
            if (bridges.size() > 8) bridges.resize(8);

            int best_used = budget + 1;
            vector<Step> best_steps;
            int best_eng = 0;

            for (u64 B : bridges) {
                if (dict.in_pool(B)) continue;
                auto bs = ds.search(B, min(best_used - 3, 6), 1, 0, res.stats, 1);
                if (bs.empty()) continue;

                vector<Step> acc = bs[0].steps;
                vector<u64> new_v;
                for (const auto &st : acc) new_v.push_back(st.r);
                auto mk = dict.mark();
                dict.add_leaves_inc(new_v);
                ds.extra_pool.clear();
                for (u64 v : new_v) ds.extra_pool.insert(v);

                int rem1 = best_used - 1 - (int)acc.size() - 1;
                bool ok1 = dict.in_pool(t1) || ds.extra_pool.count(t1);
                vector<CandSol> s1;
                if (!ok1 && rem1 >= 1) {
                    s1 = ds.search(t1, rem1, 1, 0, res.stats, 2);
                    ok1 = !s1.empty();
                }
                if (ok1) {
                    vector<u64> v1;
                    if (!s1.empty()) {
                        for (const auto &st : s1[0].steps) {
                            acc.push_back(st);
                            if (ds.extra_pool.insert(st.r).second) v1.push_back(st.r);
                        }
                        if (!v1.empty()) dict.add_leaves_inc(v1);
                    }
                    bool ok2 = dict.in_pool(t2) || ds.extra_pool.count(t2);
                    int rem2 = best_used - 1 - (int)acc.size();
                    vector<CandSol> s2;
                    if (!ok2 && rem2 >= 1) {
                        s2 = ds.search(t2, rem2, 1, 0, res.stats, 2);
                        ok2 = !s2.empty();
                    }
                    if (ok2) {
                        int eng = s2.empty() ? (s1.empty() ? bs[0].engine : s1[0].engine) : s2[0].engine;
                        if (!s2.empty()) for (const auto &st : s2[0].steps) acc.push_back(st);
                        if ((int)acc.size() < best_used) {
                            best_used = (int)acc.size();
                            best_steps = move(acc);
                            best_eng = eng;
                        }
                    }
                }
                ds.extra_pool.clear();
                dict.rollback(mk);
            }
            if (!best_steps.empty()) {
                int r = try_commit(inR, excl, best_steps, best_eng, kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;
        }
    }
    return res;
}

// ---------------------------------------------------------------- Mode 2: Scale-Aware Constructive Planner
static vector<Instr> construct_puzzle(u64 L, const vector<u64> &starts, const vector<u64> &targets,
                                      const Params &P, const string &pid, mt19937_64 &rng,
                                      SearchStats &stats) {
    unordered_set<u64> have(starts.begin(), starts.end());
    vector<u64> built(starts.begin(), starts.end());
    map<u64, Step> defs;
    vector<u64> order;
    unordered_set<u64> tset(targets.begin(), targets.end());
    unordered_map<u64, int> fanout;
    unordered_set<u64> near_vals;

    vector<u64> remaining;
    for (u64 t : targets) if (!have.count(t)) remaining.push_back(t);

    PoolDict dict;
    DeepSearch ds(dict, P, rng);

    auto apply_steps = [&](const vector<Step> &steps) {
        for (const auto &s : steps) {
            if (!have.count(s.r)) {
                have.insert(s.r);
                built.push_back(s.r);
                defs[s.r] = s;
                order.push_back(s.r);
                fanout[s.a]++;
                if (s.b != s.a) fanout[s.b]++;
            }
        }
    };

    // Binary fallback for unreachable targets
    function<u64(u64)> build_binary = [&](u64 v) -> u64 {
        if (have.count(v)) return v;
        if (v == 1) {
            u64 s0 = starts[0];
            apply_steps({{'/', s0, s0, 1}});
            return 1;
        }
        u64 h = build_binary(v / 2);
        apply_steps({{'+', h, h, 2 * h}});
        if (v % 2) {
            u64 one = build_binary(1);
            apply_steps({{'+', 2 * h, one, v}});
        }
        return v;
    };

    while (!remaining.empty()) {
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        double tb = now_sec();
        select_leaves(L, starts, built, tset, fanout, near_vals, P.n_leaves, leaves, lcosts, done_t, true);
        dict.build(L, starts, built, leaves, lcosts, done_t, P.s2_leaves, P.s2_cap);
        stats.sec[0] += now_sec() - tb;

        struct PlanCand { u64 target; int cost; int eng; vector<Step> steps; };
        vector<PlanCand> cands;
        int min_fast = 1 << 20;
        for (u64 t : remaining) {
            auto sols = ds.search(t, 10, P.la_alts, 2, stats, 1);
            for (auto &s : sols) {
                min_fast = min(min_fast, s.cost);
                cands.push_back({t, s.cost, s.engine, move(s.steps)});
            }
        }
        if (min_fast > 5) {
            for (size_t i = 0; i < remaining.size() && (i < 12 || cands.empty()); i++) {
                auto sols = ds.search(remaining[i], 28, P.la_alts, 2, stats, 2);
                for (auto &s : sols) cands.push_back({remaining[i], s.cost, s.engine, move(s.steps)});
            }
        }

        if (cands.empty()) {
            // Deep split fallback on smallest remaining target
            u64 t_min = *min_element(remaining.begin(), remaining.end());
            build_binary(t_min);
        } else {
            stable_sort(cands.begin(), cands.end(), [](const PlanCand &a, const PlanCand &b) {
                return a.cost < b.cost;
            });
            int K = min<int>(P.lookahead, (int)cands.size());
            int bi = 0;
            double bs = 1e18;
            if (K > 1 && remaining.size() > 1) {
                for (int k = 0; k < K; k++) {
                    vector<u64> add_v;
                    unordered_set<u64> made;
                    for (const auto &st : cands[k].steps) {
                        add_v.push_back(st.r);
                        made.insert(st.r);
                    }
                    auto mk = dict.mark();
                    dict.add_leaves_inc(add_v);
                    ds.extra_pool = made;
                    double rem_est = 0.0;
                    for (u64 x : remaining) {
                        if (have.count(x) || made.count(x)) continue;
                        vector<Step> sub;
                        if (ds.fast_sub_solve(x, 6, false, sub)) {
                            rem_est += (double)sub.size();
                        } else {
                            // Smooth residual proxy (beats flat 60 penalty)
                            rem_est += 6.0 + 0.14 * log2((double)x);
                        }
                    }
                    ds.extra_pool.clear();
                    dict.rollback(mk);
                    double sc = (double)cands[k].cost + P.la_weight * rem_est;
                    if (sc < bs) { bs = sc; bi = k; }
                }
            }
            apply_steps(cands[bi].steps);
            if (P.verbose > 1) {
                fprintf(stderr, "  [%s] built target=%llu cost=%d eng=%d total=%zu rem=%zu\n",
                        pid.c_str(), (unsigned long long)cands[bi].target, cands[bi].cost,
                        cands[bi].eng, order.size(), remaining.size() - 1);
            }
        }

        vector<u64> rem2;
        for (u64 t : remaining) if (!have.count(t)) rem2.push_back(t);
        remaining.swap(rem2);
    }

    Dag dag;
    dag.init(L, starts, targets);
    if (!dag.from_defs(defs, order)) throw runtime_error("construct dag cycle");
    vector<Instr> prog;
    if (!dag.compile(prog, P.reorder_tries, rng)) throw runtime_error("construct compile fail");
    return prog;
}

// ---------------------------------------------------------------- Driver
struct PuzzleIn { string id; u64 L; vector<u64> s, t; };

static vector<Instr> parse_prog(const string &js) {
    vector<Instr> p;
    size_t i = 0;
    while ((i = js.find("[\"", i)) != string::npos) {
        Instr in{};
        in.op = js[i + 2];
        i = js.find(',', i) + 1; in.i = atoi(js.c_str() + i);
        i = js.find(',', i) + 1; in.j = atoi(js.c_str() + i);
        i = js.find(',', i) + 1; in.d = atoi(js.c_str() + i);
        p.push_back(in);
    }
    return p;
}

int main(int argc, char **argv) {
    Params P;
    string inpath;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nx = [&]() -> string {
            if (i + 1 >= argc) throw runtime_error("missing arg after " + a);
            return string(argv[++i]);
        };
        if (a == "-t") P.threads = stoi(nx());
        else if (a == "-n") P.n_leaves = stoi(nx());
        else if (a == "-k2") P.s2_leaves = stoi(nx());
        else if (a == "-cap") P.s2_cap = stoull(nx());
        else if (a == "-fac") P.fac_k = stoi(nx());
        else if (a == "-w") P.beam_w = stoi(nx());
        else if (a == "-a") P.alpha = stod(nx());
        else if (a == "-construct") P.construct = stoi(nx());
        else if (a == "-la") P.lookahead = stoi(nx());
        else if (a == "-lw") P.la_weight = stod(nx());
        else if (a == "-lalt") P.la_alts = stoi(nx());
        else if (a == "-lns") P.lns_path = nx();
        else if (a == "-o") P.out_path = nx();
        else if (a == "-lsec") P.lns_sec = stod(nx());
        else if (a == "-lk") P.lns_kmax = stoi(nx());
        else if (a == "-eq") P.eq_prob = stod(nx());
        else if (a == "-kick") P.kick_stall = stoi(nx());
        else if (a == "-reorder") P.reorder_tries = stoi(nx());
        else if (a == "-seed") P.seed = stoi(nx());
        else if (a == "-v") P.verbose = 1;
        else if (a == "-vv") P.verbose = 2;
        else inpath = a;
    }
    if (inpath.empty()) {
        fprintf(stderr, "usage: elf2 PUZZLES.txt [-construct 1] [-lns PROGS.txt] [-lsec S] ...\n");
        return 1;
    }
    if (P.lns_sec < 0) P.lns_sec = P.lns_path.empty() ? 0.0 : 30.0;

    unordered_map<string, vector<Instr>> lns_in;
    if (!P.lns_path.empty()) {
        ifstream f(P.lns_path);
        string line;
        while (getline(f, line)) {
            size_t sp = line.find(' ');
            if (sp == string::npos) continue;
            string pid = line.substr(0, sp);
            size_t br = line.find('[', sp);
            if (br == string::npos) continue;
            lns_in[pid] = parse_prog(line.substr(br));
        }
    }

    vector<PuzzleIn> puzzles;
    {
        ifstream f(inpath);
        string line;
        while (getline(f, line)) {
            istringstream is(line);
            PuzzleIn pz;
            int ns = 0, nt = 0;
            if (!(is >> pz.id >> pz.L >> ns)) continue;
            pz.s.resize(ns);
            for (auto &x : pz.s) is >> x;
            is >> nt;
            pz.t.resize(nt);
            for (auto &x : pz.t) is >> x;
            puzzles.push_back(move(pz));
        }
    }

    vector<string> out(puzzles.size());
    atomic<int> nextp{0};
    mutex mu;
    long long total_new = 0, total_old = 0;
    SearchStats tot_stats;
    double t0 = now_sec();

    auto worker = [&]() {
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            const auto &pz = puzzles[k];
            mt19937_64 rng((u64)P.seed * 1000003ULL + hash<string>()(pz.id));
            try {
                vector<Instr> prog;
                size_t old_len = 0;
                SearchStats st;
                int s_acc = 0, e_acc = 0, iters = 0, rejp = 0;

                if (P.construct && !P.lns_path.empty()) {
                    // Construct fresh then polish with LNS if requested
                    prog = construct_puzzle(pz.L, pz.s, pz.t, P, pz.id, rng, st);
                    old_len = prog.size();
                    if (P.lns_sec > 0) {
                        auto lr = lns_optimize(pz.L, pz.s, pz.t, prog, P, pz.id, rng);
                        prog = move(lr.prog);
                        s_acc = lr.strict_acc; e_acc = lr.eq_acc; iters = lr.iters; rejp = lr.rej_pressure;
                        for (int e = 0; e <= 5; e++) { st.hits[e] += lr.stats.hits[e]; st.wins[e] += lr.stats.wins[e]; st.sec[e] += lr.stats.sec[e]; }
                    }
                } else if (!P.lns_path.empty()) {
                    auto it = lns_in.find(pz.id);
                    if (it == lns_in.end()) continue;
                    old_len = it->second.size();
                    auto lr = lns_optimize(pz.L, pz.s, pz.t, it->second, P, pz.id, rng);
                    prog = move(lr.prog);
                    s_acc = lr.strict_acc; e_acc = lr.eq_acc; iters = lr.iters; rejp = lr.rej_pressure;
                    st = lr.stats;
                } else {
                    prog = construct_puzzle(pz.L, pz.s, pz.t, P, pz.id, rng, st);
                    if (P.lns_sec > 0 && P.construct) {
                        old_len = prog.size();
                        auto lr = lns_optimize(pz.L, pz.s, pz.t, prog, P, pz.id, rng);
                        prog = move(lr.prog);
                        s_acc = lr.strict_acc; e_acc = lr.eq_acc; iters = lr.iters; rejp = lr.rej_pressure;
                        for (int e = 0; e <= 5; e++) { st.hits[e] += lr.stats.hits[e]; st.wins[e] += lr.stats.wins[e]; st.sec[e] += lr.stats.sec[e]; }
                    }
                }

                string why = verify_prog(pz.L, pz.s, pz.t, prog);
                if (!why.empty()) throw runtime_error("verify failed: " + why);
                out[k] = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);

                lock_guard<mutex> g(mu);
                total_new += (long long)prog.size();
                total_old += (long long)old_len;
                tot_stats.calls += st.calls;
                for (int e = 0; e <= 5; e++) {
                    tot_stats.hits[e] += st.hits[e];
                    tot_stats.wins[e] += st.wins[e];
                    tot_stats.sec[e] += st.sec[e];
                }
                if (P.verbose || !P.lns_path.empty()) {
                    fprintf(stderr,
                            "%s: %zu -> %zu (%+lld) | sacc=%d eq=%d it=%d rejp=%d | hits=[%llu,%llu,%llu,%llu,%llu] wins=[%llu,%llu,%llu,%llu,%llu] t=[%.1f|%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                            pz.id.c_str(), old_len, prog.size(),
                            (long long)prog.size() - (long long)old_len,
                            s_acc, e_acc, iters, rejp,
                            (unsigned long long)st.hits[1], (unsigned long long)st.hits[2],
                            (unsigned long long)st.hits[3], (unsigned long long)st.hits[4], (unsigned long long)st.hits[5],
                            (unsigned long long)st.wins[1], (unsigned long long)st.wins[2],
                            (unsigned long long)st.wins[3], (unsigned long long)st.wins[4], (unsigned long long)st.wins[5],
                            st.sec[0], st.sec[1], st.sec[2], st.sec[3], st.sec[4], st.sec[5]);
                }
            } catch (const exception &ex) {
                lock_guard<mutex> g(mu);
                fprintf(stderr, "%s FAILED: %s\n", pz.id.c_str(), ex.what());
            }
        }
    };

    vector<thread> th;
    int nt = max(1, min(P.threads, (int)puzzles.size()));
    for (int i = 0; i < nt; i++) th.emplace_back(worker);
    for (auto &t : th) t.join();

    if (!P.out_path.empty()) {
        FILE *fo = fopen(P.out_path.c_str(), "w");
        if (!fo) { fprintf(stderr, "cannot open %s\n", P.out_path.c_str()); return 1; }
        for (const auto &s : out) if (!s.empty()) fprintf(fo, "%s\n", s.c_str());
        fclose(fo);
    } else {
        for (const auto &s : out) if (!s.empty()) printf("%s\n", s.c_str());
    }
    fprintf(stderr,
            "TOTAL: old=%lld new=%lld (%+lld) time=%.1fs | calls=%llu hits=[%llu,%llu,%llu,%llu,%llu] wins=[%llu,%llu,%llu,%llu,%llu] t=[%.1f|%.1f,%.1f,%.1f,%.1f,%.1f]\n",
            total_old, total_new, total_new - total_old, now_sec() - t0,
            (unsigned long long)tot_stats.calls,
            (unsigned long long)tot_stats.hits[1], (unsigned long long)tot_stats.hits[2],
            (unsigned long long)tot_stats.hits[3], (unsigned long long)tot_stats.hits[4], (unsigned long long)tot_stats.hits[5],
            (unsigned long long)tot_stats.wins[1], (unsigned long long)tot_stats.wins[2],
            (unsigned long long)tot_stats.wins[3], (unsigned long long)tot_stats.wins[4], (unsigned long long)tot_stats.wins[5],
            tot_stats.sec[0], tot_stats.sec[1], tot_stats.sec[2], tot_stats.sec[3], tot_stats.sec[4], tot_stats.sec[5]);
    return 0;
}
