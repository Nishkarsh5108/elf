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
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
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
    int n_leaves = 128;      // S1 generation leaf count (S0 keeps all built values up to 512)
    int s2_leaves = 48;      // S2 outer leaf count
    size_t s2_cap = 400000;
    int fac_k = 48;
    int beam_w = 24;
    double alpha = 0.16;
    int construct = 0;
    int multi = 1;
    int beam_width = 0;
    int lookahead = 10;
    double la_weight = 0.35;
    int la_alts = 4;
    double lns_sec = -1.0;
    int lns_kmax = 4;
    double eq_prob = 0.20;
    int kick_stall = 50;
    int reorder_tries = 32;
    int dag_sa_iters = 20000;
    int seed = 1;
    int verbose = 0;
    string lns_path;
    string sa_path;
    string macro_path;
    string macro_pool_path;
    int min_len = 0;
    int max_len = 9999;
    string out_path;
    string pool_path;
    string dag_path;
    string pref_path;
    string dump_recipes_path;
    string beam_pool_path;
    string sched_test_path;
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

struct Expr {
    u64 v;
    int a, b;
    char op;
    uint8_t cost;
    uint8_t nanc;
    array<uint16_t, 4> anc;
};

struct BigPair { u64 v; int a, b; char op; };
struct Prod1 { u64 v; int a, b; };

static inline bool merge_anc(const Expr &ea, const Expr &eb, array<uint16_t, 4> &out, uint8_t &nout) {
    uint8_t k = 0;
    for (uint8_t i = 0; i < ea.nanc; i++) out[k++] = ea.anc[i];
    for (uint8_t j = 0; j < eb.nanc; j++) {
        uint16_t x = eb.anc[j];
        bool dup = false;
        for (uint8_t i = 0; i < k; i++) if (out[i] == x) { dup = true; break; }
        if (!dup) {
            if (k >= 4) return false;
            out[k++] = x;
        }
    }
    nout = k;
    return true;
}

struct PoolDict {
    u64 L = 0;
    vector<u64> starts;
    Index index;      // value -> id in e
    Index pool_idx;   // value -> id in e[0..n0-1] for actually-built values
    vector<Expr> e;
    int n0 = 0, n1_src = 0, n1 = 0, n2 = 0;
    size_t s2_cap = 450000;
    vector<pair<u64, int>> sorted_s0, sorted01, sorted012;
    vector<pair<u64, int>> extra01, extra012;  // incremental entries during branching
    vector<BigPair> big_pairs;
    vector<Prod1> prod1;
    vector<u64> done_targets;
    u64 max_v = 0;

    // Incremental add/rollback state for k>=2 branching lookahead & reconstruct_subset
    bool inc = false;
    vector<int> touched;
    vector<int> leaves_all, s1_all;
    vector<pair<int, Expr>> saved;
    struct Mark {
        size_t e, used, leaves, s1, saved;
        u64 max_v;
        vector<pair<u64, int>> e01, e012;
    };

    PoolDict() {
        index.init(20);
        pool_idx.init(14);
    }

    inline int get(u64 v) const { return index.get(v); }
    inline bool is_valid(int id, const vector<char> *excl_s0) const {
        if (id < 0) return false;
        if (!excl_s0) return true;
        const auto &ex = e[id];
        const size_t sz = excl_s0->size();
        for (uint8_t k = 0; k < ex.nanc; k++) {
            uint16_t idx = ex.anc[k];
            if ((size_t)idx < sz && (*excl_s0)[idx]) return false;
        }
        return true;
    }
    inline int get_valid(u64 v, const vector<char> *excl_s0) const {
        int id = index.get(v);
        return is_valid(id, excl_s0) ? id : -1;
    }
    inline bool in_pool(u64 v, const vector<char> *excl_s0 = nullptr) const {
        int id = pool_idx.get(v);
        if (id < 0) return false;
        if (excl_s0 && (size_t)id < excl_s0->size() && (*excl_s0)[id]) return false;
        return true;
    }

    inline void add_expr(u64 v, int a, int b, char op, int cost) {
        if (v == 0 || v > L || cost > 2 || (s2_cap && e.size() >= s2_cap)) return;
        array<uint16_t, 4> m_anc{};
        uint8_t m_nanc = 0;
        if (!merge_anc(e[a], e[b], m_anc, m_nanc)) return;
        int id = index.find_or_insert(v, (int)e.size());
        if (id == -1) {
            e.push_back({v, a, b, op, (uint8_t)cost, m_nanc, m_anc});
            if (inc) touched.push_back((int)e.size() - 1);
        } else if (id >= 0 && e[id].a >= 0 &&
                   (cost < e[id].cost || (cost == e[id].cost && m_nanc < e[id].nanc))) {
            if (inc) { saved.push_back({id, e[id]}); touched.push_back(id); }
            e[id] = {v, a, b, op, (uint8_t)cost, m_nanc, m_anc};
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
               const vector<u64> &leaves, const vector<int> &lcost, int s1_src_cnt,
               const vector<u64> &done_t, int s2_leaves, size_t cap_entries) {
        L = limit;
        starts = st;
        s2_cap = cap_entries;
        index.clear();
        pool_idx.clear();
        e.clear();
        sorted_s0.clear();
        sorted01.clear();
        sorted012.clear();
        extra01.clear();
        extra012.clear();
        big_pairs.clear();
        prod1.clear();
        leaves_all.clear();
        s1_all.clear();
        saved.clear();
        inc = false;

        done_targets = done_t;
        sort(done_targets.begin(), done_targets.end());
        done_targets.erase(unique(done_targets.begin(), done_targets.end()), done_targets.end());

        size_t est = (size_t)s1_src_cnt * (size_t)s1_src_cnt * 4 + 4096;
        if (s2_cap && est < s2_cap + 16384) est = s2_cap + 16384;
        index.ensure(est);
        pool_idx.ensure(all_built.size() + 64);

        unordered_set<u64> built_set(all_built.begin(), all_built.end());

        int actual_s1_src = 0;
        for (size_t k = 0; k < leaves.size(); k++) {
            if (leaves[k] == 0 || leaves[k] > L) continue;
            int nid = (int)e.size();
            int id = index.find_or_insert(leaves[k], nid);
            if (id == -1) {
                array<uint16_t, 4> a{};
                a[0] = (uint16_t)min<int>(nid, 65535);
                e.push_back({leaves[k], -1, -1, 0, (uint8_t)lcost[k], 1, a});
                if (built_set.count(leaves[k])) pool_idx.find_or_insert(leaves[k], nid);
                if ((int)k < s1_src_cnt) actual_s1_src = (int)e.size();
            }
        }
        n0 = (int)e.size();
        n1_src = min(n0, max(actual_s1_src, min(n0, s1_src_cnt)));
        for (int i = 0; i < n1_src; i++) leaves_all.push_back(i);

        for (int i = 0; i < n1_src; i++)
            for (int j = 0; j <= i; j++)
                combine(i, j);
        n1 = (int)e.size();
        for (int i = n0; i < n1; i++) s1_all.push_back(i);

        // S1[i] op S1[i] (2*x, x*x)
        for (int i = n0; i < n1; i++) {
            if (s2_cap && e.size() >= s2_cap) break;
            combine(i, i);
        }
        // Prioritized leaves outer loop j=0..k2-1
        int k2 = min(n1_src, s2_leaves);
        for (int j = 0; j < k2; j++) {
            if (s2_cap && e.size() >= s2_cap) break;
            for (int i = n0; i < n1; i++) {
                if (s2_cap && e.size() >= s2_cap) break;
                combine(i, j);
            }
        }
        n2 = (int)e.size();

        sorted_s0.reserve(n0);
        for (int i = 0; i < n0; i++) sorted_s0.emplace_back(e[i].v, i);
        sort(sorted_s0.begin(), sorted_s0.end());

        // Precompute big_pairs (U1 +- U2 for large nodes)
        u64 big_thr = max<u64>(1000, L / 4096);
        vector<int> big_ids;
        for (int idx = (int)sorted_s0.size() - 1; idx >= 0 && (int)big_ids.size() < 80; idx--) {
            if (sorted_s0[idx].first < big_thr && big_ids.size() >= 32) break;
            if (sorted_s0[idx].first >= max<u64>(1000, L / 16384))
                big_ids.push_back(sorted_s0[idx].second);
        }
        for (size_t i = 0; i < big_ids.size(); i++) {
            int ia = big_ids[i];
            u64 va = e[ia].v;
            for (size_t j = i + 1; j < big_ids.size(); j++) {
                int ib = big_ids[j];
                u64 vb = e[ib].v;
                // Note sorted_s0 scanned descending so va > vb
                if (vb <= L - va) big_pairs.push_back({va + vb, ia, ib, '+'});
                if (va > vb) big_pairs.push_back({va - vb, ia, ib, '-'});
            }
        }
        sort(big_pairs.begin(), big_pairs.end(), [](const BigPair &x, const BigPair &y) {
            return x.v < y.v;
        });

        // Precompute prod1 (A * B <= L across prioritized S0 up to 16000)
        int p1_lim = min((int)sorted_s0.size(), 180);
        prod1.reserve(min<size_t>(16000, (size_t)p1_lim * (size_t)p1_lim / 4 + 64));
        for (int i = 0; i < (int)sorted_s0.size() && prod1.size() < 16000; i++) {
            u64 va = sorted_s0[i].first;
            int ia = sorted_s0[i].second;
            if (va < 2) continue;
            for (int j = i; j < (int)sorted_s0.size(); j++) {
                u64 vb = sorted_s0[j].first;
                if ((u128)va * vb > (u128)L) break;
                prod1.push_back({va * vb, ia, sorted_s0[j].second});
                if (prod1.size() >= 16000) break;
            }
        }
        sort(prod1.begin(), prod1.end(), [](const Prod1 &x, const Prod1 &y) {
            return x.v < y.v;
        });

        // sorted01 includes all S0 (0..n0-1) + S1 (n0..n1-1)
        sorted01.reserve(n1);
        for (int i = 0; i < n1; i++) sorted01.emplace_back(e[i].v, i);
        sort(sorted01.begin(), sorted01.end());

        sorted012.reserve(n2);
        for (int i = 0; i < n2; i++) sorted012.emplace_back(e[i].v, i);
        sort(sorted012.begin(), sorted012.end());
        max_v = sorted012.empty() ? 0 : sorted012.back().first;
    }

    Mark mark() const {
        return {e.size(), index.used.size(), leaves_all.size(), s1_all.size(), saved.size(),
                max_v, extra01, extra012};
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
                array<uint16_t, 4> a{};
                a[0] = (uint16_t)min<int>(id, 65535);
                e[id] = {v, -1, -1, 0, 0, 1, a};
                nl.push_back(id);
            } else if (!s2_cap || e.size() < s2_cap + 16384) {
                int nid = (int)e.size();
                if (index.find_or_insert(v, nid) == -1) {
                    array<uint16_t, 4> a{};
                    a[0] = (uint16_t)min<int>(nid, 65535);
                    e.push_back({v, -1, -1, 0, 0, 1, a});
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
        extra01 = m.e01;
        extra012 = m.e012;
        max_v = m.max_v;
    }
};

// Order all built nodes so top n_s1_leaves sit first (for S1/S2 generation), followed by ALL
// remaining available values (up to 768 total in S0). Large nodes also go to done_t.
static void select_leaves(
    u64 L, const vector<u64> &starts, const vector<u64> &pool,
    const unordered_set<u64> &target_set, const unordered_map<u64, int> &fanout,
    const unordered_set<u64> &near_vals, int n_s1_leaves,
    vector<u64> &out_leaves, vector<int> &out_costs, int &out_s1_cnt,
    vector<u64> &out_done_t, bool virtual_one) {
    out_leaves.clear();
    out_costs.clear();
    out_done_t.clear();
    unordered_set<u64> seen;
    double logL = log((double)max<u64>(L, 2));
    u64 hi_cut = (u64)pow((double)L, 0.78);
    u64 tgt_hi = (u64)pow((double)L, 0.65);

    struct Scored { double sc; u64 v; };
    vector<Scored> cand, rest;
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
            rest.push_back({(double)v, v});
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
        size_t pos = min(starts.size(), cand.size());
        cand.insert(cand.begin() + pos, {5e8, 1});
    }
    int lim = min((int)cand.size(), n_s1_leaves);
    for (int i = 0; i < lim; i++) {
        out_leaves.push_back(cand[i].v);
        out_costs.push_back((cand[i].v == 1 && !have_one && virtual_one) ? 1 : 0);
    }
    out_s1_cnt = (int)out_leaves.size();
    for (size_t i = lim; i < cand.size(); i++) {
        if (out_leaves.size() >= 768) break;
        out_leaves.push_back(cand[i].v);
        out_costs.push_back((cand[i].v == 1 && !have_one && virtual_one) ? 1 : 0);
        out_done_t.push_back(cand[i].v);
    }
    for (const auto &r : rest) {
        if (out_leaves.size() >= 768) break;
        out_leaves.push_back(r.v);
        out_costs.push_back(0);
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

static bool emit_dict_rec(const PoolDict &d, int id, const vector<char> *excl_s0,
                          const unordered_set<u64> *excl_vals, u64 ctx_T,
                          const unordered_set<u64> &extra_pool,
                          LocalBuf &loc, vector<Step> &out) {
    if (id < 0 || !d.is_valid(id, excl_s0)) return false;
    u64 v = d.e[id].v;
    if (d.in_pool(v, excl_s0) || extra_pool.count(v) || loc.has(v)) return true;
    if (excl_vals && v != ctx_T && excl_vals->count(v)) return false;
    if (d.e[id].a < 0) {
        if (v == 1 && !d.starts.empty()) {
            loc.add(1);
            out.push_back({'/', d.starts[0], d.starts[0], 1});
            return true;
        }
        return false;
    }
    if (!emit_dict_rec(d, d.e[id].a, excl_s0, excl_vals, ctx_T, extra_pool, loc, out)) return false;
    if (!emit_dict_rec(d, d.e[id].b, excl_s0, excl_vals, ctx_T, extra_pool, loc, out)) return false;
    if (!loc.has(v)) {
        loc.add(v);
        out.push_back({d.e[id].op, d.e[d.e[id].a].v, d.e[d.e[id].b].v, v});
    }
    return true;
}

static bool finalize_steps(const PoolDict &d, const vector<char> *excl_s0,
                           const unordered_set<u64> *excl_vals,
                           const unordered_set<u64> &extra_pool,
                           const int *dict_ids, int ndids,
                           const Step *tail, int ntail,
                           u64 target, vector<Step> &out) {
    out.clear();
    LocalBuf loc;
    for (int i = 0; i < ndids; i++) {
        if (dict_ids[i] < 0) continue;
        if (!emit_dict_rec(d, dict_ids[i], excl_s0, excl_vals, target, extra_pool, loc, out)) return false;
    }
    for (int i = 0; i < ntail; i++) {
        const Step &s = tail[i];
        if (d.in_pool(s.r, excl_s0) || extra_pool.count(s.r) || loc.has(s.r)) continue;
        if (excl_vals && s.r != target && excl_vals->count(s.r)) return false;
        u64 chk = 0;
        if (!apply_op(s.op, s.a, s.b, d.L, chk) || chk != s.r) return false;
        loc.add(s.r);
        out.push_back(s);
    }
    if (!d.in_pool(target, excl_s0) && !extra_pool.count(target)) {
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
    const vector<char> *excl_s0 = nullptr;
    const unordered_set<u64> *excl_vals = nullptr;
    bool deep_beam = true;

    // Scratch buffers
    vector<pair<u64, int>> pf;
    vector<u64> divs;
    vector<pair<u64, int>> h_divs;

    DeepSearch(const PoolDict &dict, const Params &par, mt19937_64 &r) : d(dict), P(par), rng(r) {}

    inline bool ok_id(int id) const { return d.is_valid(id, excl_s0); }
    inline int vget(u64 v) const { return d.get_valid(v, excl_s0); }

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
        if (!finalize_steps(d, excl_s0, excl_vals, extra_pool, dids, ndids, tail, ntail, ctx.T, steps)) return;
        offer_steps(ctx, engine, steps);
    }

    // Engine 1: Exact depth <= 4 MITM
    void run_exact_le4(Ctx &ctx, bool depth4) {
        u64 T = ctx.T;
        if (d.in_pool(T, excl_s0) || extra_pool.count(T)) {
            vector<Step> empty;
            offer_steps(ctx, 1, empty);
            return;
        }
        int id = vget(T);
        if (id >= 0 && d.e[id].cost <= ctx.eff_cap()) {
            offer_combo(ctx, 1, &id, 1, nullptr, 0);
            if (ctx.eff_cap() <= 1) return;
        }
        if (ctx.eff_cap() < 2) return;

        Inv inv[6], inv2[6];
        // Cost 2..3: T = A op B with B in S0 u S1, A in S0..S2
        for (int bi = 0; bi < d.n1; bi++) {
            if (!ok_id(bi)) continue;
            int cb = d.e[bi].cost;
            if (cb + 1 > ctx.eff_cap()) continue;
            u64 B = d.e[bi].v;
            int k = inverses(T, B, d.L, inv);
            int maxA = min(2, ctx.eff_cap() - 1);
            for (int q = 0; q < k; q++) {
                int ai = vget(inv[q].w);
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
            int ai = vget(T / 2);
            if (ai >= 0 && d.e[ai].cost + 1 <= ctx.eff_cap()) {
                Step st{'+', T / 2, T / 2, T};
                offer_combo(ctx, 1, &ai, 1, &st, 1);
            }
        }
        {
            u64 s = isqrt64(T);
            if (s > 1 && s * s == T) {
                int ai = vget(s);
                if (ai >= 0 && d.e[ai].cost + 1 <= ctx.eff_cap()) {
                    Step st{'*', s, s, T};
                    offer_combo(ctx, 1, &ai, 1, &st, 1);
                }
            }
        }
        if (!depth4 || ctx.eff_cap() < 4) return;

        // Cost 4: T = (A op1 B) op2 C with B, C in S0 (prioritized n1_src), A in S0..S2
        int c_lim = min(d.n0, max(d.n1_src, 128));
        for (int ci = 0; ci < c_lim; ci++) {
            if (ctx.eff_cap() < 4) break;
            if (!ok_id(ci)) continue;
            u64 C = d.e[ci].v;
            int k1 = inverses(T, C, d.L, inv);
            for (int q1 = 0; q1 < k1; q1++) {
                u64 w1 = inv[q1].w;
                for (int bi = 0; bi < c_lim; bi++) {
                    if (!ok_id(bi)) continue;
                    u64 B = d.e[bi].v;
                    int k2 = inverses(w1, B, d.L, inv2);
                    for (int q2 = 0; q2 < k2; q2++) {
                        int ai = vget(inv2[q2].w);
                        if (ai >= 0) {
                            int ids[3] = {ai, bi, ci};
                            Step tail[2] = {mk_inv_step(inv2[q2], B, w1), mk_inv_step(inv[q1], C, T)};
                            offer_combo(ctx, 1, ids, 3, tail, 2);
                        }
                    }
                }
                if (w1 % 2 == 0) {
                    int ai = vget(w1 / 2);
                    if (ai >= 0) {
                        int ids[2] = {ai, ci};
                        Step tail[2] = {{'+', w1 / 2, w1 / 2, w1}, mk_inv_step(inv[q1], C, T)};
                        offer_combo(ctx, 1, ids, 2, tail, 2);
                    }
                }
                u64 s1 = isqrt64(w1);
                if (s1 > 1 && s1 * s1 == w1) {
                    int ai = vget(s1);
                    if (ai >= 0) {
                        int ids[2] = {ai, ci};
                        Step tail[2] = {{'*', s1, s1, w1}, mk_inv_step(inv[q1], C, T)};
                        offer_combo(ctx, 1, ids, 2, tail, 2);
                    }
                }
            }
        }
    }

    // Engine 2: Product-difference T = A*B +- C, S2 x S2, P1 +- P1 +- S012, and targeted 6-op (T+-C)
    void run_prod_diff(Ctx &ctx, bool do_s2_twoptr, bool do_extra_2) {
        u64 T = ctx.T;
        if (ctx.eff_cap() < 1 || d.sorted012.empty()) return;

        // 2a: A in S01 (and extra01), B in S012 (and extra012)
        const auto &S012 = d.sorted012;
        const auto &E012 = d.extra012;
        int N = (int)S012.size(), NE = (int)E012.size();
        auto scan_arr_for_A = [&](u64 A, int aid, int ca, const vector<pair<u64, int>> &Bvec, int bsz) {
            if (bsz <= 0 || !ok_id(aid)) return;
            u64 q = T / A;
            auto it = lower_bound(Bvec.begin(), Bvec.end(), make_pair(q, -1));
            int mid = (int)(it - Bvec.begin());
            int lo = max(0, mid - 6), hi = min(bsz - 1, mid + 6);
            for (int idx = lo; idx <= hi; idx++) {
                u64 B = Bvec[idx].first;
                int bid = Bvec[idx].second;
                if (B < 2 || (u128)A * B > (u128)d.L || !ok_id(bid)) continue;
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
                        int rid = vget(R);
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
                if (A < 2 || !ok_id(aid)) continue;
                int ca = d.e[aid].cost;
                if (ca + 1 > ctx.eff_cap()) continue;
                scan_arr_for_A(A, aid, ca, S012, N);
                if (NE > 0) scan_arr_for_A(A, aid, ca, E012, NE);
            }
        }
        // Also scan extra012 entries as A against S012
        for (const auto &pa : E012) {
            u64 A = pa.first; int aid = pa.second;
            if (A < 2 || !ok_id(aid)) continue;
            int ca = d.e[aid].cost;
            if (ca >= 2 && ca + 2 <= ctx.eff_cap() + 2) {
                scan_arr_for_A(A, aid, ca, S012, N);
                scan_arr_for_A(A, aid, ca, E012, NE);
            }
        }

        // 2c: P1 +- P1 +- S012 (all S0 x S0 products +- all S0 x S0 products +- small residual)
        if (do_extra_2 && ctx.eff_cap() >= 3 && ctx.best_found > 4 && !d.prod1.empty()) {
            const auto &P1 = d.prod1;
            int NP = (int)P1.size();
            auto try_p1_pair = [&](const Prod1 &p1, const Prod1 &p2, char sop) {
                if (!ok_id(p1.a) || !ok_id(p1.b) || !ok_id(p2.a) || !ok_id(p2.b)) return;
                u64 Sval = 0;
                if (sop == '+') {
                    if (p1.v > d.L - p2.v) return;
                    Sval = p1.v + p2.v;
                } else {
                    if (p1.v <= p2.v) return;
                    Sval = p1.v - p2.v;
                }
                if (Sval == T) {
                    int ids[4] = {p1.a, p1.b, p2.a, p2.b};
                    Step tail[3] = {
                        {'*', d.e[p1.a].v, d.e[p1.b].v, p1.v},
                        {'*', d.e[p2.a].v, d.e[p2.b].v, p2.v},
                        {sop, p1.v, p2.v, T}
                    };
                    offer_combo(ctx, 2, ids, 4, tail, 3);
                } else if (ctx.eff_cap() >= 4) {
                    u64 R = Sval > T ? Sval - T : T - Sval;
                    if (R <= d.max_v) {
                        int rid = vget(R);
                        if (rid >= 0 && 4 + d.e[rid].cost - 2 <= ctx.eff_cap()) {
                            int ids[5] = {p1.a, p1.b, p2.a, p2.b, rid};
                            Step tail[4] = {
                                {'*', d.e[p1.a].v, d.e[p1.b].v, p1.v},
                                {'*', d.e[p2.a].v, d.e[p2.b].v, p2.v},
                                {sop, p1.v, p2.v, Sval},
                                Sval > T ? Step{'-', Sval, R, T} : Step{'+', Sval, R, T}
                            };
                            offer_combo(ctx, 2, ids, 5, tail, 4);
                        }
                    }
                }
            };
            // P1[i] + P1[j] ~ T two-pointer
            int i = 0, j = NP - 1;
            while (i <= j) {
                u128 sum = (u128)P1[i].v + P1[j].v;
                for (int k = max(i, j - 2); k <= min(NP - 1, j + 2); k++) {
                    try_p1_pair(P1[i], P1[k], '+');
                }
                if (sum > T) j--;
                else i++;
            }
            // P1[j] - P1[i] ~ T two-pointer
            i = 0; j = 1;
            while (i < NP && j < NP) {
                if (P1[j].v <= P1[i].v) { j++; continue; }
                u64 df = P1[j].v - P1[i].v;
                for (int k = max(i + 1, j - 2); k <= min(NP - 1, j + 2); k++) {
                    try_p1_pair(P1[k], P1[i], '-');
                }
                if (df < T) j++;
                else i++;
                if (i == j) j++;
            }
        }

        // 2b: S2 x S2 +- S012 two-pointer scan
        if (do_s2_twoptr && ctx.eff_cap() >= 5 && (deep_beam || ctx.eff_cap() > 5)) {
            int j = N - 1;
            for (int i = 0; i < N; i++) {
                u64 A = S012[i].first;
                int aid = S012[i].second;
                if (A < 2 || !ok_id(aid)) continue;
                if ((u128)A * A > (u128)T + d.max_v) break;
                int ca = d.e[aid].cost;
                if (ca < 2) continue;  // ca <= 1 already handled in 2a
                if (ca + 1 > ctx.eff_cap()) continue;
                while (j > i && (u128)A * S012[j].first > (u128)T) j--;
                int rad = (A > 100000ULL) ? 1 : 2;
                int lo = max(i, j - rad), hi = min(N - 1, j + rad);
                for (int idx = lo; idx <= hi; idx++) {
                    u64 B = S012[idx].first;
                    int bid = S012[idx].second;
                    if ((u128)A * B > (u128)d.L || !ok_id(bid)) continue;
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
                        int rid = vget(R);
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

        // 2d: Targeted 6-op/7-op T +- C = A*B +- R for hard targets (0.4ms time cap, up to 8 smallest C)
        if (do_extra_2 && ctx.eff_cap() >= 5 && ctx.best_found > 5) {
            double t2d = now_sec();
            int tried_c = 0;
            for (const auto &sc : d.sorted_s0) {
                if (tried_c >= 8 || ctx.eff_cap() < 5 || now_sec() - t2d > 0.0004) break;
                int cid = sc.second;
                u64 C = sc.first;
                if (!ok_id(cid) || C == 0) continue;
                tried_c++;
                int cc = d.e[cid].cost;
                for (int sg = 0; sg < 2; sg++) {
                    if (now_sec() - t2d > 0.0004) break;
                    u64 Nval = 0;
                    if (sg == 0) {
                        if (C > d.L - T) continue;
                        Nval = T + C;
                    } else {
                        if (T <= C) continue;
                        Nval = T - C;
                    }
                    Step cstep = (sg == 0) ? Step{'-', Nval, C, T} : Step{'+', Nval, C, T};
                    for (const auto &pa : d.sorted01) {
                        u64 A = pa.first;
                        int aid = pa.second;
                        if (A < 2 || !ok_id(aid)) continue;
                        int ca = d.e[aid].cost;
                        if (ca + cc + 3 > ctx.eff_cap() + 2) continue;
                        u64 q = Nval / A;
                        auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                        int mid = (int)(it - S012.begin());
                        for (int idx = max(0, mid - 2); idx <= min(N - 1, mid + 2); idx++) {
                            u64 B = S012[idx].first;
                            int bid = S012[idx].second;
                            if (B < 2 || (u128)A * B > (u128)d.L || !ok_id(bid)) continue;
                            int cb = d.e[bid].cost;
                            u64 P = A * B;
                            if (P == Nval) {
                                if (ca + cb + cc + 2 - 2 <= ctx.eff_cap()) {
                                    int ids[3] = {aid, bid, cid};
                                    Step tail[2] = {{'*', A, B, Nval}, cstep};
                                    offer_combo(ctx, 2, ids, 3, tail, 2);
                                }
                            } else {
                                u64 R = P > Nval ? P - Nval : Nval - P;
                                if (R <= d.max_v) {
                                    int rid = vget(R);
                                    if (rid >= 0 && ca + cb + d.e[rid].cost + cc + 3 - 3 <= ctx.eff_cap()) {
                                        int ids[4] = {aid, bid, rid, cid};
                                        Step tail[3] = {
                                            {'*', A, B, P},
                                            P > Nval ? Step{'-', P, R, Nval} : Step{'+', P, R, Nval},
                                            cstep
                                        };
                                        offer_combo(ctx, 2, ids, 4, tail, 3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Engine 3: T +- C 64-bit Factoring (+ top-4 closest large U offsets, 3-factor, S3-cofactor)
    void run_factor64(Ctx &ctx, int max_c, bool allow_s3_cof) {
        if (ctx.eff_cap() < 3 || max_c <= 0) return;
        u64 T = ctx.T;

        struct Off { u64 v; int id; int c; };
        vector<Off> offs;
        offs.push_back({0, -1, 0});
        for (const auto &sc : d.sorted_s0) {
            if ((int)offs.size() >= max(1, max_c / 2)) break;
            if (!ok_id(sc.second)) continue;
            offs.push_back({sc.first, sc.second, d.e[sc.second].cost});
        }
        // Top-4 closest U in sorted_s0 (so |T - U| ~45b factors cleanly!)
        if (!d.sorted_s0.empty()) {
            vector<Off> nearU;
            auto it = lower_bound(d.sorted_s0.begin(), d.sorted_s0.end(), make_pair(T, -1));
            int mid = (int)(it - d.sorted_s0.begin());
            for (int idx = max(0, mid - 6); idx <= min((int)d.sorted_s0.size() - 1, mid + 6); idx++) {
                int id = d.sorted_s0[idx].second;
                u64 v = d.sorted_s0[idx].first;
                if (v == T || !ok_id(id)) continue;
                nearU.push_back({v, id, d.e[id].cost});
            }
            sort(nearU.begin(), nearU.end(), [&](const Off &a, const Off &b) {
                u64 da = a.v > T ? a.v - T : T - a.v;
                u64 db = b.v > T ? b.v - T : T - b.v;
                return da < db;
            });
            for (size_t k = 0; k < nearU.size() && k < 4; k++) offs.push_back(nearU[k]);
        }
        if ((int)offs.size() < max_c) {
            vector<Off> s1c;
            s1c.reserve(d.n1 - d.n0);
            for (int i = d.n0; i < d.n1; i++) {
                if (ok_id(i)) s1c.push_back({d.e[i].v, i, d.e[i].cost});
            }
            sort(s1c.begin(), s1c.end(), [](const Off &a, const Off &b) {
                return a.v < b.v;
            });
            for (size_t k = 0; k < s1c.size() && (int)offs.size() < max_c; k++) offs.push_back(s1c[k]);
        }

        Inv inv[6];
        for (const auto &off : offs) {
            int base_c = off.c + (off.v > 0 ? 1 : 0);
            if (base_c + 2 > ctx.eff_cap()) continue;
            for (int sg = 0; sg < 3; sg++) {
                if (off.v == 0 && sg > 0) continue;
                u64 N = 0;
                Step add_off{};
                if (sg == 0) {
                    if (off.v > d.L - T) continue;
                    N = T + off.v;
                    add_off = Step{'-', N, off.v, T};  // T = N - off
                } else if (sg == 1) {
                    if (T <= off.v) continue;
                    N = T - off.v;
                    add_off = Step{'+', N, off.v, T};  // T = N + off
                } else {
                    if (off.v <= T) continue;
                    N = off.v - T;
                    add_off = Step{'-', off.v, N, T};  // T = off - N
                }
                factor64(N, pf, rng);
                u64 bound = min(isqrt64(N), d.max_v);
                if (bound < 2) continue;
                divisors_bounded(pf, bound, divs, 512);
                h_divs.clear();
                for (u64 dv : divs) {
                    if (dv < 2) continue;
                    int did = vget(dv);
                    if (did >= 0) h_divs.emplace_back(dv, did);
                }
                if (h_divs.empty()) continue;

                for (size_t i = 0; i < h_divs.size(); i++) {
                    u64 dv = h_divs[i].first;
                    int did = h_divs[i].second;
                    u64 q = N / dv;
                    if (q < 2) continue;
                    int qid = vget(q);
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
                            int d3id = vget(d3);
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
                        int b_lim = min(d.n0, max(d.n1_src, 128));
                        for (int bi = 0; bi < b_lim; bi++) {
                            if (!ok_id(bi)) continue;
                            u64 B = d.e[bi].v;
                            int k = inverses(q, B, d.L, inv);
                            for (int z = 0; z < k; z++) {
                                int aid = vget(inv[z].w);
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

    // Fast sub-search helper for Engine 4, Engine 5 tails, & sensitive lookahead
    bool fast_sub_solve(u64 val, int sub_budget, bool allow_fac, vector<Step> &out_steps) {
        if (sub_budget < 0) return false;
        Ctx sub{val, sub_budget, 1, 0, 1 << 20, {}};
        int eid = vget(val);
        if (eid >= 0 && d.e[eid].cost <= sub_budget) {
            offer_combo(sub, 1, &eid, 1, nullptr, 0);
            if (sub.best_found <= 2) { out_steps = move(sub.sols.back().steps); return true; }
        }
        if (sub_budget >= 2) {
            Inv inv[6];
            for (int bi = 0; bi < d.n0; bi++) {
                if (!ok_id(bi)) continue;
                u64 B = d.e[bi].v;
                int cb = d.e[bi].cost;
                int k = inverses(val, B, d.L, inv);
                for (int q = 0; q < k; q++) {
                    int ai = vget(inv[q].w);
                    if (ai >= 0 && d.e[ai].cost + cb + 1 - 2 <= sub.eff_cap()) {
                        int ids[2] = {ai, bi};
                        Step st = mk_inv_step(inv[q], B, val);
                        offer_combo(sub, 1, ids, 2, &st, 1);
                    }
                }
            }
            if (val % 2 == 0) {
                int ai = vget(val / 2);
                if (ai >= 0 && d.e[ai].cost + 1 <= sub.eff_cap()) {
                    Step st{'+', val / 2, val / 2, val};
                    offer_combo(sub, 1, &ai, 1, &st, 1);
                }
            }
            // Fast S0(<=48) x S012 +- S012 scan (prod1 already covers full S0 x S0)
            const auto &S012 = d.sorted012;
            int N = (int)S012.size();
            if (N > 0) {
                int alim = min(d.n1_src, 48);
                for (int aid = 0; aid < alim; aid++) {
                    if (!ok_id(aid)) continue;
                    u64 A = d.e[aid].v;
                    int ca = d.e[aid].cost;
                    if (A < 2 || ca + 1 > sub.eff_cap()) continue;
                    u64 q = val / A;
                    auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                    int mid = (int)(it - S012.begin());
                    for (int idx = max(0, mid - 3); idx <= min(N - 1, mid + 3); idx++) {
                        u64 B = S012[idx].first;
                        int bid = S012[idx].second;
                        if (B < 2 || (u128)A * B > (u128)d.L || !ok_id(bid)) continue;
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
                                int rid = vget(R);
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
            // Full-pool prod1 +- S01 lookup (catches large S0 x S0 products +- S01 in 2..3 ops!)
            if (!d.prod1.empty()) {
                auto it = lower_bound(d.prod1.begin(), d.prod1.end(), val,
                                      [](const Prod1 &p, u64 v) { return p.v < v; });
                int mid = (int)(it - d.prod1.begin());
                for (int idx = max(0, mid - 6); idx <= min((int)d.prod1.size() - 1, mid + 6); idx++) {
                    const auto &p = d.prod1[idx];
                    if (!ok_id(p.a) || !ok_id(p.b)) continue;
                    if (p.v == val) {
                        int ids[2] = {p.a, p.b};
                        Step st{'*', d.e[p.a].v, d.e[p.b].v, val};
                        offer_combo(sub, 2, ids, 2, &st, 1);
                    } else if (sub.eff_cap() >= 2) {
                        u64 R = p.v > val ? p.v - val : val - p.v;
                        if (R <= d.max_v) {
                            int rid = vget(R);
                            if (rid >= 0 && 2 + d.e[rid].cost <= sub.eff_cap() + 2) {
                                int ids[3] = {p.a, p.b, rid};
                                Step tail[2] = {{'*', d.e[p.a].v, d.e[p.b].v, p.v},
                                                p.v > val ? Step{'-', p.v, R, val} : Step{'+', p.v, R, val}};
                                offer_combo(sub, 2, ids, 3, tail, 2);
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

    // Sensitive cost estimator for lookahead (gives continuous bit-residual gradient instead of flat fallback!)
    double estimate_cost(u64 x) {
        vector<Step> sub;
        if (fast_sub_solve(x, 5, false, sub)) return (double)sub.size();

        u64 min_res = x;
        const auto &S0 = d.sorted_s0;
        if (!S0.empty()) {
            auto it = lower_bound(S0.begin(), S0.end(), make_pair(x, -1));
            int mid = (int)(it - S0.begin());
            for (int k = max(0, mid - 2); k <= min((int)S0.size() - 1, mid + 2); k++) {
                if (!ok_id(S0[k].second)) continue;
                u64 u = S0[k].first;
                u64 df = u > x ? u - x : x - u;
                if (df < min_res) min_res = df;
            }
        }
        for (u64 v : extra_pool) {
            u64 df = v > x ? v - x : x - v;
            if (df < min_res) min_res = df;
        }
        if (!d.prod1.empty()) {
            auto it = lower_bound(d.prod1.begin(), d.prod1.end(), x,
                                  [](const Prod1 &p, u64 v) { return p.v < v; });
            int mid = (int)(it - d.prod1.begin());
            for (int k = max(0, mid - 2); k <= min((int)d.prod1.size() - 1, mid + 2); k++) {
                if (!ok_id(d.prod1[k].a) || !ok_id(d.prod1[k].b)) continue;
                u64 df = d.prod1[k].v > x ? d.prod1[k].v - x : x - d.prod1[k].v;
                if (df < min_res) min_res = df;
            }
        }
        const auto &S012 = d.sorted012;
        int N = (int)S012.size();
        if (N > 0) {
            int alim = min(d.n0, 48);
            for (int aid = 0; aid < alim; aid++) {
                if (!ok_id(aid)) continue;
                u64 a = d.e[aid].v;
                if (a < 2 || a > x) continue;
                u64 q = x / a;
                auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                int mid = (int)(it - S012.begin());
                for (int k = max(0, mid - 1); k <= min(N - 1, mid + 1); k++) {
                    if (!ok_id(S012[k].second)) continue;
                    u64 b = S012[k].first;
                    if ((u128)a * b > (u128)d.L) continue;
                    u64 p = a * b;
                    u64 df = p > x ? p - x : x - p;
                    if (df < min_res) min_res = df;
                }
            }
            if (!d.extra012.empty()) {
                for (int aid = 0; aid < alim; aid++) {
                    if (!ok_id(aid)) continue;
                    u64 a = d.e[aid].v;
                    if (a < 2 || a > x) continue;
                    u64 q = x / a;
                    auto it = lower_bound(d.extra012.begin(), d.extra012.end(), make_pair(q, -1));
                    int mid = (int)(it - d.extra012.begin());
                    for (int k = max(0, mid - 1); k <= min((int)d.extra012.size() - 1, mid + 1); k++) {
                        u64 b = d.extra012[k].first;
                        if ((u128)a * b > (u128)d.L) continue;
                        u64 p = a * b;
                        u64 df = p > x ? p - x : x - p;
                        if (df < min_res) min_res = df;
                    }
                }
            }
        }
        // Check near-divisibility / quotient residual against top 48 S0 leaves and extra_pool
        for (int aid = 0; aid < min(d.n0, 48); aid++) {
            if (!ok_id(aid)) continue;
            u64 a = d.e[aid].v;
            if (a < 2 || a > x) continue;
            u64 r = x % a;
            u64 dr = min(r, a - r);
            if (dr <= 64) {
                u64 q = (r <= a - r) ? (x / a) : (x / a + 1);
                u64 eff = q + dr;
                if (eff < min_res) min_res = eff;
            }
        }
        for (u64 a : extra_pool) {
            if (a < 2 || a > x) continue;
            u64 r = x % a;
            u64 dr = min(r, a - r);
            if (dr <= 64) {
                u64 q = (r <= a - r) ? (x / a) : (x / a + 1);
                u64 eff = q + dr;
                if (eff < min_res) min_res = eff;
            }
        }
        return 2.5 + 0.145 * log2((double)max<u64>(2, min_res));
    }

    // Engine 4: Full-pool U +- D, big_pairs (U1 +- U2) +- D, full-pool prod1 (A*B) +- D, exact div N/k
    void run_wrappers(Ctx &ctx, bool /*allow_s01_tail*/) {
        if (ctx.eff_cap() < 1) return;
        u64 T = ctx.T;
        vector<Step> sub_steps;

        // 4a: Single U in ALL sorted_s0 (plus extra_pool), top 6 closest
        if (ctx.eff_cap() >= 2) {
            vector<pair<u64, int>> Ucand;
            const auto &S0 = d.sorted_s0;
            if (!S0.empty()) {
                auto it = lower_bound(S0.begin(), S0.end(), make_pair(T, -1));
                int mid = (int)(it - S0.begin());
                int lo = max(0, mid - 8), hi = min((int)S0.size() - 1, mid + 8);
                for (int i = lo; i <= hi; i++) {
                    if (S0[i].first != T && ok_id(S0[i].second)) Ucand.push_back(S0[i]);
                }
            }
            for (u64 v : extra_pool) if (v != T) Ucand.emplace_back(v, -1);
            sort(Ucand.begin(), Ucand.end(), [&](const auto &a, const auto &b) {
                u64 da = a.first > T ? a.first - T : T - a.first;
                u64 db = b.first > T ? b.first - T : T - b.first;
                return da < db;
            });
            if (Ucand.size() > 6) Ucand.resize(6);

            for (size_t ui = 0; ui < Ucand.size(); ui++) {
                if (ctx.eff_cap() < 2) break;
                u64 U = Ucand[ui].first;
                int uid = Ucand[ui].second;
                u64 D = U > T ? U - T : T - U;
                if (fast_sub_solve(D, ctx.eff_cap() - 1, false, sub_steps)) {
                    Step fin = U > T ? Step{'-', U, D, T} : Step{'+', U, D, T};
                    sub_steps.push_back(fin);
                    offer_combo(ctx, 4, uid >= 0 ? &uid : nullptr, uid >= 0 ? 1 : 0,
                                sub_steps.data(), (int)sub_steps.size());
                }
            }
        }

        // 4b: Pairwise U1 +- U2 in big_pairs, top 6 closest
        if (ctx.eff_cap() >= 1 && !d.big_pairs.empty()) {
            const auto &BP = d.big_pairs;
            auto it = lower_bound(BP.begin(), BP.end(), T,
                                  [](const BigPair &x, u64 v) { return x.v < v; });
            int mid = (int)(it - BP.begin());
            vector<int> bp_idx;
            for (int k = max(0, mid - 16); k <= min((int)BP.size() - 1, mid + 16); k++) {
                if (ok_id(BP[k].a) && ok_id(BP[k].b)) bp_idx.push_back(k);
            }
            sort(bp_idx.begin(), bp_idx.end(), [&](int x, int y) {
                u64 dx = BP[x].v > T ? BP[x].v - T : T - BP[x].v;
                u64 dy = BP[y].v > T ? BP[y].v - T : T - BP[y].v;
                return dx < dy;
            });
            if (bp_idx.size() > 6) bp_idx.resize(6);

            for (size_t bi = 0; bi < bp_idx.size(); bi++) {
                const auto &bp = BP[bp_idx[bi]];
                int dids[2] = {bp.a, bp.b};
                Step s_bp{bp.op, d.e[bp.a].v, d.e[bp.b].v, bp.v};
                if (bp.v == T) {
                    offer_combo(ctx, 4, dids, 2, &s_bp, 1);
                    continue;
                }
                if (ctx.eff_cap() < 2) break;
                u64 D = bp.v > T ? bp.v - T : T - bp.v;
                if (fast_sub_solve(D, ctx.eff_cap() - 2, false, sub_steps)) {
                    vector<Step> full;
                    full.reserve(sub_steps.size() + 2);
                    full.push_back(s_bp);
                    for (auto &st : sub_steps) full.push_back(st);
                    full.push_back(bp.v > T ? Step{'-', bp.v, D, T} : Step{'+', bp.v, D, T});
                    offer_combo(ctx, 4, dids, 2, full.data(), (int)full.size());
                }
            }
        }

        // 4c: Full-pool A*B in prod1 +- D, top 6 closest
        if (ctx.eff_cap() >= 1 && !d.prod1.empty()) {
            const auto &P1 = d.prod1;
            auto it = lower_bound(P1.begin(), P1.end(), T,
                                  [](const Prod1 &x, u64 v) { return x.v < v; });
            int mid = (int)(it - P1.begin());
            vector<int> p_idx;
            for (int k = max(0, mid - 16); k <= min((int)P1.size() - 1, mid + 16); k++) {
                if (ok_id(P1[k].a) && ok_id(P1[k].b)) p_idx.push_back(k);
            }
            sort(p_idx.begin(), p_idx.end(), [&](int x, int y) {
                u64 dx = P1[x].v > T ? P1[x].v - T : T - P1[x].v;
                u64 dy = P1[y].v > T ? P1[y].v - T : T - P1[y].v;
                return dx < dy;
            });
            if (p_idx.size() > 6) p_idx.resize(6);

            for (size_t pi = 0; pi < p_idx.size(); pi++) {
                const auto &p = P1[p_idx[pi]];
                int dids[2] = {p.a, p.b};
                Step s_p{'*', d.e[p.a].v, d.e[p.b].v, p.v};
                if (p.v == T) {
                    offer_combo(ctx, 4, dids, 2, &s_p, 1);
                    continue;
                }
                if (ctx.eff_cap() < 2) break;
                u64 D = p.v > T ? p.v - T : T - p.v;
                if (fast_sub_solve(D, ctx.eff_cap() - 2, false, sub_steps)) {
                    vector<Step> full;
                    full.reserve(sub_steps.size() + 2);
                    full.push_back(s_p);
                    for (auto &st : sub_steps) full.push_back(st);
                    full.push_back(p.v > T ? Step{'-', p.v, D, T} : Step{'+', p.v, D, T});
                    offer_combo(ctx, 4, dids, 2, full.data(), (int)full.size());
                }
            }
        }

        // 4d: Exact division T = N / k for k in S0 (2 <= k <= 128), top 6 smallest k
        int tried = 0;
        for (const auto &sc : d.sorted_s0) {
            if (tried >= 6 || ctx.eff_cap() < 2) break;
            u64 k = sc.first;
            if (k < 2) continue;
            if (k > 128) break;
            if (!ok_id(sc.second) || (u128)k * T > (u128)d.L) continue;
            tried++;
            u64 N = k * T;
            if (fast_sub_solve(N, ctx.eff_cap() - 1, false, sub_steps)) {
                int kid = sc.second;
                sub_steps.push_back({'/', N, k, T});
                offer_combo(ctx, 4, &kid, 1, sub_steps.data(), (int)sub_steps.size());
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
            vector<Step> acc;
            LocalBuf loc;
            for (int i = 0; i < ntd; i++) {
                if (!emit_dict_rec(d, tail_dids[i], excl_s0, excl_vals, ctx.T, extra_pool, loc, acc)) return;
            }
            for (int i = 0; i < nts; i++) {
                const Step &s = tail_steps[i];
                if (!d.in_pool(s.r, excl_s0) && !extra_pool.count(s.r) && !loc.has(s.r)) {
                    if (excl_vals && s.r != ctx.T && excl_vals->count(s.r)) return;
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
                    if (!emit_dict_rec(d, bn.dids[i], excl_s0, excl_vals, ctx.T, extra_pool, loc, acc)) return;
                }
                for (int i = 0; i < bn.ntail; i++) {
                    const Step &s = bn.tail[i];
                    if (!d.in_pool(s.r, excl_s0) && !extra_pool.count(s.r) && !loc.has(s.r)) {
                        if (excl_vals && s.r != ctx.T && excl_vals->count(s.r)) return;
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
                int eid = vget(nv);
                if (eid >= 0) {
                    if (nc + d.e[eid].cost <= ctx.eff_cap()) {
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

                if (depth > 0) {
                    int rem = ctx.eff_cap() - hist[p].cost;
                    vector<Step> sub;
                    if (fast_sub_solve(val, min(5, rem), false, sub)) {
                        emit_beam_sol(p, nullptr, 0, sub.data(), (int)sub.size());
                    }
                    if (hist[p].cost + 2 > ctx.eff_cap()) continue;
                }

                // 1. Product-residual
                auto scan_prod_res = [&](u64 A, int aid, int ca) {
                    if (A < 2 || A >= val || !ok_id(aid)) return;
                    if (hist[p].cost + ca + 2 > ctx.eff_cap()) return;
                    u64 q = val / A;
                    auto it = lower_bound(S012.begin(), S012.end(), make_pair(q, -1));
                    int mid = (int)(it - S012.begin());
                    for (int idx = max(0, mid - 2); idx <= min(N - 1, mid + 2); idx++) {
                        u64 B = S012[idx].first;
                        int bid = S012[idx].second;
                        if (B < 2 || (u128)A * B > (u128)d.L || !ok_id(bid)) continue;
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
                    int a_lim = min(d.n0, max(d.n1_src, 128));
                    for (int aid = 0; aid < a_lim; aid++) scan_prod_res(d.e[aid].v, aid, d.e[aid].cost);
                }

                // 2. Quotient/remainder over S0 u S1 (all S0 + top 512/6000 S1 at depth >= 1)
                int q_lim = (depth == 0) ? d.n1 : min(d.n1, d.n0 + (deep_beam ? 6000 : 512));
                for (int i = 0; i < q_lim; i++) {
                    if (!ok_id(i)) continue;
                    u64 a = d.e[i].v;
                    if (a < 2 || a >= val) continue;
                    int ca = d.e[i].cost;
                    u64 q = val / a, r = val % a;
                    if (r == 0) {
                        Step st{'*', q, a, val};
                        push_child(p, q, ca + 1, &i, 1, &st, 1);
                    } else {
                        int rid = vget(r);
                        if (rid >= 0) {
                            int dids[2] = {i, rid};
                            Step ts[2] = {{'*', q, a, q * a}, {'+', q * a, r, val}};
                            push_child(p, q, ca + d.e[rid].cost + 2, dids, 2, ts, 2);
                        }
                        if (a - r <= d.L - val && (u128)(q + 1) * a <= (u128)d.L) {
                            int rid2 = vget(a - r);
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
                        if (a == val || !ok_id(aid)) continue;
                        u64 nv = val > a ? val - a : a - val;
                        Step st = val > a ? Step{'+', nv, a, val} : Step{'-', a, nv, val};
                        push_child(p, nv, d.e[aid].cost + 1, &aid, 1, &st, 1);
                    }
                }

                // 4. Depth 0 factor cofactors for small C in {0} u S0
                if (depth == 0) {
                    int tried_o = 0;
                    for (int oi = -1; oi < (int)d.sorted_s0.size() && tried_o < 20; oi++) {
                        u64 C = 0;
                        int cid = -1, cc = 0;
                        if (oi >= 0) {
                            cid = d.sorted_s0[oi].second;
                            if (!ok_id(cid)) continue;
                            C = d.sorted_s0[oi].first;
                            cc = d.e[cid].cost;
                            tried_o++;
                        }
                        for (int sg = 0; sg < 2; sg++) {
                            if (C == 0 && sg == 1) continue;
                            u64 Nval = sg == 0 ? (C <= d.L - val ? val + C : 0) : (val > C ? val - C : 0);
                            if (!Nval) continue;
                            factor64(Nval, pf, rng);
                            divisors_bounded(pf, min(isqrt64(Nval), d.max_v), divs, 128);
                            for (u64 dv : divs) {
                                if (dv < 2) continue;
                                int did = vget(dv);
                                if (did < 0) continue;
                                u64 q = Nval / dv;
                                if (q >= val || q < 2) continue;
                                if (C == 0) {
                                    Step st{'*', dv, q, val};
                                    push_child(p, q, d.e[did].cost + 1, &did, 1, &st, 1);
                                } else {
                                    int dids[2] = {did, cid};
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

    // Main search entry point (mode: 0=fast L<=5, 1=medium, 2=full 5 engines + P1+-P1 + 2d, 3=deep boot beam)
    vector<CandSol> search(u64 T, int budget, int max_alts, int slack, SearchStats &stats, int mode = 2) {
        stats.calls++;
        Ctx ctx{T, budget, max_alts, slack, 1 << 20, {}};
        double ts0 = now_sec();
        run_exact_le4(ctx, budget >= 4);
        double ts1 = now_sec(); stats.sec[1] += ts1 - ts0;
        if (!(max_alts == 1 && slack == 0 && ctx.best_found <= 2)) {
            run_prod_diff(ctx, mode >= 2 && ctx.eff_cap() >= 5, mode >= 2);
        }
        double ts2 = now_sec(); stats.sec[2] += ts2 - ts1;
        if (mode >= 1 && ctx.eff_cap() >= 3) {
            run_factor64(ctx, mode >= 2 ? P.fac_k : min(P.fac_k, 16), mode >= 2 && ctx.eff_cap() >= 5);
        }
        double ts3 = now_sec(); stats.sec[3] += ts3 - ts2;
        if (mode >= 1 && ctx.eff_cap() >= 1) {
            run_wrappers(ctx, mode >= 2);
        }
        double ts4 = now_sec(); stats.sec[4] += ts4 - ts3;
        if (mode >= 2 && ctx.eff_cap() >= 4 && (ctx.best_found > 5 || max_alts > 1 || mode >= 3)) {
            run_beam(ctx, mode >= 3 ? 12 : 6);
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

    struct PeakInfo { int max_peak; int over48; int worst_idx; };

    PeakInfo eval_order(const vector<int> &seq, vector<int> *lu_out = nullptr) const {
        int n = (int)nodes.size();
        int M = (int)seq.size();
        vector<int> lu_local;
        vector<int> &lu = lu_out ? *lu_out : lu_local;
        lu.assign(n, -1);
        for (int idx = 0; idx < M; idx++) {
            int u = seq[idx];
            int a = nodes[u].a, b = nodes[u].b;
            if (idx > lu[a]) lu[a] = idx;
            if (idx > lu[b]) lu[b] = idx;
        }
        int live = 0;
        for (int s = 0; s < ks; s++) if (lu[s] >= 0) live++;
        int mp = live, ov = max(0, live - 48), widx = 0;
        for (int idx = 0; idx < M; idx++) {
            int u = seq[idx];
            int a = nodes[u].a, b = nodes[u].b;
            int freed = (lu[a] == idx ? 1 : 0) + (b != a && lu[b] == idx ? 1 : 0);
            int pk = live + (freed == 0 ? 1 : 0);
            if (pk > mp) { mp = pk; widx = idx; }
            if (pk > 48) ov += (pk - 48);
            if (lu[u] > idx) live++;
            live -= freed;
        }
        return {mp, ov, widx};
    }

    bool emit_exact_order(const vector<int> &seq, vector<Instr> &prog_out) const {
        int n = (int)nodes.size();
        vector<int> rem(n, 0);
        for (int u : seq) {
            rem[nodes[u].a]++;
            if (nodes[u].b != nodes[u].a) rem[nodes[u].b]++;
        }
        vector<int> reg_of(n, -1);
        array<bool, 48> occ{};
        for (int s = 0; s < ks; s++) {
            reg_of[s] = s;
            occ[s] = (rem[s] > 0);
        }
        vector<Instr> prog;
        prog.reserve(seq.size());
        for (int u : seq) {
            int a = nodes[u].a, b = nodes[u].b;
            int ra = reg_of[a], rb = reg_of[b];
            if (ra < 0 || rb < 0) return false;
            rem[a]--;
            if (b != a) rem[b]--;
            if (rem[a] == 0) occ[ra] = false;
            if (b != a && rem[b] == 0) occ[rb] = false;
            int rd = -1;
            if (!occ[ra]) rd = ra;
            else if (!occ[rb]) rd = rb;
            else {
                for (int r = 0; r < 48; r++) if (!occ[r]) { rd = r; break; }
            }
            if (rd < 0) return false;
            prog.push_back({nodes[u].op, ra, rb, rd});
            reg_of[u] = rd;
            occ[rd] = (rem[u] > 0);
        }
        if (!verify_prog(L, starts, targets, prog).empty()) return false;
        prog_out = move(prog);
        return true;
    }

    bool fix_pressure_sa(vector<int> &seq, int max_iters, mt19937_64 &rng) const {
        int n = (int)nodes.size();
        int M = (int)seq.size();
        if (M <= 1) return eval_order(seq).max_peak <= 48;

        vector<vector<int>> deps(n);
        vector<int> indeg0(n, 0);
        for (int u = ks; u < n; u++) {
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
            if (nodes[u].a >= ks) indeg0[u]++;
            if (nodes[u].b >= ks && nodes[u].b != nodes[u].a) indeg0[u]++;
        }

        PeakInfo g_best_info = eval_order(seq);
        if (g_best_info.max_peak <= 48) return true;
        vector<int> g_best_seq = seq;

        auto make_kahn_min_live = [&]() -> vector<int> {
            vector<int> indeg = indeg0, rem_u(n, 0), ready;
            for (int u = 0; u < n; u++) rem_u[u] = (int)deps[u].size();
            for (int u = ks; u < n; u++) if (indeg[u] == 0) ready.push_back(u);
            vector<int> out;
            out.reserve(M);
            while (!ready.empty()) {
                int best_ri = 0;
                int best_sc = 1 << 29;
                for (int ri = 0; ri < (int)ready.size(); ri++) {
                    int u = ready[ri];
                    int a = nodes[u].a, b = nodes[u].b;
                    int freed = (rem_u[a] == 1) + (b != a && rem_u[b] == 1);
                    int born = deps[u].empty() ? 0 : 1;
                    int sc = (born - freed) * 16 + (int)(rng() % 5);
                    if (sc < best_sc) { best_sc = sc; best_ri = ri; }
                }
                int u = ready[best_ri];
                ready[best_ri] = ready.back();
                ready.pop_back();
                out.push_back(u);
                rem_u[nodes[u].a]--;
                if (nodes[u].b != nodes[u].a) rem_u[nodes[u].b]--;
                for (int ch : deps[u]) {
                    if (--indeg[ch] == 0) ready.push_back(ch);
                }
            }
            return (int)out.size() == M ? out : seq;
        };

        auto perturb_seq = [&](vector<int> s, int kicks) -> vector<int> {
            vector<int> p(n, -1);
            for (int i = 0; i < M; i++) p[s[i]] = i;
            for (int k = 0; k < kicks; k++) {
                int idx = (int)(rng() % M);
                int u = s[idx];
                int lo = 0;
                if (nodes[u].a >= ks) lo = max(lo, p[nodes[u].a] + 1);
                if (nodes[u].b >= ks) lo = max(lo, p[nodes[u].b] + 1);
                int hi = M - 1;
                for (int ch : deps[u]) hi = min(hi, p[ch] - 1);
                if (lo >= hi) continue;
                int np = lo + (int)(rng() % (hi - lo + 1));
                if (np == idx) continue;
                s.erase(s.begin() + idx);
                s.insert(s.begin() + np, u);
                int mn = min(idx, np), mx = max(idx, np);
                for (int i = mn; i <= mx; i++) p[s[i]] = i;
            }
            return s;
        };

        auto score_pi = [](const PeakInfo &pi) {
            return pi.max_peak * 100.0 + pi.over48 * 2.5;
        };

        int n_restarts = (max_iters >= 60000) ? 4 : 1;
        int iters_per_restart = max(1, max_iters / n_restarts);

        for (int rst = 0; rst < n_restarts; rst++) {
            vector<int> cur_seq;
            if (rst == 0) cur_seq = seq;
            else if (rst == 1 || rst == 3) cur_seq = make_kahn_min_live();
            else cur_seq = perturb_seq(g_best_seq, 40);

            vector<int> lu;
            PeakInfo cur = eval_order(cur_seq, &lu);
            if (cur.max_peak <= 48) { seq = move(cur_seq); return true; }

            vector<int> pos(n, -1);
            for (int i = 0; i < M; i++) pos[cur_seq[i]] = i;
            double curE = score_pi(cur);

            if (cur.max_peak < g_best_info.max_peak ||
                (cur.max_peak == g_best_info.max_peak && cur.over48 < g_best_info.over48)) {
                g_best_info = cur;
                g_best_seq = cur_seq;
            }

            vector<int> cand_u, trial, chosen_seq;
            for (int it = 0; it < iters_per_restart; it++) {
                if (g_best_info.max_peak <= 48) break;

                bool two_node_tried = false;
                double chosen_E = 1e18;
                chosen_seq.clear();

                // ~25% 2-node / consumer-pull moves targeting nodes live across cur.worst_idx
                if (rng() % 100 < 25) {
                    cand_u.clear();
                    int w = cur.worst_idx;
                    for (int i = 0; i <= w && i < M; i++) {
                        int u = cur_seq[i];
                        if (lu[u] >= w && !deps[u].empty()) cand_u.push_back(i);
                    }
                    if (!cand_u.empty()) {
                        int idx = cand_u[rng() % cand_u.size()];
                        int u = cur_seq[idx];
                        int cpos = lu[u];
                        if (cpos > idx && cpos < M) {
                            int ch = cur_seq[cpos];
                            int clo = 0;
                            if (nodes[ch].a >= ks) clo = max(clo, pos[nodes[ch].a] + 1);
                            if (nodes[ch].b >= ks) clo = max(clo, pos[nodes[ch].b] + 1);
                            if (clo < cpos) {
                                two_node_tried = true;
                                trial = cur_seq;
                                trial.erase(trial.begin() + cpos);
                                trial.insert(trial.begin() + clo, ch);
                                PeakInfo pi1 = eval_order(trial);
                                double E1 = score_pi(pi1);
                                if (E1 < chosen_E) { chosen_E = E1; chosen_seq = trial; }

                                // Also try pushing u right before its earliest consumer after pulling ch
                                int u_min_ch = M - 1;
                                for (int k = idx + 1; k < M; k++) {
                                    int d = trial[k];
                                    if (nodes[d].a == u || nodes[d].b == u) { u_min_ch = k; break; }
                                }
                                int u_hi = u_min_ch - 1;
                                if (u_hi > idx) {
                                    trial.erase(trial.begin() + idx);
                                    trial.insert(trial.begin() + u_hi, u);
                                    PeakInfo pi2 = eval_order(trial);
                                    double E2 = score_pi(pi2);
                                    if (E2 < chosen_E) { chosen_E = E2; chosen_seq = move(trial); }
                                }
                            }
                        }
                    }
                }

                if (!two_node_tried) {
                    int idx = -1;
                    if (rng() % 100 < 75) {
                        cand_u.clear();
                        int w = cur.worst_idx;
                        for (int i = 0; i < M; i++) {
                            int u = cur_seq[i];
                            if ((i <= w && lu[u] >= w) || abs(i - w) <= 3 || (lu[u] >= w && lu[u] <= w + 6)) {
                                cand_u.push_back(i);
                            }
                        }
                        if (!cand_u.empty()) idx = cand_u[rng() % cand_u.size()];
                    }
                    if (idx < 0) idx = (int)(rng() % M);

                    int u = cur_seq[idx];
                    int a = nodes[u].a, b = nodes[u].b;
                    int lo = 0;
                    if (a >= ks) lo = max(lo, pos[a] + 1);
                    if (b >= ks) lo = max(lo, pos[b] + 1);
                    int hi = M - 1;
                    for (int ch : deps[u]) hi = min(hi, pos[ch] - 1);
                    if (lo >= hi) continue;

                    int opts[4];
                    int nopt = 0;
                    auto add_opt = [&](int p) {
                        if (p == idx || p < lo || p > hi) return;
                        for (int k = 0; k < nopt; k++) if (opts[k] == p) return;
                        opts[nopt++] = p;
                    };
                    add_opt(hi);
                    add_opt(lo);
                    if (cur.worst_idx < idx) add_opt(max(lo, min(hi, cur.worst_idx - 1)));
                    else add_opt(max(lo, min(hi, cur.worst_idx + 1)));
                    add_opt(lo + (int)(rng() % (hi - lo + 1)));
                    if (nopt == 0) continue;

                    int chosen_p = -1;
                    for (int k = 0; k < nopt; k++) {
                        int np = opts[k];
                        trial = cur_seq;
                        trial.erase(trial.begin() + idx);
                        trial.insert(trial.begin() + np, u);
                        PeakInfo pi = eval_order(trial);
                        double E = score_pi(pi);
                        if (E < chosen_E || (E == chosen_E && np > chosen_p)) {
                            chosen_E = E; chosen_p = np; chosen_seq = move(trial);
                        }
                    }
                }

                if (chosen_seq.empty()) continue;
                double temp = max(0.05, 4.0 * (1.0 - (double)it / (double)iters_per_restart));
                double dE = chosen_E - curE;
                if (dE <= 0.0 || ((double)(rng() % 10000) / 10000.0) < exp(-dE / temp)) {
                    cur_seq.swap(chosen_seq);
                    for (int k = 0; k < M; k++) pos[cur_seq[k]] = k;
                    cur = eval_order(cur_seq, &lu);
                    curE = score_pi(cur);
                    if (cur.max_peak < g_best_info.max_peak ||
                        (cur.max_peak == g_best_info.max_peak && cur.over48 < g_best_info.over48)) {
                        g_best_info = cur;
                        g_best_seq = cur_seq;
                    }
                }
            }
            if (g_best_info.max_peak <= 48) break;
        }
        seq = move(g_best_seq);
        return g_best_info.max_peak <= 48;
    }

    // Build topological order preserving kept_ord while optimally inserting new nodes & repairing >48 peak
    vector<int> build_pref_order_with_insert(const vector<u64> &kept_ord, int sa_iters, mt19937_64 &rng) const {
        int n = (int)nodes.size();
        int M = n - ks;
        if (M <= 0) return {};

        vector<vector<int>> deps(n);
        for (int u = ks; u < n; u++) {
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
        }

        // Preserve kept_ord only where acyclic among kept subset; drop conflicts/new nodes into to_ins
        vector<int> prio(n, -1);
        for (size_t i = 0; i < kept_ord.size(); i++) {
            auto it = val2idx.find(kept_ord[i]);
            if (it != val2idx.end() && it->second >= ks && prio[it->second] < 0) {
                prio[it->second] = (int)i;
            }
        }
        // Kahn's topological sort over kept nodes preserving kept_ord priority cleanly
        vector<char> is_kept(n, 0);
        for (int u = ks; u < n; u++) if (prio[u] >= 0) is_kept[u] = 1;
        vector<int> indeg_k(n, 0);
        for (int u = ks; u < n; u++) {
            if (!is_kept[u]) continue;
            if (is_kept[nodes[u].a]) indeg_k[u]++;
            if (nodes[u].b != nodes[u].a && is_kept[nodes[u].b]) indeg_k[u]++;
        }
        vector<pair<int, int>> ready_k;
        for (int u = ks; u < n; u++) if (is_kept[u] && indeg_k[u] == 0) ready_k.emplace_back(prio[u], u);
        make_heap(ready_k.begin(), ready_k.end(), greater<pair<int, int>>());
        vector<int> seq;
        seq.reserve(M);
        vector<char> in_seq(n, 0);
        while (!ready_k.empty()) {
            pop_heap(ready_k.begin(), ready_k.end(), greater<pair<int, int>>());
            int u = ready_k.back().second;
            ready_k.pop_back();
            // Also verify its non-start parents aren't missing from is_kept if any? No, to_ins handles them before u?
            // Actually wait: if a kept node depends on a new node w in to_ins, u cannot come before w if w is inserted later,
            // UNLESS u stays in seq and w gets hi <= pos[u], which only works if w has no ancestor AFTER u!
            // If ALL parents of u are either < ks, already in_seq, or we guarantee to_ins are ancestors with earlier parents:
            // putting kept nodes that depend on new nodes into seq is SAFE iff kept_ord itself had all kept ancestors earlier!
            seq.push_back(u);
            in_seq[u] = 1;
            for (int ch : deps[u]) {
                if (is_kept[ch] && --indeg_k[ch] == 0) {
                    ready_k.emplace_back(prio[ch], ch);
                    push_heap(ready_k.begin(), ready_k.end(), greater<pair<int, int>>());
                }
            }
        }

        vector<int> to_ins;
        for (int u = ks; u < n; u++) if (!in_seq[u]) to_ins.push_back(u);

        vector<int> pos(n, -1), trial;
        for (int u : to_ins) {
            fill(pos.begin(), pos.end(), -1);
            for (int i = 0; i < (int)seq.size(); i++) pos[seq[i]] = i;
            int lo = 0;
            if (nodes[u].a >= ks && pos[nodes[u].a] >= 0) lo = max(lo, pos[nodes[u].a] + 1);
            if (nodes[u].b >= ks && pos[nodes[u].b] >= 0) lo = max(lo, pos[nodes[u].b] + 1);
            int hi = (int)seq.size();
            for (int ch : deps[u]) if (pos[ch] >= 0) hi = min(hi, pos[ch]);
            if (lo > hi) {
                // Rare cycle inversion in kept_ord: minimal-perturbation backward-propagated Kahn's topo
                vector<int> eff_prio(n, M + n);
                for (int x = ks; x < n; x++) {
                    if (prio[x] >= 0) eff_prio[x] = prio[x];
                }
                for (int x = n - 1; x >= ks; --x) {
                    for (int ch : deps[x]) eff_prio[x] = min(eff_prio[x], eff_prio[ch]);
                }
                vector<int> indeg_all(n, 0);
                for (int x = ks; x < n; x++) {
                    if (nodes[x].a >= ks) indeg_all[x]++;
                    if (nodes[x].b >= ks && nodes[x].b != nodes[x].a) indeg_all[x]++;
                }
                using Key = tuple<int, int, int, int>;
                priority_queue<Key, vector<Key>, greater<Key>> pq;
                for (int x = ks; x < n; x++) {
                    if (indeg_all[x] == 0) {
                        pq.emplace(eff_prio[x], prio[x] < 0 ? 1 : 0, prio[x] >= 0 ? prio[x] : x, x);
                    }
                }
                seq.clear();
                while (!pq.empty()) {
                    auto [ep, isn, sp, x] = pq.top();
                    pq.pop();
                    seq.push_back(x);
                    for (int ch : deps[x]) {
                        if (--indeg_all[ch] == 0) {
                            pq.emplace(eff_prio[ch], prio[ch] < 0 ? 1 : 0, prio[ch] >= 0 ? prio[ch] : ch, ch);
                        }
                    }
                }
                break;
            }
            if (lo == hi) {
                seq.insert(seq.begin() + lo, u);
                continue;
            }
            vector<int> cands;
            auto add_c = [&](int p) {
                if (p < lo || p > hi) return;
                for (int x : cands) if (x == p) return;
                cands.push_back(p);
            };
            if (hi - lo <= 24) {
                for (int p = lo; p <= hi; p++) cands.push_back(p);
            } else {
                for (int d = 0; d <= 3; d++) { add_c(lo + d); add_c(hi - d); }
                for (int k = 1; k < 20; k++) add_c(lo + (int)((long long)(hi - lo) * k / 20));
            }
            int best_p = hi;
            PeakInfo best_pi{999, 999999, 0};
            for (int p : cands) {
                trial = seq;
                trial.insert(trial.begin() + p, u);
                PeakInfo pi = eval_order(trial);
                if (pi.max_peak < best_pi.max_peak ||
                    (pi.max_peak == best_pi.max_peak && (pi.over48 < best_pi.over48 ||
                     (pi.over48 == best_pi.over48 && p > best_p)))) {
                    best_pi = pi;
                    best_p = p;
                }
            }
            seq.insert(seq.begin() + best_p, u);
        }

        if (eval_order(seq).max_peak > 48 && sa_iters > 0) {
            fix_pressure_sa(seq, sa_iters, rng);
        }
        return seq;
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
            if (emit_exact_order(seq, prog_out)) return true;
        }
        return false;
    }

    // Rematerialisation-aware Belady fallback
    vector<Instr> compile_belady_mode(const vector<u64> &tord, bool planorder, bool seed_evict,
                                      const vector<u64> *seq_override = nullptr) const {
        map<u64, Step> defs;
        vector<u64> order;
        for (int u = ks; u < (int)nodes.size(); u++) {
            const auto &nd = nodes[u];
            defs[nd.v] = {nd.op, nodes[nd.a].v, nodes[nd.b].v, nd.v};
            order.push_back(nd.v);
        }
        set<u64> startset(starts.begin(), starts.end());
        vector<u64> seq;
        if (seq_override) {
            seq = *seq_override;
        } else if (planorder) {
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

    bool compile(vector<Instr> &prog_out, int reorder_tries, mt19937_64 &rng,
                 const vector<u64> *kept_ord = nullptr, const vector<u64> * /*new_vals*/ = nullptr,
                 int sa_iters = 4000) const {
        int n = (int)nodes.size();
        vector<int> best_low_seq;
        int best_low_pk = 9999;

        auto consider_seq = [&](vector<int> &seq, int siters) -> bool {
            PeakInfo pi = eval_order(seq);
            if (pi.max_peak <= 48 && emit_exact_order(seq, prog_out)) return true;
            if (siters > 0 && fix_pressure_sa(seq, siters, rng)) {
                if (emit_exact_order(seq, prog_out)) return true;
            }
            pi = eval_order(seq);
            if (pi.max_peak < best_low_pk) {
                best_low_pk = pi.max_peak;
                best_low_seq = seq;
            }
            return false;
        };

        // 0. If kept_ord given, insert new nodes at optimal min-peak positions + SA
        if (kept_ord && !kept_ord->empty()) {
            vector<int> seq_k_init = build_pref_order_with_insert(*kept_ord, 0, rng);
            vector<int> seq_k = seq_k_init;
            if (consider_seq(seq_k, sa_iters)) return true;
            if (sa_iters >= 8000) {
                for (int rep = 0; rep < 2; rep++) {
                    vector<int> seq_rep = seq_k_init;
                    if (consider_seq(seq_rep, sa_iters)) return true;
                }
            }
        }

        // 1. DAG's own topological order + SA
        {
            vector<int> seq_self;
            for (int u = ks; u < n; u++) seq_self.push_back(u);
            if (consider_seq(seq_self, sa_iters)) return true;
        }

        // 2. Greedy schedule_min_live (8 restarts)
        if (schedule_min_live(prog_out, 8, rng)) return true;

        // 3. LIFO-DFS + SA (2 restarts)
        {
            vector<vector<int>> deps(n);
            vector<int> indeg0(n, 0);
            for (int u = ks; u < n; u++) {
                deps[nodes[u].a].push_back(u);
                if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
                if (nodes[u].a >= ks) indeg0[u]++;
                if (nodes[u].b >= ks && nodes[u].b != nodes[u].a) indeg0[u]++;
            }
            for (int rep = 0; rep < 2; rep++) {
                vector<int> indeg = indeg0, ready;
                for (int u = ks; u < n; u++) if (indeg[u] == 0) ready.push_back(u);
                if (rep == 1) shuffle(ready.begin(), ready.end(), rng);
                vector<int> seq_l;
                seq_l.reserve(n - ks);
                while (!ready.empty()) {
                    int u = ready.back();
                    ready.pop_back();
                    seq_l.push_back(u);
                    for (int ch : deps[u]) {
                        if (--indeg[ch] == 0) ready.push_back(ch);
                    }
                }
                if (consider_seq(seq_l, max(2000, sa_iters / 2))) return true;
            }
        }

        // 4. Belady fallback (try lowest-peak sequence first, plus standard 4 modes + target reorder)
        vector<u64> tord;
        unordered_set<u64> tset(targets.begin(), targets.end());
        for (int u = ks; u < n; u++) {
            if (tset.count(nodes[u].v)) tord.push_back(nodes[u].v);
        }

        vector<Instr> best;
        bool have = false;
        int dag_len = n - ks;
        if (!best_low_seq.empty()) {
            vector<u64> seq_v;
            seq_v.reserve(best_low_seq.size());
            for (int u : best_low_seq) seq_v.push_back(nodes[u].v);
            for (bool se : {true, false}) {
                try {
                    auto p = compile_belady_mode(tord, true, se, &seq_v);
                    if (verify_prog(L, starts, targets, p).empty()) {
                        if (!have || p.size() < best.size()) { best = move(p); have = true; }
                    }
                } catch (...) {}
            }
        }
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
    int rej_strict = 0;
    int rej_eq = 0;
    array<u64, 6> move_wins{};  // 0..5
    vector<Step> harvested_recipes;
    SearchStats stats;
};

static vector<u64> extract_prog_order(const vector<u64> &starts, const vector<Instr> &prog) {
    array<u64, 48> reg{};
    array<bool, 48> def{};
    unordered_set<u64> have(starts.begin(), starts.end());
    for (size_t k = 0; k < starts.size(); k++) { reg[k] = starts[k]; def[k] = true; }
    vector<u64> ord;
    ord.reserve(prog.size());
    for (const auto &in : prog) {
        if (!def[in.i] || !def[in.j]) break;
        u64 r = 0;
        switch (in.op) {
            case '+': r = reg[in.i] + reg[in.j]; break;
            case '-': r = reg[in.i] - reg[in.j]; break;
            case '*': r = reg[in.i] * reg[in.j]; break;
            case '/': r = reg[in.i] / reg[in.j]; break;
        }
        reg[in.d] = r;
        def[in.d] = true;
        if (have.insert(r).second) ord.push_back(r);
    }
    return ord;
}

static vector<Step> reconstruct_subset(u64 L, const vector<u64> &subset_targets,
                                       PoolDict &wdict, const Params &P,
                                       const unordered_set<u64> &base_have,
                                       mt19937_64 &rng, SearchStats &stats,
                                       int max_budget, int pass_seed_idx) {
    unordered_set<u64> have = base_have;
    vector<Step> acc;
    vector<u64> rem;
    for (u64 t : subset_targets) if (!have.count(t)) rem.push_back(t);
    if (rem.empty()) return acc;

    auto mk0 = wdict.mark();
    DeepSearch ds(wdict, P, rng);
    ds.deep_beam = false;
    ds.extra_pool.clear();

    double lw = (pass_seed_idx == 0) ? P.la_weight : max(0.15, P.la_weight - 0.10);
    double jitter = (pass_seed_idx == 0) ? 0.0 : 0.35;

    while (!rem.empty()) {
        if ((int)acc.size() + (int)rem.size() >= max_budget) {
            acc.clear();
            break;
        }
        struct SubCand { u64 target; int cost; vector<Step> steps; };
        vector<SubCand> cands;
        int min_fast = 1 << 20;
        for (u64 t : rem) {
            int rem_b = min(9, max_budget - 1 - (int)acc.size() - ((int)rem.size() - 1));
            if (rem_b < 1) continue;
            auto sols = ds.search(t, rem_b, 2, 2, stats, 1);
            for (auto &s : sols) {
                min_fast = min(min_fast, s.cost);
                cands.push_back({t, s.cost, move(s.steps)});
            }
        }
        if (min_fast > 5) {
            for (size_t i = 0; i < rem.size() && (i < 6 || cands.empty()); i++) {
                int rem_b = min(14, max_budget - 1 - (int)acc.size() - ((int)rem.size() - 1));
                if (rem_b < 1) continue;
                auto sols = ds.search(rem[i], rem_b, 2, 1, stats, 2);
                for (auto &s : sols) cands.push_back({rem[i], s.cost, move(s.steps)});
            }
        }
        if (cands.empty()) {
            acc.clear();
            break;
        }
        stable_sort(cands.begin(), cands.end(), [](const SubCand &a, const SubCand &b) {
            return a.cost < b.cost;
        });
        int K = min<int>(6, (int)cands.size());
        int bi = 0;
        double bs = 1e18;
        if (K > 1 && rem.size() > 1) {
            for (int k = 0; k < K; k++) {
                vector<u64> add_v;
                for (const auto &st : cands[k].steps) {
                    if (!have.count(st.r)) add_v.push_back(st.r);
                }
                auto mk = wdict.mark();
                wdict.add_leaves_inc(add_v);
                for (u64 v : add_v) ds.extra_pool.insert(v);
                double rem_est = 0.0;
                for (u64 x : rem) {
                    if (have.count(x) || ds.extra_pool.count(x)) continue;
                    rem_est += ds.estimate_cost(x);
                }
                for (u64 v : add_v) ds.extra_pool.erase(v);
                wdict.rollback(mk);
                double sc = (double)cands[k].cost + lw * rem_est;
                if (jitter > 0.0) sc += jitter * ((double)(rng() % 1000) / 1000.0);
                if (sc < bs) { bs = sc; bi = k; }
            }
        }
        vector<u64> newly;
        for (const auto &st : cands[bi].steps) {
            if (!have.count(st.r)) {
                u64 chk = 0;
                if (!have.count(st.a) || !have.count(st.b) || !apply_op(st.op, st.a, st.b, L, chk) || chk != st.r) {
                    acc.clear();
                    rem.clear();
                    break;
                }
                have.insert(st.r);
                ds.extra_pool.insert(st.r);
                newly.push_back(st.r);
                acc.push_back(st);
            }
        }
        if (rem.empty()) break;
        if (!newly.empty() && rem.size() > 1) wdict.add_leaves_inc(newly);

        vector<u64> rem2;
        for (u64 t : rem) if (!have.count(t)) rem2.push_back(t);
        rem.swap(rem2);
    }

    ds.extra_pool.clear();
    wdict.rollback(mk0);
    return acc;
}

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
    vector<u64> cur_ord = extract_prog_order(starts, res.prog);

    {
        vector<Instr> init_opt;
        if (best_dag.compile(init_opt, P.reorder_tries, rng, &cur_ord, nullptr, 4000) &&
            init_opt.size() < res.prog.size()) {
            res.prog = move(init_opt);
            cur_ord = extract_prog_order(starts, res.prog);
            res.strict_acc++;
        }
    }

    int nt = (int)cur_dag.targets.size();
    if (nt == 0) return res;

    // Persistent full-pool PoolDict rebuilt ONLY when cur_dag changes!
    PoolDict pdict;
    vector<int> pdict_s0_dag;
    auto rebuild_pdict = [&]() {
        int n = (int)cur_dag.nodes.size();
        vector<u64> all_built;
        all_built.reserve(n);
        unordered_set<u64> tset(cur_dag.targets.begin(), cur_dag.targets.end());
        unordered_map<u64, int> fanout;
        unordered_set<u64> near_vals;
        for (int u = 0; u < n; u++) {
            all_built.push_back(cur_dag.nodes[u].v);
            fanout[cur_dag.nodes[u].v] = (int)cur_dag.users[u].size();
        }
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, all_built, tset, fanout, near_vals, P.n_leaves,
                      leaves, lcosts, s1_cnt, done_t, false);
        pdict.build(L, starts, all_built, leaves, lcosts, s1_cnt, done_t, P.s2_leaves, P.s2_cap);
        pdict_s0_dag.resize(pdict.n0);
        for (int i = 0; i < pdict.n0; i++) {
            auto it = cur_dag.val2idx.find(pdict.e[i].v);
            pdict_s0_dag[i] = (it != cur_dag.val2idx.end()) ? it->second : 0;
        }
        res.stats.sec[0] += now_sec() - tb;
    };
    rebuild_pdict();

    PoolDict wdict;  // smaller fast scratch dict for Move B / D / E

    double t0 = now_sec();
    int stall = 0;
    int cur_prog_len = (int)res.prog.size();
    bool analysed = true;

    auto try_commit = [&](const vector<char> &inR, const vector<char> & /*excl*/,
                          const vector<Step> &new_steps, int engine_id, int move_type,
                          int slack_budget) -> int {
        map<u64, Step> defs;
        vector<u64> kept_ord;
        for (int u = cur_dag.ks; u < (int)cur_dag.nodes.size(); u++) {
            if (inR[u]) continue;
            const auto &nd = cur_dag.nodes[u];
            defs[nd.v] = {nd.op, cur_dag.nodes[nd.a].v, cur_dag.nodes[nd.b].v, nd.v};
        }
        for (u64 v : cur_ord) {
            if (defs.count(v)) kept_ord.push_back(v);
        }
        vector<u64> new_vals;
        for (const auto &s : new_steps) {
            if (!defs.count(s.r)) {
                defs[s.r] = s;
                new_vals.push_back(s.r);
            }
        }
        vector<u64> pref_ord = kept_ord;
        for (u64 v : new_vals) pref_ord.push_back(v);

        Dag ndag;
        ndag.init(L, starts, all_targets);
        if (!ndag.from_defs(defs, pref_ord)) return -1;

        int new_dag_len = (int)ndag.nodes.size() - ndag.ks;
        if (new_dag_len > cur_prog_len + slack_budget) return -1;

        vector<Instr> nprog;
        int sa_it = (new_dag_len < (int)res.prog.size()) ? 15000 : 3500;
        if (!ndag.compile(nprog, P.reorder_tries, rng, &kept_ord, &new_vals, sa_it)) return -1;
        int nlen = (int)nprog.size();
        if (nlen > cur_prog_len + slack_budget) {
            if (new_dag_len < (int)res.prog.size()) res.rej_strict++;
            else if (new_dag_len <= cur_prog_len + slack_budget) res.rej_eq++;
            return -1;
        }
        if (nlen < (int)res.prog.size()) {
            res.prog = nprog;
            best_dag = ndag;
            best_dag.analyse();
            cur_dag = best_dag;
            cur_ord = extract_prog_order(starts, res.prog);
            cur_prog_len = nlen;
            analysed = true;
            res.strict_acc++;
            if (engine_id >= 1 && engine_id <= 5) res.stats.wins[engine_id]++;
            if (move_type >= 0 && move_type < 5) res.move_wins[move_type]++;
            stall = 0;
            rebuild_pdict();
            if (P.verbose > 1) {
                fprintf(stderr, "  [%s] iter=%d STRICT %d (mv=%c eng=%d dag=%d)\n",
                        pid.c_str(), res.iters, nlen, 'A' + move_type, engine_id, new_dag_len);
            }
            return 1;
        }
        if (nlen <= cur_prog_len + slack_budget) {
            cur_dag = move(ndag);
            cur_dag.analyse();
            cur_ord = extract_prog_order(starts, nprog);
            cur_prog_len = nlen;
            analysed = true;
            res.eq_acc++;
            rebuild_pdict();
            return 0;
        }
        return -1;
    };

    auto build_window_dict = [&](const vector<char> &excl, const vector<char> &inR,
                                 size_t w_cap, int w_s1, int w_s2, unordered_set<u64> &out_all_built) {
        int n = (int)cur_dag.nodes.size();
        vector<u64> all_built;
        out_all_built.clear();
        unordered_set<u64> tset(cur_dag.targets.begin(), cur_dag.targets.end());
        unordered_map<u64, int> fanout;
        unordered_set<u64> near_vals;
        for (int u = 0; u < n; u++) {
            if (!excl[u]) {
                all_built.push_back(cur_dag.nodes[u].v);
                out_all_built.insert(cur_dag.nodes[u].v);
                fanout[cur_dag.nodes[u].v] = (int)cur_dag.users[u].size();
            } else if (inR[u] && u >= cur_dag.ks) {
                int a = cur_dag.nodes[u].a, b = cur_dag.nodes[u].b;
                if (!excl[a]) near_vals.insert(cur_dag.nodes[a].v);
                if (!excl[b]) near_vals.insert(cur_dag.nodes[b].v);
            }
        }
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, all_built, tset, fanout, near_vals, w_s1, leaves, lcosts, s1_cnt, done_t, false);
        wdict.build(L, starts, all_built, leaves, lcosts, s1_cnt, done_t, w_s2, w_cap);
        res.stats.sec[0] += now_sec() - tb;
    };

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

    auto collect_excl_vals = [&](const vector<char> &excl, const vector<char> &inR) {
        unordered_set<u64> ev;
        for (size_t u = cur_dag.ks; u < cur_dag.nodes.size(); u++) {
            if (excl[u] && !inR[u]) ev.insert(cur_dag.nodes[u].v);
        }
        return ev;
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
                cur_ord = extract_prog_order(starts, res.prog);
                cur_prog_len = (int)res.prog.size();
                analysed = true;
                stall = 0;
                rebuild_pdict();
            } else {
                kick_slack = 1;
            }
        }

        int roll = (int)(rng() % 100);
        // Move probabilities: A=25%, B=30%, C=15%, D=15%, E=15% (plus boost E when stalling)
        bool force_E = (nt >= 5 && stall > 0 && (stall % 15 == 0));

        if (!force_E && (roll < 25 || nt == 1)) {
            // Move Type A: k=1 single-target deep resynthesis on persistent pdict (ZERO rebuild!)
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
            vector<char> excl_s0(pdict.n0, 0);
            for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[pdict_s0_dag[i]];
            auto excl_vals = collect_excl_vals(excl, inR);

            DeepSearch ds(pdict, P, rng);
            ds.deep_beam = false;
            ds.excl_s0 = &excl_s0;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
            auto sols = ds.search(cur_dag.targets[ti], budget, 1, 0, res.stats, 2);
            if (!sols.empty()) {
                int r = try_commit(inR, excl, sols[0].steps, sols[0].engine, 0,
                                   kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else if (!force_E && roll < 55) {
            // Move Type B: k = 2..lns_kmax joint resynthesis with A=3 branching lookahead
            u64 qm = 0;
            if (rng() % 2 == 0) {
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
            unordered_set<u64> w_have;
            build_window_dict(excl, inR, min<size_t>(P.s2_cap, 200000), min(P.n_leaves, 88),
                              min(P.s2_leaves, 32), w_have);
            auto excl_vals = collect_excl_vals(excl, inR);
            DeepSearch ds(wdict, P, rng);
            ds.deep_beam = false;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;

            int best_used = budget + 1;
            vector<Step> best_steps;
            int best_eng = 0;

            int n_ord = Q.size() == 2 ? 2 : min(3, (int)Q.size());
            for (int ord = 0; ord < n_ord; ord++) {
                if (ord == 1 && Q.size() == 2) swap(Q[0], Q[1]);
                else if (ord > 0) shuffle(Q.begin(), Q.end(), rng);

                int rem_first = best_used - 1 - ((int)Q.size() - 1);
                if (rem_first < 1) break;
                int alts0 = (ord == 0 && Q.size() <= 3) ? min(3, P.la_alts) : 1;
                auto sols0 = ds.search(cur_dag.targets[Q[0]], rem_first, alts0, 1, res.stats, ord == 0 ? 2 : 1);
                if (sols0.empty()) continue;

                for (size_t sidx = 0; sidx < sols0.size(); sidx++) {
                    const auto &s0 = sols0[sidx];
                    if ((int)s0.steps.size() + ((int)Q.size() - 1) >= best_used) continue;
                    vector<Step> acc = s0.steps;
                    vector<u64> new_v;
                    for (const auto &st : s0.steps) new_v.push_back(st.r);
                    auto mk = wdict.mark();
                    wdict.add_leaves_inc(new_v);
                    ds.extra_pool.clear();
                    for (u64 v : new_v) ds.extra_pool.insert(v);

                    bool ok = true;
                    int sub_mode = (ord == 0 && sidx == 0) ? 2 : 1;
                    for (size_t qi = 1; qi < Q.size(); qi++) {
                        u64 tq = cur_dag.targets[Q[qi]];
                        if (wdict.in_pool(tq) || ds.extra_pool.count(tq)) continue;
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
                        if (qi + 1 < Q.size() && !add_v.empty()) wdict.add_leaves_inc(add_v);
                    }
                    ds.extra_pool.clear();
                    wdict.rollback(mk);

                    if (ok && (int)acc.size() < best_used) {
                        best_used = (int)acc.size();
                        best_steps = move(acc);
                        best_eng = s0.engine;
                    }
                }
            }
            if (!best_steps.empty()) {
                int r = try_commit(inR, excl, best_steps, best_eng, 1,
                                   kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else if (!force_E && roll < 70) {
            // Move Type C: Helper-dominated cone replacement on persistent pdict (ZERO rebuild!)
            unordered_set<int> isT(cur_dag.tnode.begin(), cur_dag.tnode.end());
            vector<int> helpers;
            for (int u = cur_dag.ks; u < n; u++) {
                if (!isT.count(u) && __builtin_popcountll(cur_dag.owners[u]) >= 2) helpers.push_back(u);
            }
            if (helpers.empty()) { stall++; continue; }
            int hv = helpers[rng() % helpers.size()];

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
            vector<char> excl_s0(pdict.n0, 0);
            for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[pdict_s0_dag[i]];
            auto excl_vals = collect_excl_vals(excl, inR);

            DeepSearch ds(pdict, P, rng);
            ds.deep_beam = false;
            ds.excl_s0 = &excl_s0;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
            auto sols = ds.search(cur_dag.nodes[hv].v, budget, 1, 0, res.stats, 2);
            if (!sols.empty()) {
                int r = try_commit(inR, excl, sols[0].steps, sols[0].engine, 2,
                                   kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else if (!force_E && roll < 85) {
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
            unordered_set<u64> w_have;
            build_window_dict(excl, inR, min<size_t>(P.s2_cap, 200000), min(P.n_leaves, 88),
                              min(P.s2_leaves, 32), w_have);
            auto excl_vals = collect_excl_vals(excl, inR);
            DeepSearch ds(wdict, P, rng);
            ds.deep_beam = false;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;

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
                if (wdict.in_pool(B)) continue;
                auto bs = ds.search(B, min(best_used - 3, 6), 1, 0, res.stats, 1);
                if (bs.empty()) continue;

                vector<Step> acc = bs[0].steps;
                vector<u64> new_v;
                for (const auto &st : acc) new_v.push_back(st.r);
                auto mk = wdict.mark();
                wdict.add_leaves_inc(new_v);
                ds.extra_pool.clear();
                for (u64 v : new_v) ds.extra_pool.insert(v);

                int rem1 = best_used - 1 - (int)acc.size() - 1;
                bool ok1 = wdict.in_pool(t1) || ds.extra_pool.count(t1);
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
                        if (!v1.empty()) wdict.add_leaves_inc(v1);
                    }
                    bool ok2 = wdict.in_pool(t2) || ds.extra_pool.count(t2);
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
                wdict.rollback(mk);
            }
            if (!best_steps.empty()) {
                int r = try_commit(inR, excl, best_steps, best_eng, 3,
                                   kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;

        } else {
            // Move Type E: Large-window hub/slice ruin-and-recreate (k = 5..12)
            if (nt < 5) { stall++; continue; }
            u64 qm = 0;
            if (rng() % 2 == 0) {
                vector<u64> hubs;
                for (int u = cur_dag.ks; u < n; u++) {
                    u64 m = cur_dag.owners[u];
                    int pc = __builtin_popcountll(m);
                    if (pc >= 4 && pc <= 12) hubs.push_back(m);
                }
                if (!hubs.empty()) qm = hubs[rng() % hubs.size()];
            }
            if (qm == 0) {
                int k = 5 + (int)(rng() % min(8, max(1, nt - 4)));
                if (rng() % 2 == 0) {
                    // Contiguous execution-order window of targets
                    vector<pair<int, int>> ord_t;
                    for (int ti = 0; ti < nt; ti++) ord_t.emplace_back(cur_dag.tnode[ti], ti);
                    sort(ord_t.begin(), ord_t.end());
                    int st = (int)(rng() % max(1, nt - k + 1));
                    for (int j = 0; j < k && st + j < nt; j++) qm |= (1ULL << ord_t[st + j].second);
                } else {
                    // Contiguous sorted value window
                    int st = (int)(rng() % max(1, nt - k + 1));
                    for (int j = 0; j < k && st + j < nt; j++) qm |= (1ULL << (st + j));
                }
            }
            // Iteratively drop any target whose node is used by targets outside qm
            bool changed = true;
            while (changed) {
                changed = false;
                for (int ti = 0; ti < nt; ti++) {
                    if (((qm >> ti) & 1) && (cur_dag.owners[cur_dag.tnode[ti]] & ~qm) != 0) {
                        qm &= ~(1ULL << ti);
                        changed = true;
                    }
                }
            }
            vector<int> Q;
            vector<u64> Q_vals;
            for (int ti = 0; ti < nt; ti++) {
                if ((qm >> ti) & 1) {
                    Q.push_back(ti);
                    Q_vals.push_back(cur_dag.targets[ti]);
                }
            }
            if (Q.size() < 4) { stall++; continue; }

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            int budget = rs - ((accept_eq || kick_slack) ? 0 : 1) + kick_slack;
            if (rs < 8 || budget < (int)Q.size()) { stall++; continue; }

            vector<int> Q_nodes;
            for (int ti : Q) Q_nodes.push_back(cur_dag.tnode[ti]);
            vector<char> excl = compute_excl(inR, Q_nodes);
            unordered_set<u64> w_have;
            build_window_dict(excl, inR, min<size_t>(P.s2_cap, 200000), min(P.n_leaves, 88),
                              min(P.s2_leaves, 32), w_have);

            int best_used = budget + 1;
            vector<Step> best_steps;
            for (int pass = 0; pass < 2; pass++) {
                auto sub = reconstruct_subset(L, Q_vals, wdict, P, w_have, rng, res.stats, best_used, pass);
                if (!sub.empty() && (int)sub.size() < best_used) {
                    best_used = (int)sub.size();
                    best_steps = move(sub);
                }
            }
            if (!best_steps.empty()) {
                int r = try_commit(inR, excl, best_steps, 5, 4,
                                   kick_slack ? 1 : (accept_eq ? 0 : -1));
                if (r <= 0) stall++;
            } else stall++;
        }
    }
    return res;
}

// ---------------------------------------------------------------- Mode 1b: Soft-Peak Herfindahl DAG Simulated Annealing (-sa)
static LnsResult sa_optimize(u64 L, const vector<u64> &starts, const vector<u64> &all_targets,
                             const vector<Instr> &init_prog, const Params &P,
                             const string &pid, mt19937_64 &rng) {
    LnsResult res;
    res.prog = init_prog;
    Dag best_dag, cur_dag;
    best_dag.init(L, starts, all_targets);
    if (!best_dag.from_prog(init_prog)) return res;
    best_dag.analyse();
    cur_dag = best_dag;
    vector<u64> cur_ord = extract_prog_order(starts, res.prog);

    {
        vector<Instr> init_opt;
        if (best_dag.compile(init_opt, P.reorder_tries, rng, &cur_ord, nullptr, 4000) &&
            init_opt.size() < res.prog.size()) {
            res.prog = move(init_opt);
            cur_ord = extract_prog_order(starts, res.prog);
            res.strict_acc++;
        }
    }

    int nt = (int)cur_dag.targets.size();
    if (nt == 0) return res;

    auto dag_energy = [&](const Dag &dg, const Dag::PeakInfo &pi) -> double {
        int n = (int)dg.nodes.size();
        int dag_len = n - dg.ks;
        vector<int> fcnt(n, 0);
        for (int u = dg.ks; u < n; u++) {
            fcnt[dg.nodes[u].a]++;
            if (dg.nodes[u].b != dg.nodes[u].a) fcnt[dg.nodes[u].b]++;
        }
        double share_sum = 0.0;
        for (int u = dg.ks; u < n; u++) {
            int f = fcnt[u];
            if (f >= 2) share_sum += (double)f * (double)(f - 1);
        }
        return (double)dag_len + 0.40 * (double)max(0, pi.max_peak - 48) + 0.025 * (double)pi.over48 - 0.0004 * share_sum;
    };

    auto dag_sig = [&](const Dag &dg) -> u64 {
        u64 h = 0x9e3779b97f4a7c15ULL ^ (u64)(dg.nodes.size() - dg.ks);
        for (int u = dg.ks; u < (int)dg.nodes.size(); u++) {
            const auto &nd = dg.nodes[u];
            u64 z = mix64(nd.v ^ ((u64)(unsigned char)nd.op << 56));
            z ^= mix64(dg.nodes[nd.a].v * 1315423911ULL + dg.nodes[nd.b].v);
            h ^= z;
        }
        return h;
    };

    vector<int> cur_seq = cur_dag.build_pref_order_with_insert(cur_ord, 0, rng);
    Dag::PeakInfo cur_peak = cur_dag.eval_order(cur_seq);
    double cur_E = dag_energy(cur_dag, cur_peak);
    u64 cur_hash = dag_sig(cur_dag);
    int best_unc_pk = 999;
    unordered_set<u64> failed_compile_hashes;

    bool do_harvest = !P.dump_recipes_path.empty();
    unordered_set<u64> seen_rec_keys;
    auto harvest_dag = [&](const Dag &dg) {
        if (!do_harvest || res.harvested_recipes.size() >= 2500) return;
        if ((int)dg.nodes.size() - dg.ks > (int)init_prog.size() + 2) return;
        for (int u = dg.ks; u < (int)dg.nodes.size(); u++) {
            const auto &nd = dg.nodes[u];
            u64 av = dg.nodes[nd.a].v, bv = dg.nodes[nd.b].v;
            if ((nd.op == '+' || nd.op == '*') && av > bv) swap(av, bv);
            u64 k = mix64(nd.v ^ ((u64)(unsigned char)nd.op << 56)) ^ mix64(av * 0x9e3779b97f4a7c15ULL + bv);
            if (seen_rec_keys.insert(k).second) {
                res.harvested_recipes.push_back({nd.op, av, bv, nd.v});
                if (res.harvested_recipes.size() >= 2500) break;
            }
        }
    };
    harvest_dag(cur_dag);

    const int sa_n1 = min(P.n_leaves, 88);
    const int sa_k2 = min(P.s2_leaves, 24);
    const size_t sa_cap = min<size_t>(P.s2_cap, 100000);

    PoolDict pdict;
    vector<int> pdict_s0_dag;
    auto rebuild_pdict = [&]() {
        int n = (int)cur_dag.nodes.size();
        vector<u64> all_built;
        all_built.reserve(n);
        unordered_set<u64> tset(cur_dag.targets.begin(), cur_dag.targets.end());
        unordered_map<u64, int> fanout;
        unordered_set<u64> near_vals;
        for (int u = 0; u < n; u++) {
            all_built.push_back(cur_dag.nodes[u].v);
            fanout[cur_dag.nodes[u].v] = (int)cur_dag.users[u].size();
        }
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, all_built, tset, fanout, near_vals, sa_n1,
                      leaves, lcosts, s1_cnt, done_t, false);
        pdict.build(L, starts, all_built, leaves, lcosts, s1_cnt, done_t, sa_k2, sa_cap);
        pdict_s0_dag.resize(pdict.n0);
        for (int i = 0; i < pdict.n0; i++) {
            auto it = cur_dag.val2idx.find(pdict.e[i].v);
            pdict_s0_dag[i] = (it != cur_dag.val2idx.end()) ? it->second : 0;
        }
        res.stats.sec[0] += now_sec() - tb;
    };
    rebuild_pdict();

    PoolDict wdict;
    auto build_window_dict = [&](const vector<char> &excl, const vector<char> &inR,
                                 size_t w_cap, int w_s1, int w_s2) {
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
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, all_built, tset, fanout, near_vals, w_s1, leaves, lcosts, s1_cnt, done_t, false);
        wdict.build(L, starts, all_built, leaves, lcosts, s1_cnt, done_t, w_s2, w_cap);
        res.stats.sec[0] += now_sec() - tb;
    };

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

    auto collect_excl_vals = [&](const vector<char> &excl, const vector<char> &inR) {
        unordered_set<u64> ev;
        for (size_t u = cur_dag.ks; u < cur_dag.nodes.size(); u++) {
            if (excl[u] && !inR[u]) ev.insert(cur_dag.nodes[u].v);
        }
        return ev;
    };

    struct CandState {
        bool valid = false;
        Dag dag;
        vector<u64> kept_ord;
        vector<u64> new_vals;
        vector<int> seq;
        Dag::PeakInfo peak{};
        double E = 1e18;
        u64 h = 0;
        int eng = 0;
    };

    auto eval_cand_dag = [&](const vector<char> &inR, const vector<Step> &new_steps, int engine_id) -> CandState {
        CandState cs;
        map<u64, Step> defs;
        for (int u = cur_dag.ks; u < (int)cur_dag.nodes.size(); u++) {
            if (inR[u]) continue;
            const auto &nd = cur_dag.nodes[u];
            defs[nd.v] = {nd.op, cur_dag.nodes[nd.a].v, cur_dag.nodes[nd.b].v, nd.v};
        }
        for (u64 v : cur_ord) {
            if (defs.count(v)) cs.kept_ord.push_back(v);
        }
        for (const auto &s : new_steps) {
            if (!defs.count(s.r)) {
                defs[s.r] = s;
                cs.new_vals.push_back(s.r);
            }
        }
        vector<u64> pref_ord = cs.kept_ord;
        for (u64 v : cs.new_vals) pref_ord.push_back(v);

        cs.dag.init(L, starts, all_targets);
        if (!cs.dag.from_defs(defs, pref_ord)) return cs;
        int new_dag_len = (int)cs.dag.nodes.size() - cs.dag.ks;
        if (new_dag_len > (int)res.prog.size() + 3) return cs;
        cs.h = dag_sig(cs.dag);
        if (cs.h == cur_hash) return cs;

        cs.seq = cs.dag.build_pref_order_with_insert(cs.kept_ord, 0, rng);
        cs.peak = cs.dag.eval_order(cs.seq);
        cs.E = dag_energy(cs.dag, cs.peak);
        cs.eng = engine_id;
        cs.valid = true;
        return cs;
    };

    auto commit_cand_sa = [&](CandState &cs, int move_type, double temp) -> bool {
        if (!cs.valid) return false;
        int cur_dag_len = (int)cur_dag.nodes.size() - cur_dag.ks;
        int new_dag_len = (int)cs.dag.nodes.size() - cs.dag.ks;
        bool try_win = (new_dag_len < (int)res.prog.size()) ||
                       (cur_dag_len < (int)res.prog.size() && new_dag_len == cur_dag_len && cs.peak.max_peak < best_unc_pk);
        if (try_win && !failed_compile_hashes.count(cs.h)) {
            if (new_dag_len < (int)res.prog.size() && cs.peak.max_peak < best_unc_pk) {
                best_unc_pk = cs.peak.max_peak;
            }
            vector<Instr> nprog;
            if (cs.dag.compile(nprog, P.reorder_tries, rng, &cs.kept_ord, &cs.new_vals, 20000) &&
                nprog.size() < res.prog.size() &&
                verify_prog(L, starts, all_targets, nprog).empty()) {
                res.prog = nprog;
                best_dag = cs.dag;
                best_dag.analyse();
                cur_dag = best_dag;
                cur_ord = extract_prog_order(starts, res.prog);
                cur_seq = cur_dag.build_pref_order_with_insert(cur_ord, 0, rng);
                cur_peak = cur_dag.eval_order(cur_seq);
                cur_E = dag_energy(cur_dag, cur_peak);
                cur_hash = dag_sig(cur_dag);
                best_unc_pk = 999;
                failed_compile_hashes.clear();
                res.strict_acc++;
                if (cs.eng >= 1 && cs.eng <= 5) res.stats.wins[cs.eng]++;
                if (move_type >= 0 && move_type < 6) res.move_wins[move_type]++;
                harvest_dag(cur_dag);
                rebuild_pdict();
                if (P.verbose > 1) {
                    fprintf(stderr, "  [%s] SA iter=%d STRICT %zu (mv=%d eng=%d dag=%d pk=%d)\n",
                            pid.c_str(), res.iters, res.prog.size(), move_type, cs.eng, new_dag_len, cs.peak.max_peak);
                }
                return true;
            } else {
                failed_compile_hashes.insert(cs.h);
                if (new_dag_len < (int)res.prog.size()) res.rej_strict++;
            }
        }

        double dE = cs.E - cur_E;
        double u01 = (double)(rng() % 100000) / 100000.0;
        if (dE <= 0.0 || (new_dag_len <= (int)res.prog.size() + 3 && u01 < exp(-dE / max(0.01, temp)))) {
            cur_dag = move(cs.dag);
            cur_dag.analyse();
            cur_ord.clear();
            cur_ord.reserve(cs.seq.size());
            for (int u : cs.seq) cur_ord.push_back(cur_dag.nodes[u].v);
            cur_peak = cs.peak;
            cur_E = cs.E;
            cur_hash = cs.h;
            res.eq_acc++;
            harvest_dag(cur_dag);
            rebuild_pdict();
            return true;
        }
        return false;
    };

    double t0 = now_sec();
    double total_sec = max(0.1, P.lns_sec);
    double cycle_len = max(0.25, total_sec / 4.0);
    int last_cyc = 0;
    unordered_set<u64> sset(starts.begin(), starts.end());

    while (true) {
        double el = now_sec() - t0;
        if (el >= total_sec) break;
        int cyc = min(3, (int)(el / cycle_len));
        if (cyc > last_cyc) {
            last_cyc = cyc;
            int cur_dag_len = (int)cur_dag.nodes.size() - cur_dag.ks;
            int best_dag_len = (int)best_dag.nodes.size() - best_dag.ks;
            if (cur_dag_len > best_dag_len + 2) {
                cur_dag = best_dag;
                cur_ord = extract_prog_order(starts, res.prog);
                cur_seq = cur_dag.build_pref_order_with_insert(cur_ord, 0, rng);
                cur_peak = cur_dag.eval_order(cur_seq);
                cur_E = dag_energy(cur_dag, cur_peak);
                cur_hash = dag_sig(cur_dag);
                rebuild_pdict();
            }
        }
        double frac = min(1.0, (el - cyc * cycle_len) / cycle_len);
        double temp = 1.2 * pow(0.04 / 1.2, frac);

        res.iters++;
        int n = (int)cur_dag.nodes.size();
        int roll = (int)(rng() % 100);

        if (roll < 35 || nt == 1) {
            // Move 0: k=1 single-target SA on persistent pdict (ZERO dict rebuild)
            vector<int> wts(nt, 1);
            int wsum = 0;
            for (int ti = 0; ti < nt; ti++) {
                u64 bit = 1ULL << ti;
                int c = 0;
                for (int u = cur_dag.ks; u < n; u++)
                    if (cur_dag.owners[u] && (cur_dag.owners[u] & ~bit) == 0) c++;
                wts[ti] = max(1, c + 1);
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
            int up = (temp > 0.35 ? 2 : 1);
            int budget = rs + up;
            if (budget < 1) continue;

            vector<char> excl = compute_excl(inR, {cur_dag.tnode[ti]});
            vector<char> excl_s0(pdict.n0, 0);
            for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[pdict_s0_dag[i]];
            auto excl_vals = collect_excl_vals(excl, inR);

            DeepSearch ds(pdict, P, rng);
            ds.deep_beam = false;
            ds.excl_s0 = &excl_s0;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
            int smode = (rng() % 5 == 0) ? 2 : 1;
            auto sols = ds.search(cur_dag.targets[ti], budget, 6, 1, res.stats, smode);

            CandState best_cand;
            for (const auto &sol : sols) {
                CandState cs = eval_cand_dag(inR, sol.steps, sol.engine);
                if (cs.valid && cs.E < best_cand.E) best_cand = move(cs);
            }
            if (best_cand.valid) commit_cand_sa(best_cand, 0, temp);

        } else if (roll < 55) {
            // Move 5: Helper-rooted dom_cut(u) subtree destroy (rs in [2, 12], DeepSearch mode 2)
            unordered_set<int> isT(cur_dag.tnode.begin(), cur_dag.tnode.end());
            vector<int> shared_u;
            for (int u = cur_dag.ks; u < n; u++) {
                if (!isT.count(u) && cur_dag.users[u].size() >= 2) shared_u.push_back(u);
            }
            if (shared_u.empty()) continue;

            int picked_u = -1, rs = 0;
            vector<char> inR(n, 0);
            for (int att = 0; att < 8; att++) {
                int u = shared_u[rng() % shared_u.size()];
                vector<char> vis(n, 0);
                vector<int> st;
                for (int tn : cur_dag.tnode) if (tn != u) st.push_back(tn);
                while (!st.empty()) {
                    int x = st.back(); st.pop_back();
                    if (x < cur_dag.ks || x == u || vis[x]) continue;
                    vis[x] = 1;
                    st.push_back(cur_dag.nodes[x].a);
                    st.push_back(cur_dag.nodes[x].b);
                }
                int cnt = 0;
                fill(inR.begin(), inR.end(), 0);
                for (int x = cur_dag.ks; x < n; x++) if (!vis[x]) { inR[x] = 1; cnt++; }
                if (cnt >= 2 && cnt <= 12) { picked_u = u; rs = cnt; break; }
            }
            if (picked_u < 0) continue;

            vector<char> excl = compute_excl(inR, {picked_u});
            vector<char> excl_s0(pdict.n0, 0);
            for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[pdict_s0_dag[i]];
            auto excl_vals = collect_excl_vals(excl, inR);

            DeepSearch ds(pdict, P, rng);
            ds.deep_beam = false;
            ds.excl_s0 = &excl_s0;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
            auto sols = ds.search(cur_dag.nodes[picked_u].v, rs + 1, 6, 1, res.stats, 2);

            vector<CandState> cands;
            for (const auto &sol : sols) {
                CandState cs = eval_cand_dag(inR, sol.steps, 6);
                if (cs.valid) cands.push_back(move(cs));
            }
            sort(cands.begin(), cands.end(), [](const CandState &a, const CandState &b) { return a.E < b.E; });
            for (auto &cs : cands) {
                if (commit_cand_sa(cs, 5, temp)) break;
            }

        } else if (roll < 68) {
            // Move 1: Shared-helper cone replacement (users >= 2) on persistent pdict
            unordered_set<int> isT(cur_dag.tnode.begin(), cur_dag.tnode.end());
            vector<int> helpers;
            for (int u = cur_dag.ks; u < n; u++) {
                if (!isT.count(u) && __builtin_popcountll(cur_dag.owners[u]) >= 2) helpers.push_back(u);
            }
            if (helpers.empty()) continue;
            int hv = helpers[rng() % helpers.size()];

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
            int budget = rs + 1;

            vector<char> excl = compute_excl(inR, {hv});
            vector<char> excl_s0(pdict.n0, 0);
            for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[pdict_s0_dag[i]];
            auto excl_vals = collect_excl_vals(excl, inR);

            DeepSearch ds(pdict, P, rng);
            ds.deep_beam = false;
            ds.excl_s0 = &excl_s0;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
            int smode = (rng() % 5 == 0) ? 2 : 1;
            auto sols = ds.search(cur_dag.nodes[hv].v, budget, 4, 1, res.stats, smode);

            CandState best_cand;
            for (const auto &sol : sols) {
                CandState cs = eval_cand_dag(inR, sol.steps, sol.engine);
                if (cs.valid && cs.E < best_cand.E) best_cand = move(cs);
            }
            if (best_cand.valid) commit_cand_sa(best_cand, 1, temp);

        } else if (roll < 85) {
            // Move 2: Fast k=2 pair resynthesis with compact wdict
            u64 qm = 0;
            if (rng() % 2 == 0) {
                vector<u64> masks;
                for (int u = cur_dag.ks; u < n; u++) {
                    u64 m = cur_dag.owners[u];
                    if (__builtin_popcountll(m) == 2) masks.push_back(m);
                }
                if (!masks.empty()) qm = masks[rng() % masks.size()];
            }
            if (qm == 0) {
                int ti1 = (int)(rng() % nt), ti2 = (int)(rng() % nt);
                if (ti1 == ti2) ti2 = (ti1 + 1) % nt;
                qm = (1ULL << ti1) | (1ULL << ti2);
            }
            vector<int> Q;
            for (int ti = 0; ti < nt; ti++) {
                if (((qm >> ti) & 1) && (cur_dag.owners[cur_dag.tnode[ti]] & ~qm) == 0) Q.push_back(ti);
            }
            if (Q.size() != 2) continue;

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            int budget = rs + 1;
            if (budget < 2) continue;

            vector<char> excl = compute_excl(inR, {cur_dag.tnode[Q[0]], cur_dag.tnode[Q[1]]});
            build_window_dict(excl, inR, min<size_t>(P.s2_cap, 60000), min(P.n_leaves, 64), min(P.s2_leaves, 16));
            auto excl_vals = collect_excl_vals(excl, inR);
            DeepSearch ds(wdict, P, rng);
            ds.deep_beam = false;
            ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;

            CandState best_cand;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1) swap(Q[0], Q[1]);
                int rem0 = budget - 1;
                if (rem0 < 1) continue;
                auto sols0 = ds.search(cur_dag.targets[Q[0]], rem0, 3, 1, res.stats, 1);
                for (const auto &s0 : sols0) {
                    if ((int)s0.steps.size() + 1 > budget) continue;
                    vector<u64> new_v;
                    for (const auto &st : s0.steps) new_v.push_back(st.r);
                    auto mk = wdict.mark();
                    wdict.add_leaves_inc(new_v);
                    ds.extra_pool.clear();
                    for (u64 v : new_v) ds.extra_pool.insert(v);

                    u64 t1 = cur_dag.targets[Q[1]];
                    if (wdict.in_pool(t1) || ds.extra_pool.count(t1)) {
                        CandState cs = eval_cand_dag(inR, s0.steps, s0.engine);
                        if (cs.valid && cs.E < best_cand.E) best_cand = move(cs);
                    } else {
                        int rem1 = budget - (int)s0.steps.size();
                        if (rem1 >= 1) {
                            auto sols1 = ds.search(t1, rem1, 2, 1, res.stats, 1);
                            for (const auto &s1 : sols1) {
                                vector<Step> acc = s0.steps;
                                for (const auto &st : s1.steps) acc.push_back(st);
                                CandState cs = eval_cand_dag(inR, acc, s0.engine);
                                if (cs.valid && cs.E < best_cand.E) best_cand = move(cs);
                            }
                        }
                    }
                    ds.extra_pool.clear();
                    wdict.rollback(mk);
                }
            }
            if (best_cand.valid) commit_cand_sa(best_cand, 2, temp);

        } else if (roll < 95) {
            // Move 3: Fast local E-graph algebraic rewrite (< 0.1ms, ZERO dict search)
            if (n <= cur_dag.ks) continue;
            unordered_set<int> isT(cur_dag.tnode.begin(), cur_dag.tnode.end());
            int u = cur_dag.ks + (int)(rng() % (n - cur_dag.ks));
            const auto &nd = cur_dag.nodes[u];
            u64 Y = nd.v, va = cur_dag.nodes[nd.a].v, vb = cur_dag.nodes[nd.b].v;

            CandState best_cand;
            auto offer_rw = [&](const vector<char> &inR, const vector<char> &excl, const vector<Step> &steps) {
                for (const auto &st : steps) {
                    if (st.r != Y && sset.count(st.r)) return;
                    auto itv = cur_dag.val2idx.find(st.r);
                    if (itv != cur_dag.val2idx.end() && excl[itv->second] && !inR[itv->second] && st.r != Y) return;
                }
                CandState cs = eval_cand_dag(inR, steps, 4);
                if (cs.valid && cs.E < best_cand.E) best_cand = move(cs);
            };

            auto make_rw_masks = [&](int parent_drop) {
                vector<char> inR(n, 0);
                inR[u] = 1;
                if (parent_drop >= cur_dag.ks && cur_dag.users[parent_drop].size() == 1 && !isT.count(parent_drop)) {
                    inR[parent_drop] = 1;
                }
                vector<char> excl = compute_excl(inR, {u});
                return make_pair(inR, excl);
            };

            if (nd.op == '+' || nd.op == '-') {
                auto try_add3 = [&](int ch_idx, int s1, u64 t1, int s2, u64 t2, int s3, u64 t3) {
                    auto [inR, excl] = make_rw_masks(ch_idx);
                    int sg[3] = {s1, s2, s3};
                    u64 tm[3] = {t1, t2, t3};
                    int pairs[2][3] = {{0, 2, 1}, {1, 2, 0}};
                    for (int pr = 0; pr < 2; pr++) {
                        int i = pairs[pr][0], j = pairs[pr][1], remk = pairs[pr][2];
                        int si = sg[i], sj = sg[j], sk = sg[remk];
                        u64 xi = tm[i], xj = tm[j], xk = tm[remk];
                        i64 m_signed = (i64)si * (i64)xi + (i64)sj * (i64)xj;
                        if (m_signed == 0) continue;
                        u64 M = (u64)(m_signed > 0 ? m_signed : -m_signed);
                        if (M > L || M == Y || sset.count(M)) continue;
                        char mop = (si == sj) ? '+' : '-';
                        u64 ma = (si == sj) ? xi : (si > 0 ? (xi > xj ? xi : xj) : (xj > xi ? xj : xi));
                        u64 mb = (ma == xi) ? xj : xi;
                        char yop = 0; u64 ya = 0, yb = 0;
                        if (m_signed > 0) {
                            yop = (sk > 0) ? '+' : '-';
                            ya = M; yb = xk;
                        } else {
                            if (sk <= 0) continue;
                            yop = '-';
                            ya = xk; yb = M;
                        }
                        u64 c1 = 0, c2 = 0;
                        if (M == ma || M == mb || Y == ya || Y == yb) continue;
                        if (!apply_op(mop, ma, mb, L, c1) || c1 != M || !apply_op(yop, ya, yb, L, c2) || c2 != Y) continue;
                        auto itM = cur_dag.val2idx.find(M);
                        vector<Step> st;
                        if (itM == cur_dag.val2idx.end() || inR[itM->second]) {
                            st.push_back({mop, ma, mb, M});
                        } else if (excl[itM->second]) continue;
                        st.push_back({yop, ya, yb, Y});
                        offer_rw(inR, excl, st);
                    }
                };
                if (nd.a >= cur_dag.ks && (cur_dag.nodes[nd.a].op == '+' || cur_dag.nodes[nd.a].op == '-')) {
                    const auto &ch = cur_dag.nodes[nd.a];
                    try_add3(nd.a, 1, cur_dag.nodes[ch.a].v, ch.op == '+' ? 1 : -1, cur_dag.nodes[ch.b].v,
                             nd.op == '+' ? 1 : -1, vb);
                }
                if (nd.b >= cur_dag.ks && (cur_dag.nodes[nd.b].op == '+' || cur_dag.nodes[nd.b].op == '-')) {
                    const auto &ch = cur_dag.nodes[nd.b];
                    int sb = (nd.op == '+') ? 1 : -1;
                    try_add3(nd.b, 1, va, sb, cur_dag.nodes[ch.a].v, (ch.op == '+') ? sb : -sb, cur_dag.nodes[ch.b].v);
                }
            } else if (nd.op == '*') {
                auto try_mul3 = [&](int ch_idx, u64 P1, u64 P2, u64 other) {
                    auto [inR, excl] = make_rw_masks(ch_idx);
                    for (auto [x, y] : {make_pair(P1, P2), make_pair(P2, P1)}) {
                        if (x < 2 || other < 2 || (u128)x * other > (u128)L) continue;
                        u64 M = x * other;
                        if (M == Y || sset.count(M)) continue;
                        auto itM = cur_dag.val2idx.find(M);
                        vector<Step> st;
                        if (itM == cur_dag.val2idx.end() || inR[itM->second]) {
                            st.push_back({'*', x, other, M});
                        } else if (excl[itM->second]) continue;
                        st.push_back({'*', M, y, Y});
                        offer_rw(inR, excl, st);
                    }
                };
                if (nd.a >= cur_dag.ks && cur_dag.nodes[nd.a].op == '*') {
                    const auto &ch = cur_dag.nodes[nd.a];
                    try_mul3(nd.a, cur_dag.nodes[ch.a].v, cur_dag.nodes[ch.b].v, vb);
                }
                if (nd.b >= cur_dag.ks && cur_dag.nodes[nd.b].op == '*') {
                    const auto &ch = cur_dag.nodes[nd.b];
                    try_mul3(nd.b, cur_dag.nodes[ch.a].v, cur_dag.nodes[ch.b].v, va);
                }
                // Factor shifts (A/g)*(B*g) using g in cur_dag
                auto [inR, excl] = make_rw_masks(-1);
                for (u64 g : {2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 8ULL, 16ULL}) {
                    auto itg = cur_dag.val2idx.find(g);
                    if (itg == cur_dag.val2idx.end() || excl[itg->second]) continue;
                    for (auto [A, B] : {make_pair(va, vb), make_pair(vb, va)}) {
                        if (A % g != 0 || (u128)B * g > (u128)L) continue;
                        u64 A2 = A / g, B2 = B * g;
                        if (A2 < 2 || B2 < 2 || A2 == Y || B2 == Y || sset.count(A2) || sset.count(B2)) continue;
                        auto itA = cur_dag.val2idx.find(A2), itB = cur_dag.val2idx.find(B2);
                        bool hasA = (itA != cur_dag.val2idx.end() && !excl[itA->second]);
                        bool hasB = (itB != cur_dag.val2idx.end() && !excl[itB->second]);
                        if ((itA != cur_dag.val2idx.end() && excl[itA->second]) ||
                            (itB != cur_dag.val2idx.end() && excl[itB->second])) continue;
                        vector<Step> st;
                        if (!hasA) st.push_back({'/', A, g, A2});
                        if (!hasB) st.push_back(g == 2 ? Step{'+', B, B, B2} : Step{'*', B, g, B2});
                        st.push_back({'*', A2, B2, Y});
                        offer_rw(inR, excl, st);
                    }
                }
            }
            if (best_cand.valid) commit_cand_sa(best_cand, 3, temp);

        } else {
            // Move 4: Fast k=3..4 target subset ruin-and-recreate using compact 60k wdict
            if (nt < 3) continue;
            u64 qm = 0;
            if (rng() % 2 == 0) {
                vector<u64> hubs;
                for (int u = cur_dag.ks; u < n; u++) {
                    u64 m = cur_dag.owners[u];
                    int pc = __builtin_popcountll(m);
                    if (pc >= 3 && pc <= 4) hubs.push_back(m);
                }
                if (!hubs.empty()) qm = hubs[rng() % hubs.size()];
            }
            if (qm == 0) {
                int k = 3 + (int)(rng() % min(2, max(1, nt - 2)));
                vector<pair<int, int>> ord_t;
                for (int ti = 0; ti < nt; ti++) ord_t.emplace_back(cur_dag.tnode[ti], ti);
                sort(ord_t.begin(), ord_t.end());
                int st = (int)(rng() % max(1, nt - k + 1));
                for (int j = 0; j < k && st + j < nt; j++) qm |= (1ULL << ord_t[st + j].second);
            }
            bool chg = true;
            while (chg) {
                chg = false;
                for (int ti = 0; ti < nt; ti++) {
                    if (((qm >> ti) & 1) && (cur_dag.owners[cur_dag.tnode[ti]] & ~qm) != 0) {
                        qm &= ~(1ULL << ti);
                        chg = true;
                    }
                }
            }
            vector<int> Q;
            vector<u64> Q_vals;
            for (int ti = 0; ti < nt; ti++) {
                if ((qm >> ti) & 1) {
                    Q.push_back(ti);
                    Q_vals.push_back(cur_dag.targets[ti]);
                }
            }
            if (Q.size() < 3) continue;

            vector<char> inR(n, 0);
            int rs = 0;
            for (int u = cur_dag.ks; u < n; u++) {
                if (cur_dag.owners[u] && (cur_dag.owners[u] & ~qm) == 0) { inR[u] = 1; rs++; }
            }
            int budget = rs + 2;
            if (rs < 4 || budget < (int)Q.size()) continue;

            vector<int> Q_nodes;
            for (int ti : Q) Q_nodes.push_back(cur_dag.tnode[ti]);
            vector<char> excl = compute_excl(inR, Q_nodes);
            unordered_set<u64> w_have;
            for (int u = 0; u < n; u++) if (!excl[u]) w_have.insert(cur_dag.nodes[u].v);
            build_window_dict(excl, inR, min<size_t>(P.s2_cap, 60000), min(P.n_leaves, 64), min(P.s2_leaves, 16));
            auto sub = reconstruct_subset(L, Q_vals, wdict, P, w_have, rng, res.stats, budget + 1, 0);
            if (!sub.empty()) {
                CandState cs = eval_cand_dag(inR, sub, 5);
                if (cs.valid) commit_cand_sa(cs, 4, temp);
            }
        }
    }
    return res;
}

// ---------------------------------------------------------------- Mode 2: Scale-Aware Constructive Planner
static vector<Instr> construct_puzzle(u64 L, const vector<u64> &starts, const vector<u64> &targets,
                                      const Params &P, const string &pid, mt19937_64 &rng,
                                      SearchStats &stats, double jitter = 0.0) {
    unordered_set<u64> have(starts.begin(), starts.end());
    vector<u64> built(starts.begin(), starts.end());
    map<u64, Step> defs;
    vector<u64> order;
    unordered_set<u64> tset(targets.begin(), targets.end());
    unordered_map<u64, int> fanout;
    unordered_map<u64, int> def_pos;
    unordered_set<u64> near_vals;

    vector<u64> remaining;
    for (u64 t : targets) if (!have.count(t)) remaining.push_back(t);

    // Fast hidden-base discovery (< 3 ms): exact pairwise gcd + small offset gcd over smallest targets
    vector<u64> bonus_hubs;
    {
        vector<u64> st = remaining;
        sort(st.begin(), st.end());
        int n24 = min(24, (int)st.size());
        int n10 = min(10, (int)st.size());
        vector<u64> cand_g;
        for (int i = 0; i < n24; i++) {
            for (int j = i + 1; j < n24; j++) {
                u64 g = bgcd(st[i], st[j]);
                if (g >= 64 && g <= L / 100) cand_g.push_back(g);
            }
        }
        for (int i = 0; i < n10; i++) {
            for (int j = i + 1; j < n10; j++) {
                for (int da = -8; da <= 8; da++) {
                    if ((i64)st[i] + da <= 0) continue;
                    u64 va = (u64)((i64)st[i] + da);
                    for (int db = -8; db <= 8; db++) {
                        if ((i64)st[j] + db <= 0) continue;
                        u64 g = bgcd(va, (u64)((i64)st[j] + db));
                        if (g >= 128 && g <= L / 100) cand_g.push_back(g);
                    }
                }
            }
        }
        sort(cand_g.begin(), cand_g.end());
        cand_g.erase(unique(cand_g.begin(), cand_g.end()), cand_g.end());
        vector<pair<double, u64>> ranked_g;
        for (u64 g : cand_g) {
            if (have.count(g)) continue;
            int hits = 0;
            for (u64 t : targets) {
                u64 r = t % g;
                if (min(r, g - r) <= 64) hits++;
            }
            if (hits >= 4) {
                ranked_g.emplace_back((double)hits * log2((double)g), g);
            }
        }
        sort(ranked_g.rbegin(), ranked_g.rend());
        for (size_t i = 0; i < ranked_g.size() && i < 3; i++) {
            bonus_hubs.push_back(ranked_g[i].second);
            near_vals.insert(ranked_g[i].second);
        }
    }

    PoolDict dict;
    DeepSearch ds(dict, P, rng);

    auto apply_steps = [&](const vector<Step> &steps) {
        for (const auto &s : steps) {
            if (!have.count(s.r)) {
                u64 chk = 0;
                if (!have.count(s.a) || !have.count(s.b) || !apply_op(s.op, s.a, s.b, L, chk) || chk != s.r) continue;
                have.insert(s.r);
                built.push_back(s.r);
                defs[s.r] = s;
                order.push_back(s.r);
                def_pos[s.r] = (int)order.size() - 1;
                fanout[s.a]++;
                if (s.b != s.a) fanout[s.b]++;
            }
        }
    };

    // Binary fallback (strictly last resort!)
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

    bool boot_ladder_used = false;
    int hub_commits = 0, consec_hub = 0;
    while (!remaining.empty()) {
        vector<u64> leaves, done_t;
        vector<int> lcosts;
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, built, tset, fanout, near_vals, P.n_leaves, leaves, lcosts, s1_cnt, done_t, true);
        dict.build(L, starts, built, leaves, lcosts, s1_cnt, done_t, P.s2_leaves, P.s2_cap);
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
            for (size_t i = 0; i < remaining.size() && (i < 16 || cands.empty()); i++) {
                auto sols = ds.search(remaining[i], 28, P.la_alts, 2, stats, 2);
                for (auto &s : sols) cands.push_back({remaining[i], s.cost, s.engine, move(s.steps)});
            }
        }

        if (cands.empty()) {
            u64 t_min = *min_element(remaining.begin(), remaining.end());
            auto boot = ds.search(t_min, 36, 1, 0, stats, 3);  // deep depth=12 bootstrap beam
            if (!boot.empty()) {
                apply_steps(boot[0].steps);
            } else if (!boot_ladder_used && !starts.empty()) {
                boot_ladder_used = true;
                // Compact 7-op power/product bootstrap basis before falling back to binary!
                size_t before = order.size();
                u64 s0 = starts[0];
                if (!have.count(1)) apply_steps({{'/', s0, s0, 1}});
                vector<u64> pws{1, 2, 4, 16, 256, 65536};
                if (!have.count(2) && 2 <= L) apply_steps({{'+', 1, 1, 2}});
                for (size_t pi = 2; pi < pws.size(); pi++) {
                    u64 prev = pws[pi - 1];
                    if (have.count(prev) && (u128)prev * prev <= (u128)L) {
                        apply_steps({{'*', prev, prev, prev * prev}});
                    }
                }
                if (starts.size() >= 2 && (u128)starts[0] * starts[1] <= (u128)L) {
                    apply_steps({{'*', starts[0], starts[1], starts[0] * starts[1]}});
                }
                if (order.size() == before) build_binary(t_min);
            } else {
                build_binary(t_min);
            }
            consec_hub = 0;
        } else {
            // Multi-target hub injection (distinct-target intermediates & bonus hidden bases)
            vector<PlanCand> hub_cands;
            if (hub_commits < 8 && consec_hub < 4 && remaining.size() >= 2) {
                unordered_map<u64, unordered_set<u64>> mid_targets;
                for (const auto &c : cands) {
                    if (c.target == 0) continue;
                    for (const auto &st : c.steps) {
                        if (st.r != c.target && !have.count(st.r)) mid_targets[st.r].insert(c.target);
                    }
                }
                vector<pair<int, u64>> mranked;
                for (const auto &[hv, ts_set] : mid_targets) {
                    if (ts_set.size() >= 2) mranked.emplace_back((int)ts_set.size(), hv);
                }
                sort(mranked.rbegin(), mranked.rend());
                unordered_set<u64> picked_hubs;
                for (size_t hi = 0; hi < mranked.size() && hub_cands.size() < 3; hi++) {
                    u64 hv = mranked[hi].second;
                    if (!picked_hubs.insert(hv).second) continue;
                    vector<Step> best_pref;
                    for (const auto &c : cands) {
                        for (size_t j = 0; j < c.steps.size(); j++) {
                            if (c.steps[j].r == hv) {
                                vector<Step> pref;
                                for (size_t p = 0; p <= j; p++) pref.push_back(c.steps[p]);
                                // Prune unused branches in pref keeping only dependencies of hv
                                unordered_set<u64> need{hv};
                                vector<Step> clean;
                                for (int q = (int)pref.size() - 1; q >= 0; --q) {
                                    if (need.count(pref[q].r)) {
                                        clean.push_back(pref[q]);
                                        need.insert(pref[q].a);
                                        need.insert(pref[q].b);
                                    }
                                }
                                reverse(clean.begin(), clean.end());
                                if (!clean.empty() && clean.size() <= 4 &&
                                    (best_pref.empty() || clean.size() < best_pref.size())) {
                                    best_pref = move(clean);
                                }
                            }
                        }
                    }
                    if (!best_pref.empty()) hub_cands.push_back({0, (int)best_pref.size(), 5, move(best_pref)});
                }
                for (u64 bh : bonus_hubs) {
                    if (have.count(bh) || picked_hubs.count(bh)) continue;
                    auto sols = ds.search(bh, 14, 1, 0, stats, 2);
                    if (!sols.empty()) {
                        picked_hubs.insert(bh);
                        hub_cands.push_back({0, sols[0].cost, sols[0].engine, move(sols[0].steps)});
                    }
                }
            }

            stable_sort(cands.begin(), cands.end(), [](const PlanCand &a, const PlanCand &b) {
                return a.cost < b.cost;
            });
            if ((int)cands.size() > P.lookahead) cands.resize(P.lookahead);
            for (auto &hc : hub_cands) cands.push_back(move(hc));

            int K = (int)cands.size();
            int bi = 0;
            double bs = 1e18;
            if (K > 1 && remaining.size() > 1) {
                unordered_set<u64> sset_c(starts.begin(), starts.end());
                int cur_ord_n = (int)order.size();
                for (int k = 0; k < K; k++) {
                    vector<u64> add_v;
                    unordered_set<u64> made;
                    for (const auto &st : cands[k].steps) {
                        add_v.push_back(st.r);
                        made.insert(st.r);
                    }
                    int old_reuses = 0;
                    for (const auto &st : cands[k].steps) {
                        for (u64 opv : {st.a, st.b}) {
                            if (sset_c.count(opv) || made.count(opv)) continue;
                            auto it_dp = def_pos.find(opv);
                            if (it_dp != def_pos.end() && cur_ord_n - it_dp->second > 42 && fanout[opv] <= 2) {
                                old_reuses++;
                            }
                        }
                    }
                    auto mk = dict.mark();
                    dict.add_leaves_inc(add_v);
                    ds.extra_pool = made;
                    double rem_est = 0.0;
                    for (u64 x : remaining) {
                        if (have.count(x) || made.count(x)) continue;
                        rem_est += ds.estimate_cost(x);
                    }
                    ds.extra_pool.clear();
                    dict.rollback(mk);
                    double sc = (double)cands[k].cost + P.la_weight * rem_est + 0.25 * (double)old_reuses;
                    if (jitter > 0.0) sc += jitter * ((double)(rng() % 1000) / 1000.0);
                    if (sc < bs) { bs = sc; bi = k; }
                }
            }
            bool is_hub = (cands[bi].target == 0);
            if (is_hub) { hub_commits++; consec_hub++; }
            else consec_hub = 0;
            apply_steps(cands[bi].steps);
            if (P.verbose > 1) {
                fprintf(stderr, "  [%s] built target=%llu cost=%d eng=%d total=%zu rem=%zu\n",
                        pid.c_str(), (unsigned long long)cands[bi].target, cands[bi].cost,
                        cands[bi].eng, order.size(), remaining.size() - (is_hub ? 0 : 1));
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
    if (!dag.compile(prog, max(P.reorder_tries, 64), rng, &order, nullptr, 20000))
        throw runtime_error("construct compile fail");
    return prog;
}

static vector<u64> discover_cf_and_gcd_bases(u64 L, const vector<u64> &targets_sub,
                                             const unordered_set<u64> &have,
                                             const unordered_set<u64> &banned_vals);

static vector<Instr> construct_beam_puzzle(
    u64 L, const vector<u64> &starts, const vector<u64> &targets,
    const Params &P, const string &pid, mt19937_64 &rng, SearchStats &stats,
    vector<vector<Instr>> *all_beam_progs = nullptr,
    vector<Step> *out_recipes = nullptr) {

    int nt = (int)targets.size();
    unordered_set<u64> tset(targets.begin(), targets.end());
    unordered_set<u64> sset_global(starts.begin(), starts.end());
    unordered_set<u64> empty_ban;
    vector<u64> bonus_hubs = discover_cf_and_gcd_bases(L, targets, sset_global, empty_ban);
    unordered_set<u64> base_near(bonus_hubs.begin(), bonus_hubs.end());

    struct BeamState {
        vector<u64> built;
        vector<u64> order;
        unordered_set<u64> have;
        map<u64, Step> defs;
        unordered_map<u64, int> fanout;
        unordered_map<u64, int> def_pos;
        u64 solved_mask = 0;
        int unsolved_cnt = 0;
        double score = 0.0;
        u64 sig = 0;
        bool boot_ladder_used = false;
    };

    auto refresh_solved = [&](BeamState &S) {
        u64 m = 0;
        int un = 0;
        for (int ti = 0; ti < nt; ti++) {
            if (S.have.count(targets[ti])) {
                if (ti < 64) m |= (1ULL << ti);
            } else {
                un++;
            }
        }
        S.solved_mask = m;
        S.unsolved_cnt = un;
    };

    auto recompute_sig = [&](BeamState &S) {
        u64 h = mix64(S.solved_mask ^ ((u64)S.order.size() * 0x9e3779b97f4a7c15ULL));
        for (u64 v : S.order) h ^= mix64(v);
        S.sig = h;
    };

    auto apply_steps_state = [&](BeamState &S, const vector<Step> &steps) {
        for (const auto &s : steps) {
            if (!S.have.count(s.r)) {
                u64 chk = 0;
                if (!S.have.count(s.a) || !S.have.count(s.b) || !apply_op(s.op, s.a, s.b, L, chk) || chk != s.r) continue;
                S.have.insert(s.r);
                S.built.push_back(s.r);
                S.defs[s.r] = s;
                S.order.push_back(s.r);
                S.def_pos[s.r] = (int)S.order.size() - 1;
                S.fanout[s.a]++;
                if (s.b != s.a) S.fanout[s.b]++;
            }
        }
        refresh_solved(S);
        recompute_sig(S);
    };

    function<void(BeamState &, u64)> build_binary_state = [&](BeamState &S, u64 v) {
        if (S.have.count(v)) return;
        if (v == 1 && !starts.empty()) {
            u64 s0 = starts[0];
            apply_steps_state(S, {{'/', s0, s0, 1}});
            return;
        }
        build_binary_state(S, v / 2);
        apply_steps_state(S, {{'+', v / 2, v / 2, (v / 2) * 2}});
        if (v % 2) {
            build_binary_state(S, 1);
            apply_steps_state(S, {{'+', (v / 2) * 2, 1, v}});
        }
    };

    auto apply_boot_ladder = [&](BeamState &S) -> bool {
        if (S.boot_ladder_used || starts.empty()) return false;
        S.boot_ladder_used = true;
        size_t before = S.order.size();
        u64 s0 = starts[0];
        if (!S.have.count(1)) apply_steps_state(S, {{'/', s0, s0, 1}});
        vector<u64> pws{1, 2, 4, 16, 256, 65536};
        if (!S.have.count(2) && 2 <= L) apply_steps_state(S, {{'+', 1, 1, 2}});
        for (size_t pi = 2; pi < pws.size(); pi++) {
            u64 prev = pws[pi - 1];
            if (S.have.count(prev) && (u128)prev * prev <= (u128)L) {
                apply_steps_state(S, {{'*', prev, prev, prev * prev}});
            }
        }
        if (starts.size() >= 2 && (u128)starts[0] * starts[1] <= (u128)L) {
            apply_steps_state(S, {{'*', starts[0], starts[1], starts[0] * starts[1]}});
        }
        return S.order.size() > before;
    };

    BeamState S0;
    S0.built.assign(starts.begin(), starts.end());
    S0.have.insert(starts.begin(), starts.end());
    refresh_solved(S0);
    recompute_sig(S0);

    vector<BeamState> beam, finished;
    if (S0.unsolved_cnt == 0) {
        finished.push_back(S0);
    } else {
        beam.push_back(S0);
        // Optional 1..4-op start-diff / power seed for low-start puzzles when W >= 2
        if (P.beam_width >= 2 && starts.size() <= 2 && !starts.empty() && *max_element(starts.begin(), starts.end()) > 32) {
            BeamState S1 = S0;
            u64 s0 = starts[0];
            if (!S1.have.count(1)) apply_steps_state(S1, {{'/', s0, s0, 1}});
            if (!S1.have.count(2) && 2 <= L) apply_steps_state(S1, {{'+', 1, 1, 2}});
            if (!S1.have.count(4) && 4 <= L) apply_steps_state(S1, {{'+', 2, 2, 4}});
            if (starts.size() == 2 && starts[0] != starts[1]) {
                u64 df = starts[0] > starts[1] ? starts[0] - starts[1] : starts[1] - starts[0];
                if (df > 0 && df <= L && !S1.have.count(df)) {
                    apply_steps_state(S1, {{'-', max(starts[0], starts[1]), min(starts[0], starts[1]), df}});
                }
            }
            if (S1.sig != S0.sig) {
                S1.score = (double)S1.order.size();
                if (S1.unsolved_cnt == 0) finished.push_back(move(S1));
                else beam.push_back(move(S1));
            }
        }
    }

    const int b_n1 = min(P.n_leaves, 112);
    const int b_k2 = min(P.s2_leaves, 32);
    const size_t b_cap = min<size_t>(P.s2_cap, 160000);

    PoolDict dict;
    DeepSearch ds(dict, P, rng);

    int guard = 0;
    while (!beam.empty() && guard++ < nt + 4) {
        vector<BeamState> next_states;
        int min_rem = nt;

        struct PlanCand { u64 target; int cost; int eng; vector<Step> steps; };
        struct LightCand { double score; size_t nnew; u64 target; vector<Step> steps; };

        auto expand_state = [&](BeamState &S) {
            vector<u64> rem;
            for (u64 t : targets) if (!S.have.count(t)) rem.push_back(t);
            if (rem.empty()) { finished.push_back(S); return; }
            min_rem = min(min_rem, (int)rem.size());

            vector<u64> rem_sorted = rem;
            sort(rem_sorted.begin(), rem_sorted.end());
            u64 t_min = rem_sorted[0];

            auto build_s_dict = [&]() {
                vector<u64> leaves, done_t;
                vector<int> lcosts;
                int s1_cnt = 0;
                double tb = now_sec();
                select_leaves(L, starts, S.built, tset, S.fanout, base_near, b_n1, leaves, lcosts, s1_cnt, done_t, true);
                dict.build(L, starts, S.built, leaves, lcosts, s1_cnt, done_t, b_k2, b_cap);
                stats.sec[0] += now_sec() - tb;
            };
            build_s_dict();

            vector<PlanCand> cands;
            int min_fast = 1 << 20;
            for (u64 t : rem) {
                auto sols = ds.search(t, 10, 3, 1, stats, 1);
                for (auto &sol : sols) {
                    min_fast = min(min_fast, sol.cost);
                    cands.push_back({t, sol.cost, sol.engine, move(sol.steps)});
                }
            }
            if (min_fast > 5) {
                for (size_t i = 0; i < rem_sorted.size() && (i < 10 || cands.empty()); i++) {
                    auto sols = ds.search(rem_sorted[i], 28, 3, 1, stats, 2);
                    for (auto &sol : sols) cands.push_back({rem_sorted[i], sol.cost, sol.engine, move(sol.steps)});
                }
            }

            if (cands.empty()) {
                auto boot = ds.search(t_min, 36, 1, 0, stats, 3);
                if (!boot.empty()) {
                    BeamState ch = S;
                    apply_steps_state(ch, boot[0].steps);
                    ch.score = (double)ch.order.size();
                    if (ch.unsolved_cnt == 0) finished.push_back(move(ch));
                    else next_states.push_back(move(ch));
                    return;
                }
                BeamState Sb = S;
                if (apply_boot_ladder(Sb)) {
                    vector<u64> rem_b;
                    for (u64 t : targets) if (!Sb.have.count(t)) rem_b.push_back(t);
                    if (rem_b.empty()) {
                        Sb.score = (double)Sb.order.size();
                        finished.push_back(move(Sb));
                        return;
                    }
                    vector<u64> leaves, done_t;
                    vector<int> lcosts;
                    int s1_cnt = 0;
                    select_leaves(L, starts, Sb.built, tset, Sb.fanout, base_near, b_n1, leaves, lcosts, s1_cnt, done_t, true);
                    dict.build(L, starts, Sb.built, leaves, lcosts, s1_cnt, done_t, b_k2, b_cap);
                    vector<u64> rbs = rem_b;
                    sort(rbs.begin(), rbs.end());
                    for (size_t i = 0; i < rbs.size() && (i < 10 || cands.empty()); i++) {
                        auto sols = ds.search(rbs[i], 28, 3, 1, stats, 2);
                        for (auto &sol : sols) cands.push_back({rbs[i], sol.cost, sol.engine, move(sol.steps)});
                    }
                    if (!cands.empty()) {
                        S = move(Sb);
                        rem.swap(rem_b);
                        rem_sorted.swap(rbs);
                        t_min = rem_sorted[0];
                    } else {
                        build_binary_state(Sb, rbs[0]);
                        Sb.score = (double)Sb.order.size();
                        if (Sb.unsolved_cnt == 0) finished.push_back(move(Sb));
                        else next_states.push_back(move(Sb));
                        return;
                    }
                } else {
                    build_binary_state(Sb, t_min);
                    Sb.score = (double)Sb.order.size();
                    if (Sb.unsolved_cnt == 0) finished.push_back(move(Sb));
                    else next_states.push_back(move(Sb));
                    return;
                }
            }

            // Group candidates by target, rank targets by minimum cost, take <= 6 distinct targets (top 5 + smallest)
            unordered_map<u64, vector<PlanCand>> by_t;
            for (auto &c : cands) by_t[c.target].push_back(move(c));
            vector<pair<int, u64>> t_rank;
            t_rank.reserve(by_t.size());
            for (auto &[tv, vec] : by_t) {
                sort(vec.begin(), vec.end(), [](const PlanCand &a, const PlanCand &b) {
                    if (a.cost != b.cost) return a.cost < b.cost;
                    return a.steps.size() < b.steps.size();
                });
                t_rank.emplace_back(vec[0].cost, tv);
            }
            sort(t_rank.begin(), t_rank.end());

            vector<u64> picked_t;
            unordered_set<u64> picked_set;
            for (size_t i = 0; i < t_rank.size() && picked_t.size() < 5; i++) {
                picked_t.push_back(t_rank[i].second);
                picked_set.insert(t_rank[i].second);
            }
            if (by_t.count(t_min) && !picked_set.count(t_min)) {
                picked_t.push_back(t_min);
                picked_set.insert(t_min);
            } else if (picked_t.size() < 6 && t_rank.size() > 5) {
                picked_t.push_back(t_rank[5].second);
                picked_set.insert(t_rank[5].second);
            }

            vector<LightCand> evals;
            int cur_ord_n = (int)S.order.size();
            for (u64 pt : picked_t) {
                auto &vec = by_t[pt];
                size_t rlim = min<size_t>(vec.size(), 3);
                for (size_t ri = 0; ri < rlim; ri++) {
                    vector<u64> add_v;
                    unordered_set<u64> made;
                    for (const auto &st : vec[ri].steps) {
                        if (!S.have.count(st.r) && made.insert(st.r).second) add_v.push_back(st.r);
                    }
                    int old_reuses = 0;
                    for (const auto &st : vec[ri].steps) {
                        for (u64 opv : {st.a, st.b}) {
                            if (sset_global.count(opv) || made.count(opv)) continue;
                            auto it_dp = S.def_pos.find(opv);
                            if (it_dp != S.def_pos.end() && cur_ord_n - it_dp->second > 42) {
                                auto it_f = S.fanout.find(opv);
                                if (it_f != S.fanout.end() && it_f->second <= 2) old_reuses++;
                            }
                        }
                    }

                    auto mk = dict.mark();
                    dict.add_leaves_inc(add_v);
                    ds.extra_pool = made;
                    double rem_est = 0.0;
                    int div_hits = 0;
                    for (u64 x : rem) {
                        if (S.have.count(x) || made.count(x)) continue;
                        rem_est += ds.estimate_cost(x);
                        if (x >= 16) {
                            for (u64 nv : add_v) {
                                if (nv < 16) continue;
                                u64 rmod = x % nv;
                                u64 dclose = rmod == 0 ? 0 : min(rmod, nv - rmod);
                                if (dclose <= 64) { div_hits++; break; }
                            }
                        }
                    }
                    ds.extra_pool.clear();
                    dict.rollback(mk);

                    double div_bonus = min(3.0, 0.6 * (double)div_hits);
                    double sc = (double)(cur_ord_n + (int)add_v.size()) + P.la_weight * rem_est
                                + 0.25 * (double)old_reuses - div_bonus;
                    evals.push_back({sc, add_v.size(), pt, vec[ri].steps});
                }
            }
            sort(evals.begin(), evals.end(), [](const LightCand &a, const LightCand &b) {
                if (a.score != b.score) return a.score < b.score;
                if (a.nnew != b.nnew) return a.nnew < b.nnew;
                return a.target < b.target;
            });
            size_t take_c = min<size_t>(evals.size(), 4);
            for (size_t ci = 0; ci < take_c; ci++) {
                BeamState ch = S;
                apply_steps_state(ch, evals[ci].steps);
                ch.score = evals[ci].score;
                if (ch.unsolved_cnt == 0) finished.push_back(move(ch));
                else next_states.push_back(move(ch));
            }
        };

        for (size_t si = 0; si < beam.size(); si++) {
            expand_state(beam[si]);
        }

        beam.clear();
        if (!next_states.empty()) {
            sort(next_states.begin(), next_states.end(), [](const BeamState &a, const BeamState &b) {
                if (a.score != b.score) return a.score < b.score;
                return a.order.size() < b.order.size();
            });
            unordered_set<u64> seen_sigs;
            unordered_map<u64, int> mask_cnt;
            vector<size_t> skipped_idxs;
            int max_b = max(1, P.beam_width);
            for (size_t i = 0; i < next_states.size() && (int)beam.size() < max_b; i++) {
                if (!seen_sigs.insert(next_states[i].sig).second) continue;
                int &mc = mask_cnt[next_states[i].solved_mask];
                if (mc >= 2) { skipped_idxs.push_back(i); continue; }
                mc++;
                beam.push_back(move(next_states[i]));
            }
            if ((int)beam.size() < max_b && min_rem <= 2) {
                for (size_t i : skipped_idxs) {
                    if ((int)beam.size() >= max_b) break;
                    beam.push_back(move(next_states[i]));
                }
            }
        }
    }

    if (finished.empty()) finished = move(beam);
    sort(finished.begin(), finished.end(), [](const BeamState &a, const BeamState &b) {
        if (a.order.size() != b.order.size()) return a.order.size() < b.order.size();
        return a.score < b.score;
    });

    unordered_set<u64> seen_fin;
    vector<vector<Instr>> valid_progs;
    unordered_set<string> seen_jsons;
    unordered_set<u64> seen_rec_keys;
    auto push_dag_recipes = [&](const Dag &dg) {
        if (!out_recipes || out_recipes->size() >= 2500) return;
        for (int u = dg.ks; u < (int)dg.nodes.size(); u++) {
            const auto &nd = dg.nodes[u];
            u64 av = dg.nodes[nd.a].v, bv = dg.nodes[nd.b].v;
            if ((nd.op == '+' || nd.op == '*') && av > bv) swap(av, bv);
            u64 k = mix64(nd.v ^ ((u64)(unsigned char)nd.op << 56)) ^ mix64(av * 0x9e3779b97f4a7c15ULL + bv);
            if (seen_rec_keys.insert(k).second) {
                out_recipes->push_back({nd.op, av, bv, nd.v});
                if (out_recipes->size() >= 2500) break;
            }
        }
    };

    for (const auto &S : finished) {
        if (!seen_fin.insert(S.sig).second) continue;
        Dag dag;
        dag.init(L, starts, targets);
        if (!dag.from_defs(S.defs, S.order)) continue;
        push_dag_recipes(dag);
        vector<Instr> prog;
        if (!dag.compile(prog, max(P.reorder_tries, 64), rng, &S.order, nullptr, 20000)) continue;
        if (!verify_prog(L, starts, targets, prog).empty()) continue;
        string js = to_json(prog);
        if (seen_jsons.insert(js).second) valid_progs.push_back(move(prog));
        if ((int)valid_progs.size() >= max(2, P.beam_width)) break;
    }

    sort(valid_progs.begin(), valid_progs.end(), [](const vector<Instr> &a, const vector<Instr> &b) {
        return a.size() < b.size();
    });

    if (P.lns_sec > 0 && !valid_progs.empty()) {
        int npol = min<int>((int)valid_progs.size(), 2);
        Params sp = P;
        sp.lns_sec = P.lns_sec * 0.5;
        vector<vector<Instr>> sa_progs;
        for (int i = 0; i < npol; i++) {
            mt19937_64 srng((u64)(P.seed + (i + 1) * 7919) * 1000003ULL + hash<string>()(pid));
            auto lr = sa_optimize(L, starts, targets, valid_progs[i], sp, pid, srng);
            if (verify_prog(L, starts, targets, lr.prog).empty()) {
                string js = to_json(lr.prog);
                if (seen_jsons.insert(js).second) sa_progs.push_back(move(lr.prog));
            }
            if (out_recipes && out_recipes->size() < 2500) {
                for (const auto &hr : lr.harvested_recipes) {
                    u64 av = hr.a, bv = hr.b;
                    if ((hr.op == '+' || hr.op == '*') && av > bv) swap(av, bv);
                    u64 k = mix64(hr.r ^ ((u64)(unsigned char)hr.op << 56)) ^ mix64(av * 0x9e3779b97f4a7c15ULL + bv);
                    if (seen_rec_keys.insert(k).second) {
                        out_recipes->push_back({hr.op, av, bv, hr.r});
                        if (out_recipes->size() >= 2500) break;
                    }
                }
            }
        }
        for (auto &p : sa_progs) valid_progs.push_back(move(p));
        sort(valid_progs.begin(), valid_progs.end(), [](const vector<Instr> &a, const vector<Instr> &b) {
            return a.size() < b.size();
        });
    }

    if (all_beam_progs) *all_beam_progs = valid_progs;
    if (valid_progs.empty()) return {};
    return valid_progs[0];
}

// ---------------------------------------------------------------- Driver & Auxiliary Modes
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

static unordered_map<string, vector<Instr>> load_progs_file(const string &path) {
    unordered_map<string, vector<Instr>> res;
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        size_t sp = line.find(' ');
        if (sp == string::npos) continue;
        string pid = line.substr(0, sp);
        size_t br = line.find('[', sp);
        if (br == string::npos) continue;
        res[pid] = parse_prog(line.substr(br));
    }
    return res;
}

struct RecipeRec {
    u64 r; char op; u64 a, b;
    bool operator<(const RecipeRec &o) const {
        return tie(r, op, a, b) < tie(o.r, o.op, o.a, o.b);
    }
};

static int run_sched_test(const vector<PuzzleIn> &puzzles, const Params &P) {
    auto progs = load_progs_file(P.sched_test_path);
    atomic<int> nextp{0};
    mutex mu;
    int scr_exact = 0, scr_total = 0;
    int pert_exact = 0, pert_total = 0;
    double sum_scr_ms = 0.0, sum_pert_ms = 0.0;

    auto worker = [&]() {
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            const auto &pz = puzzles[k];
            auto it = progs.find(pz.id);
            if (it == progs.end()) continue;
            mt19937_64 rng((u64)P.seed * 1000003ULL + hash<string>()(pz.id));

            Dag d0;
            d0.init(pz.L, pz.s, pz.t);
            if (!d0.from_prog(it->second)) continue;
            d0.analyse();
            int dag_len = (int)d0.nodes.size() - d0.ks;

            // Scrambled-from-scratch DAG test (no pref_order)
            map<u64, Step> defs;
            for (int u = d0.ks; u < (int)d0.nodes.size(); u++) {
                const auto &nd = d0.nodes[u];
                defs[nd.v] = {nd.op, d0.nodes[nd.a].v, d0.nodes[nd.b].v, nd.v};
            }
            Dag dscr;
            dscr.init(pz.L, pz.s, pz.t);
            dscr.from_defs(defs, {});
            vector<Instr> p_scr;
            double ts0 = now_sec();
            bool ok_s = dscr.compile(p_scr, P.reorder_tries, rng, nullptr, nullptr, 4000);
            double ms_s = (now_sec() - ts0) * 1000.0;
            bool ex_s = ok_s && (int)p_scr.size() == dag_len;

            // Perturbation test: remove 1 target cone from PUB pref_ord and re-insert via compile(&kept_ord)
            int pe = 0, pt = 0;
            double p_ms = 0.0;
            vector<int> cands;
            for (size_t ti = 0; ti < d0.targets.size(); ti++) cands.push_back((int)ti);
            shuffle(cands.begin(), cands.end(), rng);
            vector<u64> full_ord = extract_prog_order(pz.s, it->second);
            for (int ti : cands) {
                if (pt >= 10) break;
                u64 bit = 1ULL << ti;
                vector<u64> kept_ord;
                for (int u = d0.ks; u < (int)d0.nodes.size(); u++) {
                    if (d0.owners[u] && (d0.owners[u] & ~bit) == 0) continue;
                    kept_ord.push_back(d0.nodes[u].v);
                }
                // keep relative order of full_ord
                unordered_set<u64> kset(kept_ord.begin(), kept_ord.end());
                vector<u64> kept_in_ord;
                for (u64 v : full_ord) if (kset.count(v)) kept_in_ord.push_back(v);

                vector<Instr> pp;
                double tp0 = now_sec();
                bool okp = d0.compile(pp, P.reorder_tries, rng, &kept_in_ord, nullptr, 3500);
                p_ms += (now_sec() - tp0) * 1000.0;
                pt++;
                if (okp && (int)pp.size() == dag_len) pe++;
            }

            lock_guard<mutex> g(mu);
            scr_total++;
            if (ex_s) scr_exact++;
            sum_scr_ms += ms_s;
            pert_exact += pe;
            pert_total += pt;
            sum_pert_ms += p_ms;
            fprintf(stderr, "%s dag=%d | scratch_exact=%d (%zu, %.1fms) | perturb_exact=%d/%d (avg %.1fms)\n",
                    pz.id.c_str(), dag_len, ex_s ? 1 : 0, p_scr.size(), ms_s, pe, pt, pt ? p_ms / pt : 0.0);
        }
    };
    vector<thread> th;
    int nt = max(1, min(P.threads, (int)puzzles.size()));
    for (int i = 0; i < nt; i++) th.emplace_back(worker);
    for (auto &t : th) t.join();
    fprintf(stderr, "SCHED-TEST TOTAL: scratch_exact=%d/%d (avg %.1fms) | perturb_exact=%d/%d (avg %.1fms)\n",
            scr_exact, scr_total, scr_total ? sum_scr_ms / scr_total : 0.0,
            pert_exact, pert_total, pert_total ? sum_pert_ms / pert_total : 0.0);
    return 0;
}

static int run_export_pool(const vector<PuzzleIn> &puzzles, const Params &P) {
    auto progs = load_progs_file(P.lns_path);
    vector<string> out(puzzles.size());
    atomic<int> nextp{0};
    mutex mu;
    long long tot_recs = 0, tot_hubs = 0;

    auto worker = [&]() {
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            const auto &pz = puzzles[k];
            auto it = progs.find(pz.id);
            if (it == progs.end()) continue;
            mt19937_64 rng((u64)P.seed * 1000003ULL + hash<string>()(pz.id));

            Dag d0;
            d0.init(pz.L, pz.s, pz.t);
            if (!d0.from_prog(it->second)) continue;
            d0.analyse();
            int n = (int)d0.nodes.size();
            int nt = (int)d0.targets.size();
            vector<int> seq0;
            for (int u = d0.ks; u < n; u++) seq0.push_back(u);
            vector<int> lu0;
            auto pk0 = d0.eval_order(seq0, &lu0);
            vector<int> step_peak;
            if (pk0.max_peak >= 47 && !seq0.empty()) {
                int M0 = (int)seq0.size();
                step_peak.assign(M0, 0);
                int live0 = 0;
                for (int s = 0; s < d0.ks; s++) if (lu0[s] >= 0) live0++;
                for (int idx = 0; idx < M0; idx++) {
                    int u = seq0[idx];
                    int a = d0.nodes[u].a, b = d0.nodes[u].b;
                    int freed = (lu0[a] == idx ? 1 : 0) + (b != a && lu0[b] == idx ? 1 : 0);
                    step_peak[idx] = live0 + (freed == 0 ? 1 : 0);
                    if (lu0[u] > idx) live0++;
                    live0 -= freed;
                }
            }

            set<RecipeRec> recs;
            unordered_set<u64> init_have;
            unordered_set<u64> start_set(pz.s.begin(), pz.s.end());
            unordered_set<u64> nonstart_vals;
            for (int u = 0; u < d0.ks; u++) init_have.insert(d0.nodes[u].v);
            for (int u = d0.ks; u < n; u++) {
                const auto &nd = d0.nodes[u];
                init_have.insert(nd.v);
                nonstart_vals.insert(nd.v);
                recs.insert({nd.v, nd.op, d0.nodes[nd.a].v, d0.nodes[nd.b].v});
            }

            PoolDict pdict;
            vector<u64> all_built;
            unordered_set<u64> tset(d0.targets.begin(), d0.targets.end());
            unordered_map<u64, int> fanout;
            unordered_set<u64> near_vals;
            for (int u = 0; u < n; u++) {
                all_built.push_back(d0.nodes[u].v);
                fanout[d0.nodes[u].v] = (int)d0.users[u].size();
            }
            vector<u64> leaves, done_t;
            vector<int> lcosts;
            int s1_cnt = 0;
            select_leaves(pz.L, pz.s, all_built, tset, fanout, near_vals, P.n_leaves,
                          leaves, lcosts, s1_cnt, done_t, false);
            pdict.build(pz.L, pz.s, all_built, leaves, lcosts, s1_cnt, done_t, P.s2_leaves, P.s2_cap);
            vector<int> s0_dag(pdict.n0);
            for (int i = 0; i < pdict.n0; i++) {
                auto jt = d0.val2idx.find(pdict.e[i].v);
                s0_dag[i] = (jt != d0.val2idx.end()) ? jt->second : 0;
            }

            auto emit_step_checked = [&](u64 r, char op, u64 a, u64 b) -> bool {
                if (r == 0 || r > pz.L || start_set.count(r) || r == a || r == b) return false;
                u64 chk = 0;
                if (!apply_op(op, a, b, pz.L, chk) || chk != r) return false;
                recs.insert({r, op, a, b});
                return true;
            };

            auto downstream_for = [&](int node_u) {
                vector<char> ds_mask(n, 0);
                ds_mask[node_u] = 1;
                vector<int> stk{node_u};
                while (!stk.empty()) {
                    int x = stk.back(); stk.pop_back();
                    for (int w : d0.users[x]) if (!ds_mask[w]) { ds_mask[w] = 1; stk.push_back(w); }
                }
                return ds_mask;
            };

            auto build_cheap_val = [&](u64 val, const vector<char> &ds_s0, const unordered_set<u64> &ds_vals,
                                       u64 target_v, vector<Step> &out_steps) -> bool {
                out_steps.clear();
                if (start_set.count(val)) return true;
                auto it_u = d0.val2idx.find(val);
                if (it_u != d0.val2idx.end() && !ds_vals.count(val)) return true;
                int id = pdict.get_valid(val, &ds_s0);
                if (id < 0 || pdict.e[id].cost > 1) return false;
                unordered_set<u64> empty_extra;
                LocalBuf loc;
                return emit_dict_rec(pdict, id, &ds_s0, &ds_vals, target_v, empty_extra, loc, out_steps);
            };

            // Rule 1: All 1-op DAG closure among existing DAG node values
            for (int i = 0; i < n; i++) {
                u64 va = d0.nodes[i].v;
                for (int j = i; j < n; j++) {
                    u64 vb = d0.nodes[j].v;
                    if (va <= pz.L - vb && nonstart_vals.count(va + vb)) emit_step_checked(va + vb, '+', va, vb);
                    if (va > vb && nonstart_vals.count(va - vb)) emit_step_checked(va - vb, '-', va, vb);
                    else if (vb > va && nonstart_vals.count(vb - va)) emit_step_checked(vb - va, '-', vb, va);
                    if ((u128)va * vb <= (u128)pz.L && nonstart_vals.count(va * vb)) emit_step_checked(va * vb, '*', va, vb);
                    if (va % vb == 0 && nonstart_vals.count(va / vb)) emit_step_checked(va / vb, '/', va, vb);
                    if (va != vb && vb % va == 0 && nonstart_vals.count(vb / va)) emit_step_checked(vb / va, '/', vb, va);
                }
            }

            unordered_map<u64, u64> hub_votes;
            unordered_set<u64> egraph_mids;

            // Rules 2, 3, 4: Algebraic E-Graph rewrites over each DAG node u
            for (int u = d0.ks; u < n; u++) {
                const auto &nd = d0.nodes[u];
                u64 Y = nd.v, va = d0.nodes[nd.a].v, vb = d0.nodes[nd.b].v;
                vector<char> ds_mask = downstream_for(u);
                vector<char> ds_s0(pdict.n0, 0);
                for (int i = 0; i < pdict.n0; i++) ds_s0[i] = ds_mask[s0_dag[i]];
                unordered_set<u64> ds_vals;
                for (int w = d0.ks; w < n; w++) if (ds_mask[w]) ds_vals.insert(d0.nodes[w].v);

                // Rule 2: 2-step additive reassociation
                if (nd.op == '+' || nd.op == '-') {
                    auto handle_3terms = [&](int s1, u64 t1, int s2, u64 t2, int s3, u64 t3) {
                        int sg[3] = {s1, s2, s3};
                        u64 tm[3] = {t1, t2, t3};
                        int pairs[2][3] = {{0, 2, 1}, {1, 2, 0}};
                        for (int pr = 0; pr < 2; pr++) {
                            int i = pairs[pr][0], j = pairs[pr][1], remk = pairs[pr][2];
                            int si = sg[i], sj = sg[j], sk = sg[remk];
                            u64 xi = tm[i], xj = tm[j], xk = tm[remk];
                            i64 m_signed = (i64)si * (i64)xi + (i64)sj * (i64)xj;
                            if (m_signed == 0) continue;
                            u64 M = (u64)(m_signed > 0 ? m_signed : -m_signed);
                            if (M > pz.L || M == Y || start_set.count(M)) continue;
                            char mop = (si == sj) ? '+' : '-';
                            u64 ma = (si == sj) ? xi : (si > 0 ? (xi > xj ? xi : xj) : (xj > xi ? xj : xi));
                            u64 mb = (ma == xi) ? xj : xi;
                            char yop = 0; u64 ya = 0, yb = 0;
                            if (m_signed > 0) {
                                yop = (sk > 0) ? '+' : '-';
                                ya = M; yb = xk;
                            } else {
                                if (sk <= 0) continue;
                                yop = '-';
                                ya = xk; yb = M;
                            }
                            u64 c1 = 0, c2 = 0;
                            if (M == ma || M == mb || Y == ya || Y == yb) continue;
                            if (apply_op(mop, ma, mb, pz.L, c1) && c1 == M &&
                                apply_op(yop, ya, yb, pz.L, c2) && c2 == Y) {
                                emit_step_checked(M, mop, ma, mb);
                                emit_step_checked(Y, yop, ya, yb);
                                if (!init_have.count(M)) {
                                    egraph_mids.insert(M);
                                    hub_votes[M] |= d0.owners[u];
                                }
                            }
                        }
                    };
                    if (nd.a >= d0.ks && (d0.nodes[nd.a].op == '+' || d0.nodes[nd.a].op == '-')) {
                        const auto &ch = d0.nodes[nd.a];
                        handle_3terms(1, d0.nodes[ch.a].v, ch.op == '+' ? 1 : -1, d0.nodes[ch.b].v,
                                      nd.op == '+' ? 1 : -1, vb);
                    }
                    if (nd.b >= d0.ks && (d0.nodes[nd.b].op == '+' || d0.nodes[nd.b].op == '-')) {
                        const auto &ch = d0.nodes[nd.b];
                        int sb = (nd.op == '+') ? 1 : -1;
                        handle_3terms(1, va, sb, d0.nodes[ch.a].v, (ch.op == '+') ? sb : -sb, d0.nodes[ch.b].v);
                    }
                    // Rule 4b: GCD factoring A +- B = g * (Y/g)
                    u64 g = bgcd(va, vb);
                    if (g >= 2 && Y % g == 0) {
                        u64 q = Y / g;
                        vector<Step> sg_st, sq_st;
                        if (q >= 2 && build_cheap_val(g, ds_s0, ds_vals, Y, sg_st) &&
                            build_cheap_val(q, ds_s0, ds_vals, Y, sq_st)) {
                            for (const auto &st : sg_st) emit_step_checked(st.r, st.op, st.a, st.b);
                            for (const auto &st : sq_st) emit_step_checked(st.r, st.op, st.a, st.b);
                            emit_step_checked(Y, '*', g, q);
                        }
                    }
                }

                // Rule 3 & Rule 4a: Multiplicative associativity, factor-shifting, distributivity
                if (nd.op == '*') {
                    // 3a. Associativity: (P * Q) * B -> P * (Q * B), Q * (P * B)
                    if (nd.a >= d0.ks && d0.nodes[nd.a].op == '*') {
                        u64 P = d0.nodes[d0.nodes[nd.a].a].v, Q = d0.nodes[d0.nodes[nd.a].b].v;
                        for (auto [x, y] : {make_pair(P, Q), make_pair(Q, P)}) {
                            if (x >= 2 && vb >= 2 && (u128)x * vb <= (u128)pz.L) {
                                u64 M = x * vb;
                                if (!start_set.count(M) && M != Y && !ds_vals.count(M)) {
                                    emit_step_checked(M, '*', x, vb);
                                    emit_step_checked(Y, '*', M, y);
                                    if (!init_have.count(M)) hub_votes[M] |= d0.owners[u];
                                }
                            }
                        }
                    }
                    if (nd.b >= d0.ks && d0.nodes[nd.b].op == '*') {
                        u64 P = d0.nodes[d0.nodes[nd.b].a].v, Q = d0.nodes[d0.nodes[nd.b].b].v;
                        for (auto [x, y] : {make_pair(P, Q), make_pair(Q, P)}) {
                            if (va >= 2 && x >= 2 && (u128)va * x <= (u128)pz.L) {
                                u64 M = va * x;
                                if (!start_set.count(M) && M != Y && !ds_vals.count(M)) {
                                    emit_step_checked(M, '*', va, x);
                                    emit_step_checked(Y, '*', M, y);
                                    if (!init_have.count(M)) hub_votes[M] |= d0.owners[u];
                                }
                            }
                        }
                    }

                    // 3b. Factor shifting A*B -> (A/g)*(B*g) and (A*g)*(B/g)
                    vector<u64> g_cands{2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64};
                    vector<pair<u64, int>> pf;
                    factor64(va, pf, rng);
                    for (auto [p, e] : pf) if (p <= 1000000ULL) g_cands.push_back(p);
                    factor64(vb, pf, rng);
                    for (auto [p, e] : pf) if (p <= 1000000ULL) g_cands.push_back(p);
                    for (int s = 0; s < d0.ks; s++) if (d0.nodes[s].v >= 2) g_cands.push_back(d0.nodes[s].v);
                    for (int d_idx = d0.ks; d_idx < u; d_idx++) {
                        u64 gv = d0.nodes[d_idx].v;
                        if (gv >= 2 && (va % gv == 0 || vb % gv == 0)) g_cands.push_back(gv);
                    }
                    sort(g_cands.begin(), g_cands.end());
                    g_cands.erase(unique(g_cands.begin(), g_cands.end()), g_cands.end());

                    auto try_shift = [&](u64 X, u64 Z, u64 g) {
                        if (g < 2 || X % g != 0 || (u128)Z * g > (u128)pz.L) return;
                        u64 X2 = X / g, Z2 = Z * g;
                        if (X2 < 2 || Z2 < 2 || start_set.count(X2) || start_set.count(Z2) ||
                            ds_vals.count(X2) || ds_vals.count(Z2) || X2 == Y || Z2 == Y) return;

                        vector<Step> sx, sz;
                        bool x_pdict = build_cheap_val(X2, ds_s0, ds_vals, Y, sx);
                        bool z_pdict = build_cheap_val(Z2, ds_s0, ds_vals, Y, sz);
                        if (!x_pdict && !z_pdict) return;  // filter: at least one must be cheap in pdict / d0!

                        vector<Step> gx_st;
                        if (!x_pdict) {
                            if (!build_cheap_val(g, ds_s0, ds_vals, Y, gx_st)) return;
                            sx = gx_st;
                            sx.push_back({'/', X, g, X2});
                        }
                        if (!z_pdict) {
                            if (g == 2) {
                                sz.push_back({'+', Z, Z, Z2});
                            } else {
                                if (!build_cheap_val(g, ds_s0, ds_vals, Y, gx_st)) return;
                                sz = gx_st;
                                sz.push_back({'*', Z, g, Z2});
                            }
                        }
                        for (const auto &st : sx) emit_step_checked(st.r, st.op, st.a, st.b);
                        for (const auto &st : sz) emit_step_checked(st.r, st.op, st.a, st.b);
                        if (emit_step_checked(Y, '*', X2, Z2)) {
                            if (!init_have.count(X2)) hub_votes[X2] |= d0.owners[u];
                            if (!init_have.count(Z2)) hub_votes[Z2] |= d0.owners[u];
                        }
                    };
                    for (u64 g : g_cands) {
                        if (va % g == 0) try_shift(va, vb, g);
                        if (vb % g == 0) try_shift(vb, va, g);
                    }

                    // 4a. Distributivity: (P +- Q) * B -> P*B +- Q*B
                    auto try_distrib = [&](int ch_idx, u64 K) {
                        if (ch_idx < d0.ks) return;
                        const auto &ch = d0.nodes[ch_idx];
                        if (ch.op != '+' && ch.op != '-') return;
                        u64 P = d0.nodes[ch.a].v, Q = d0.nodes[ch.b].v;
                        if ((u128)P * K > (u128)pz.L || (u128)Q * K > (u128)pz.L) return;
                        u64 PK = P * K, QK = Q * K;
                        if (start_set.count(PK) || start_set.count(QK) || ds_vals.count(PK) || ds_vals.count(QK)) return;
                        int id_p = pdict.get_valid(PK, &ds_s0), id_q = pdict.get_valid(QK, &ds_s0);
                        bool has_p = d0.val2idx.count(PK) || (id_p >= 0 && pdict.e[id_p].cost <= 1);
                        bool has_q = d0.val2idx.count(QK) || (id_q >= 0 && pdict.e[id_q].cost <= 1);
                        if (!has_p && !has_q) return;
                        u64 chk = 0;
                        if (!apply_op(ch.op, PK, QK, pz.L, chk) || chk != Y) return;
                        emit_step_checked(PK, '*', P, K);
                        emit_step_checked(QK, '*', Q, K);
                        emit_step_checked(Y, ch.op, PK, QK);
                    };
                    try_distrib(nd.a, vb);
                    try_distrib(nd.b, va);
                }
            }

            // Rule 5: Cheap alternatives for non-target shared helpers (users >= 2)
            SearchStats st;
            for (int u = d0.ks; u < n; u++) {
                if (d0.users[u].size() < 2 || tset.count(d0.nodes[u].v)) continue;
                vector<char> excl = downstream_for(u);
                vector<char> excl_s0(pdict.n0, 0);
                for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[s0_dag[i]];
                unordered_set<u64> excl_vals;
                for (int w = d0.ks; w < n; w++) if (excl[w] && w != u) excl_vals.insert(d0.nodes[w].v);

                DeepSearch ds(pdict, P, rng);
                ds.excl_s0 = &excl_s0;
                ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
                auto sols = ds.search(d0.nodes[u].v, 3, 5, 1, st, 1);
                for (const auto &sol : sols) {
                    for (const auto &step : sol.steps) emit_step_checked(step.r, step.op, step.a, step.b);
                }
            }

            // Rule 6: Per-target alternative recipes (max_alts=16, slack=1) + hub voting
            for (int ti = 0; ti < nt; ti++) {
                u64 bit = 1ULL << ti;
                vector<char> inR(n, 0);
                int rs = 0;
                for (int u = d0.ks; u < n; u++) {
                    if (d0.owners[u] && (d0.owners[u] & ~bit) == 0) { inR[u] = 1; rs++; }
                }
                int tn = d0.tnode[ti];
                vector<char> excl = inR;
                excl[tn] = 1;
                vector<int> stk{tn};
                while (!stk.empty()) {
                    int x = stk.back(); stk.pop_back();
                    for (int w : d0.users[x]) if (!excl[w]) { excl[w] = 1; stk.push_back(w); }
                }
                vector<char> excl_s0(pdict.n0, 0);
                for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[s0_dag[i]];
                unordered_set<u64> excl_vals;
                for (int u = d0.ks; u < n; u++) if (excl[u] && !inR[u] && u != tn) excl_vals.insert(d0.nodes[u].v);

                DeepSearch ds(pdict, P, rng);
                ds.excl_s0 = &excl_s0;
                ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
                int bud = min(max(rs + 1, 4), 9);
                auto sols = ds.search(d0.targets[ti], bud, 16, 1, st, 2);
                for (const auto &sol : sols) {
                    for (const auto &step : sol.steps) {
                        if (emit_step_checked(step.r, step.op, step.a, step.b)) {
                            if (!init_have.count(step.r) && step.r != d0.targets[ti]) {
                                hub_votes[step.r] |= bit;
                            }
                        }
                    }
                }
                if (pk0.max_peak >= 47) {
                    int tpos = d0.tnode[ti] - d0.ks;
                    vector<char> safe_excl_s0 = excl_s0;
                    for (int i = 0; i < pdict.n0; i++) {
                        int u = s0_dag[i];
                        if (u >= d0.ks && !safe_excl_s0[i]) {
                            int lu_u = lu0[u];
                            if (lu_u >= 0 && lu_u < tpos) {
                                int lim = min(tpos, (int)step_peak.size() - 1);
                                for (int idx = lu_u; idx <= lim; idx++) {
                                    if (step_peak[idx] >= 48) { safe_excl_s0[i] = 1; break; }
                                }
                            }
                        }
                    }
                    DeepSearch ds_safe(pdict, P, rng);
                    ds_safe.excl_s0 = &safe_excl_s0;
                    ds_safe.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
                    auto sols_s = ds_safe.search(d0.targets[ti], bud, 8, 1, st, 2);
                    for (const auto &sol : sols_s) {
                        for (const auto &step : sol.steps) {
                            if (emit_step_checked(step.r, step.op, step.a, step.b)) {
                                if (!init_have.count(step.r) && step.r != d0.targets[ti]) {
                                    hub_votes[step.r] |= bit;
                                }
                            }
                        }
                    }
                }
            }

            // Shared-hub voting pass: promote top 32 non-pool intermediates used by >= 2 targets
            vector<pair<int, u64>> vlist;
            for (auto &[hv, m] : hub_votes) {
                int pc = __builtin_popcountll(m);
                if (pc >= 2) vlist.emplace_back(pc, hv);
            }
            sort(vlist.rbegin(), vlist.rend());
            vector<u64> top_hubs;
            for (size_t i = 0; i < vlist.size() && i < 32; i++) top_hubs.push_back(vlist[i].second);
            if (!top_hubs.empty()) {
                pdict.add_leaves_inc(top_hubs);
                for (int ti = 0; ti < nt; ti++) {
                    u64 bit = 1ULL << ti;
                    vector<char> inR(n, 0);
                    int rs = 0;
                    for (int u = d0.ks; u < n; u++) {
                        if (d0.owners[u] && (d0.owners[u] & ~bit) == 0) { inR[u] = 1; rs++; }
                    }
                    int tn = d0.tnode[ti];
                    vector<char> excl = inR;
                    excl[tn] = 1;
                    vector<int> stk{tn};
                    while (!stk.empty()) {
                        int x = stk.back(); stk.pop_back();
                        for (int w : d0.users[x]) if (!excl[w]) { excl[w] = 1; stk.push_back(w); }
                    }
                    vector<char> excl_s0(pdict.n0, 0);
                    for (int i = 0; i < pdict.n0; i++) excl_s0[i] = excl[s0_dag[i]];
                    unordered_set<u64> excl_vals;
                    for (int u = d0.ks; u < n; u++) if (excl[u] && !inR[u] && u != tn) excl_vals.insert(d0.nodes[u].v);

                    DeepSearch ds(pdict, P, rng);
                    ds.excl_s0 = &excl_s0;
                    ds.excl_vals = excl_vals.empty() ? nullptr : &excl_vals;
                    auto sols = ds.search(d0.targets[ti], min(max(rs + 1, 4), 8), 12, 1, st, 1);
                    for (const auto &sol : sols) {
                        for (const auto &step : sol.steps) emit_step_checked(step.r, step.op, step.a, step.b);
                    }
                }
            }

            ostringstream os;
            os << pz.id << ' ' << recs.size() << '\n';
            for (const auto &rc : recs) os << rc.r << ' ' << rc.op << ' ' << rc.a << ' ' << rc.b << '\n';
            out[k] = os.str();
            lock_guard<mutex> g(mu);
            tot_recs += (long long)recs.size();
            tot_hubs += (long long)top_hubs.size();
            fprintf(stderr, "%s: pool=%zu recipes (voted_hubs=%zu)\n", pz.id.c_str(), recs.size(), top_hubs.size());
        }
    };
    vector<thread> th;
    int nt = max(1, min(P.threads, (int)puzzles.size()));
    for (int i = 0; i < nt; i++) th.emplace_back(worker);
    for (auto &t : th) t.join();

    FILE *fo = fopen(P.pool_path.c_str(), "w");
    if (!fo) { fprintf(stderr, "cannot open %s\n", P.pool_path.c_str()); return 1; }
    for (const auto &s : out) if (!s.empty()) fwrite(s.data(), 1, s.size(), fo);
    fclose(fo);
    fprintf(stderr, "POOL TOTAL: recipes=%lld voted_hubs=%lld -> %s\n", tot_recs, tot_hubs, P.pool_path.c_str());
    return 0;
}

static int run_dag_mode(const vector<PuzzleIn> &puzzles, const Params &P) {
    unordered_map<string, map<u64, Step>> dags;
    {
        ifstream f(P.dag_path);
        string pid;
        int m = 0;
        while (f >> pid >> m) {
            map<u64, Step> defs;
            for (int i = 0; i < m; i++) {
                u64 r = 0, a = 0, b = 0;
                char op = 0;
                f >> r >> op >> a >> b;
                defs[r] = {op, a, b, r};
            }
            dags[pid] = move(defs);
        }
    }
    unordered_map<string, vector<Instr>> pref_progs;
    if (!P.pref_path.empty()) pref_progs = load_progs_file(P.pref_path);

    vector<string> out(puzzles.size());
    atomic<int> nextp{0};
    mutex mu;
    long long tot_dag = 0, tot_prog = 0;
    auto worker = [&]() {
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            const auto &pz = puzzles[k];
            auto it = dags.find(pz.id);
            if (it == dags.end()) continue;
            mt19937_64 rng((u64)P.seed * 1000003ULL + hash<string>()(pz.id));

            vector<u64> pref_ord;
            auto pit = pref_progs.find(pz.id);
            if (pit != pref_progs.end()) pref_ord = extract_prog_order(pz.s, pit->second);

            Dag dag;
            dag.init(pz.L, pz.s, pz.t);
            if (!dag.from_defs(it->second, pref_ord)) {
                lock_guard<mutex> g(mu);
                fprintf(stderr, "%s: DAG cycle or missing def!\n", pz.id.c_str());
                continue;
            }
            int dlen = (int)dag.nodes.size() - dag.ks;
            vector<Instr> prog;
            if (!dag.compile(prog, max(P.reorder_tries, 64), rng,
                             pref_ord.empty() ? nullptr : &pref_ord, nullptr, P.dag_sa_iters)) {
                lock_guard<mutex> g(mu);
                fprintf(stderr, "%s: DAG compile failed!\n", pz.id.c_str());
                continue;
            }
            string why = verify_prog(pz.L, pz.s, pz.t, prog);
            if (!why.empty()) {
                lock_guard<mutex> g(mu);
                fprintf(stderr, "%s: DAG verify failed: %s\n", pz.id.c_str(), why.c_str());
                continue;
            }
            out[k] = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);
            lock_guard<mutex> g(mu);
            tot_dag += dlen;
            tot_prog += (long long)prog.size();
            fprintf(stderr, "%s: dag=%d -> prog=%zu (spills=%d)\n",
                    pz.id.c_str(), dlen, prog.size(), (int)prog.size() - dlen);
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
    fprintf(stderr, "DAG COMPILE TOTAL: dag=%lld -> prog=%lld (spills=%lld)\n",
            tot_dag, tot_prog, tot_prog - tot_dag);
    return 0;
}

// ---------------------------------------------------------------- Mode 5: Macro-Subprogram SA + Re-Plan
static vector<u64> discover_cf_and_gcd_bases(u64 L, const vector<u64> &targets_sub,
                                             const unordered_set<u64> &have,
                                             const unordered_set<u64> &banned_vals) {
    vector<u64> st;
    for (u64 t : targets_sub) if (!have.count(t) && t >= 256) st.push_back(t);
    sort(st.begin(), st.end());
    st.erase(unique(st.begin(), st.end()), st.end());
    if (st.size() < 2) return {};

    vector<u64> cand_g;
    int n20 = min(20, (int)st.size());
    int n10 = min(10, (int)st.size());
    // 1. Pairwise exact + small-offset GCDs
    for (int i = 0; i < n20; i++) {
        for (int j = i + 1; j < n20; j++) {
            u64 g = bgcd(st[i], st[j]);
            if (g >= 64 && g <= L / 100) cand_g.push_back(g);
        }
    }
    for (int i = 0; i < n10; i++) {
        for (int j = i + 1; j < n10; j++) {
            for (int da = -8; da <= 8; da++) {
                if ((i64)st[i] + da <= 0) continue;
                u64 va = (u64)((i64)st[i] + da);
                for (int db = -8; db <= 8; db++) {
                    if ((i64)st[j] + db <= 0) continue;
                    u64 g = bgcd(va, (u64)((i64)st[j] + db));
                    if (g >= 128 && g <= L / 100) cand_g.push_back(g);
                }
            }
        }
    }
    // 2. Continued-fraction convergents h/k of T_j / T_i (k <= 4096)
    for (int i = 0; i < n20; i++) {
        for (int j = i + 1; j < n20; j++) {
            u64 p = st[j], q = st[i], h0 = 0, h1 = 1, k0 = 1, k1 = 0;
            for (int step = 0; step < 32 && q > 0; step++) {
                u64 a = p / q, r = p % q;
                p = q; q = r;
                u128 nh = (u128)a * h1 + h0, nk = (u128)a * k1 + k0;
                if (nh > 4096 || nk > 4096) break;
                h0 = h1; h1 = (u64)nh;
                k0 = k1; k1 = (u64)nk;
                if (k1 >= 2) {
                    u64 B = (st[i] + k1 / 2) / k1;
                    if (B >= 128 && B <= L / 4) cand_g.push_back(B);
                }
            }
        }
    }
    sort(cand_g.begin(), cand_g.end());
    cand_g.erase(unique(cand_g.begin(), cand_g.end()), cand_g.end());

    struct RankedBase { double sc; u64 b; };
    vector<RankedBase> ranked;
    for (u64 B : cand_g) {
        if (have.count(B) || banned_vals.count(B)) continue;
        int hits = 0;
        u64 tol = max<u64>(64, B >> 12);
        for (u64 t : st) {
            u64 q = (t + B / 2) / B;
            if (q >= 1 && q <= 4096) {
                u128 qb = (u128)q * B;
                u64 r = (u64)(t >= qb ? (u128)t - qb : qb - (u128)t);
                if (r <= tol) hits++;
            }
        }
        if (hits >= 3) ranked.push_back({(double)hits * log2((double)B), B});
    }
    sort(ranked.begin(), ranked.end(), [](const RankedBase &a, const RankedBase &b) { return a.sc > b.sc; });
    vector<u64> out;
    for (size_t i = 0; i < ranked.size() && out.size() < 4; i++) out.push_back(ranked[i].b);
    // Add pairwise products of top 4 bases if within L
    size_t nbase = out.size();
    for (size_t i = 0; i < nbase && out.size() < 4; i++) {
        for (size_t j = i + 1; j < nbase && out.size() < 4; j++) {
            u128 pr = (u128)out[i] * out[j];
            if (pr <= (u128)L && !have.count((u64)pr) && !banned_vals.count((u64)pr)) {
                out.push_back((u64)pr);
            }
        }
    }
    return out;
}

static bool construct_subprogram(
    u64 L, const vector<u64> &starts, const vector<u64> &all_targets,
    const unordered_set<u64> &init_have, const vector<u64> &init_built,
    const map<u64, Step> &init_defs, const vector<u64> &init_order,
    const unordered_map<u64, int> &init_fanout,
    const unordered_set<u64> &banned_vals,
    const Params &P, mt19937_64 &rng, SearchStats &stats,
    double la_weight_override, double jitter, int max_dag_budget,
    map<u64, Step> &out_defs, vector<u64> &out_order, int &out_dag_len) {

    unordered_set<u64> have = init_have;
    vector<u64> built = init_built;
    map<u64, Step> defs = init_defs;
    vector<u64> order = init_order;
    unordered_map<u64, int> fanout = init_fanout;
    unordered_map<u64, int> def_pos;
    for (size_t i = 0; i < order.size(); i++) def_pos[order[i]] = (int)i;
    unordered_set<u64> tset(all_targets.begin(), all_targets.end());
    unordered_set<u64> near_vals;

    vector<u64> remaining;
    for (u64 t : all_targets) if (!have.count(t)) remaining.push_back(t);
    if (remaining.empty()) {
        Dag td;
        td.init(L, starts, all_targets);
        if (!td.from_defs(defs, order)) return false;
        out_defs = move(defs); out_order = move(order);
        out_dag_len = (int)td.nodes.size() - td.ks;
        return true;
    }

    vector<u64> bonus_hubs = discover_cf_and_gcd_bases(L, remaining, have, banned_vals);
    for (u64 bh : bonus_hubs) near_vals.insert(bh);

    PoolDict dict;
    DeepSearch ds(dict, P, rng);

    auto apply_steps = [&](const vector<Step> &steps) {
        for (const auto &s : steps) {
            if (!have.count(s.r)) {
                u64 chk = 0;
                if (!have.count(s.a) || !have.count(s.b) || !apply_op(s.op, s.a, s.b, L, chk) || chk != s.r) continue;
                have.insert(s.r);
                built.push_back(s.r);
                defs[s.r] = s;
                order.push_back(s.r);
                def_pos[s.r] = (int)order.size() - 1;
                fanout[s.a]++;
                if (s.b != s.a) fanout[s.b]++;
            }
        }
    };

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

    bool boot_ladder_used = !init_order.empty();
    int hub_commits = 0, consec_hub = 0;
    unordered_set<u64> sset_c(starts.begin(), starts.end());

    while (!remaining.empty()) {
        if ((int)order.size() + (int)remaining.size() > max_dag_budget) return false;
        ds.deep_beam = (remaining.size() <= 1);

        vector<u64> leaves, done_t;
        vector<int> lcosts;
        int s1_cnt = 0;
        double tb = now_sec();
        select_leaves(L, starts, built, tset, fanout, near_vals, P.n_leaves, leaves, lcosts, s1_cnt, done_t, true);
        dict.build(L, starts, built, leaves, lcosts, s1_cnt, done_t, P.s2_leaves, P.s2_cap);
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
            for (size_t i = 0; i < remaining.size() && (i < 16 || cands.empty()); i++) {
                auto sols = ds.search(remaining[i], 28, P.la_alts, 2, stats, 2);
                for (auto &s : sols) cands.push_back({remaining[i], s.cost, s.engine, move(s.steps)});
            }
        }

        if (cands.empty()) {
            u64 t_min = *min_element(remaining.begin(), remaining.end());
            auto boot = ds.search(t_min, 36, 1, 0, stats, 3);
            if (!boot.empty()) {
                apply_steps(boot[0].steps);
            } else if (!boot_ladder_used && !starts.empty()) {
                boot_ladder_used = true;
                size_t before = order.size();
                u64 s0 = starts[0];
                if (!have.count(1)) apply_steps({{'/', s0, s0, 1}});
                vector<u64> pws{1, 2, 4, 16, 256, 65536};
                if (!have.count(2) && 2 <= L) apply_steps({{'+', 1, 1, 2}});
                for (size_t pi = 2; pi < pws.size(); pi++) {
                    u64 prev = pws[pi - 1];
                    if (have.count(prev) && (u128)prev * prev <= (u128)L) {
                        apply_steps({{'*', prev, prev, prev * prev}});
                    }
                }
                if (starts.size() >= 2 && (u128)starts[0] * starts[1] <= (u128)L) {
                    apply_steps({{'*', starts[0], starts[1], starts[0] * starts[1]}});
                }
                if (order.size() == before) build_binary(t_min);
            } else {
                build_binary(t_min);
            }
            consec_hub = 0;
        } else {
            vector<PlanCand> hub_cands;
            if (hub_commits < 6 && consec_hub < 3 && remaining.size() >= 2) {
                unordered_map<u64, unordered_set<u64>> mid_targets;
                for (const auto &c : cands) {
                    if (c.target == 0) continue;
                    for (const auto &st : c.steps) {
                        if (st.r != c.target && !have.count(st.r) && !banned_vals.count(st.r))
                            mid_targets[st.r].insert(c.target);
                    }
                }
                vector<pair<int, u64>> mranked;
                for (const auto &[hv, ts_set] : mid_targets) {
                    if (ts_set.size() >= 2) mranked.emplace_back((int)ts_set.size(), hv);
                }
                sort(mranked.rbegin(), mranked.rend());
                unordered_set<u64> picked_hubs;
                for (size_t hi = 0; hi < mranked.size() && hub_cands.size() < 3; hi++) {
                    u64 hv = mranked[hi].second;
                    if (!picked_hubs.insert(hv).second) continue;
                    vector<Step> best_pref;
                    for (const auto &c : cands) {
                        for (size_t j = 0; j < c.steps.size(); j++) {
                            if (c.steps[j].r == hv) {
                                unordered_set<u64> need{hv};
                                vector<Step> clean;
                                for (int q = (int)j; q >= 0; --q) {
                                    if (need.count(c.steps[q].r)) {
                                        clean.push_back(c.steps[q]);
                                        need.insert(c.steps[q].a);
                                        need.insert(c.steps[q].b);
                                    }
                                }
                                reverse(clean.begin(), clean.end());
                                if (!clean.empty() && clean.size() <= 4 &&
                                    (best_pref.empty() || clean.size() < best_pref.size())) {
                                    best_pref = move(clean);
                                }
                            }
                        }
                    }
                    if (!best_pref.empty()) hub_cands.push_back({0, (int)best_pref.size(), 5, move(best_pref)});
                }
                for (u64 bh : bonus_hubs) {
                    if (have.count(bh) || picked_hubs.count(bh) || banned_vals.count(bh)) continue;
                    auto sols = ds.search(bh, 14, 1, 0, stats, 2);
                    if (!sols.empty()) {
                        picked_hubs.insert(bh);
                        hub_cands.push_back({0, sols[0].cost, sols[0].engine, move(sols[0].steps)});
                    }
                }
            }

            stable_sort(cands.begin(), cands.end(), [](const PlanCand &a, const PlanCand &b) {
                return a.cost < b.cost;
            });
            if ((int)cands.size() > P.lookahead) cands.resize(P.lookahead);
            for (auto &hc : hub_cands) cands.push_back(move(hc));

            int K = (int)cands.size();
            int bi = 0;
            double bs = 1e18;
            if (K > 1 && (remaining.size() > 1 || K <= 6)) {
                int cur_ord_n = (int)order.size();
                for (int k = 0; k < K; k++) {
                    vector<u64> add_v;
                    unordered_set<u64> made;
                    int ban_hits = 0;
                    for (const auto &st : cands[k].steps) {
                        add_v.push_back(st.r);
                        made.insert(st.r);
                        if (banned_vals.count(st.r)) ban_hits++;
                    }
                    int old_reuses = 0;
                    for (const auto &st : cands[k].steps) {
                        for (u64 opv : {st.a, st.b}) {
                            if (sset_c.count(opv) || made.count(opv)) continue;
                            auto it_dp = def_pos.find(opv);
                            if (it_dp != def_pos.end() && cur_ord_n - it_dp->second > 42 && fanout[opv] <= 2) {
                                old_reuses++;
                            }
                        }
                    }
                    auto mk = dict.mark();
                    dict.add_leaves_inc(add_v);
                    ds.extra_pool = made;
                    double rem_est = 0.0;
                    for (u64 x : remaining) {
                        if (have.count(x) || made.count(x)) continue;
                        rem_est += ds.estimate_cost(x);
                    }
                    ds.extra_pool.clear();
                    dict.rollback(mk);
                    double sc = (double)cands[k].cost + la_weight_override * rem_est
                                + 0.25 * (double)old_reuses + 3.5 * (double)ban_hits;
                    if (jitter > 0.0) sc += jitter * ((double)(rng() % 1000) / 1000.0);
                    if (sc < bs) { bs = sc; bi = k; }
                }
            }
            bool is_hub = (cands[bi].target == 0);
            if (is_hub) { hub_commits++; consec_hub++; }
            else consec_hub = 0;
            apply_steps(cands[bi].steps);
        }

        vector<u64> rem2;
        for (u64 t : remaining) if (!have.count(t)) rem2.push_back(t);
        remaining.swap(rem2);
    }

    Dag td;
    td.init(L, starts, all_targets);
    if (!td.from_defs(defs, order)) return false;
    out_defs = move(defs);
    out_order = move(order);
    out_dag_len = (int)td.nodes.size() - td.ks;
    return true;
}

struct MacroResult {
    vector<Instr> prog;
    vector<vector<Instr>> archive;
    int iters = 0;
    int dag_wins = 0;
    int cmp_wins = 0;
    int spill_drops = 0;
    int best_delta = 9999;
    SearchStats stats;
};

static MacroResult macro_optimize(u64 L, const vector<u64> &starts, const vector<u64> &all_targets,
                                  const vector<Instr> &init_prog, const Params &P, mt19937_64 &rng) {
    MacroResult res;
    res.prog = init_prog;
    Dag best_dag, cur_dag;
    best_dag.init(L, starts, all_targets);
    if (!best_dag.from_prog(init_prog)) {
        res.archive.push_back(init_prog);
        return res;
    }
    best_dag.analyse();
    cur_dag = best_dag;
    vector<u64> best_ord = extract_prog_order(starts, init_prog);
    vector<u64> cur_ord = best_ord;

    auto dag_sig = [](const Dag &dg) -> u64 {
        vector<u64> hs;
        hs.reserve(dg.nodes.size() - dg.ks);
        for (int u = dg.ks; u < (int)dg.nodes.size(); u++) {
            u64 va = dg.nodes[dg.nodes[u].a].v, vb = dg.nodes[dg.nodes[u].b].v;
            if ((dg.nodes[u].op == '+' || dg.nodes[u].op == '*') && va > vb) swap(va, vb);
            u64 z = mix64(dg.nodes[u].v ^ ((u64)(unsigned char)dg.nodes[u].op << 56));
            z ^= mix64(va * 0x9e3779b97f4a7c15ULL + 17ULL) ^ mix64(vb * 0xbf58476d1ce4e5b9ULL + 31ULL);
            hs.push_back(z);
        }
        sort(hs.begin(), hs.end());
        u64 h = dg.nodes.size() - dg.ks;
        for (u64 x : hs) h = mix64(h ^ x);
        return h;
    };

    unordered_set<u64> arch_sigs;
    res.archive.push_back(init_prog);
    arch_sigs.insert(dag_sig(best_dag));

    struct HelperCluster {
        int u_idx;
        u64 u_val;
        vector<int> dep_t; // indices into cur_dag.targets
        int freed_ops;
    };
    vector<HelperCluster> clusters;
    vector<vector<int>> target_desc_t; // per target ti: all targets tj that depend on ti (incl ti)

    auto rebuild_clusters = [&]() {
        clusters.clear();
        int n = (int)cur_dag.nodes.size();
        int nt = (int)cur_dag.targets.size();
        unordered_map<int, int> node2ti;
        for (int ti = 0; ti < nt; ti++) node2ti[cur_dag.tnode[ti]] = ti;

        auto get_desc_targets = [&](int start_u) -> vector<int> {
            vector<char> vis(n, 0);
            vector<int> st = {start_u}, out_t;
            while (!st.empty()) {
                int x = st.back(); st.pop_back();
                if (vis[x]) continue;
                vis[x] = 1;
                auto it = node2ti.find(x);
                if (it != node2ti.end()) out_t.push_back(it->second);
                for (int w : cur_dag.users[x]) st.push_back(w);
            }
            sort(out_t.begin(), out_t.end());
            return out_t;
        };

        target_desc_t.assign(nt, {});
        for (int ti = 0; ti < nt; ti++) target_desc_t[ti] = get_desc_targets(cur_dag.tnode[ti]);

        for (int u = cur_dag.ks; u < n; u++) {
            vector<int> dt = get_desc_targets(u);
            if ((int)dt.size() < 2 || (int)dt.size() > 16) continue;
            vector<char> is_dep(nt, 0);
            for (int ti : dt) is_dep[ti] = 1;
            vector<char> anc(n, 0);
            vector<int> st;
            for (int s = 0; s < cur_dag.ks; s++) anc[s] = 1;
            for (int ti = 0; ti < nt; ti++) if (!is_dep[ti]) st.push_back(cur_dag.tnode[ti]);
            while (!st.empty()) {
                int x = st.back(); st.pop_back();
                if (anc[x]) continue;
                anc[x] = 1;
                if (x >= cur_dag.ks) {
                    st.push_back(cur_dag.nodes[x].a);
                    st.push_back(cur_dag.nodes[x].b);
                }
            }
            int anc_cnt = 0;
            for (int x = cur_dag.ks; x < n; x++) if (anc[x]) anc_cnt++;
            int freed = (n - cur_dag.ks) - anc_cnt;
            if (freed >= 4) clusters.push_back({u, cur_dag.nodes[u].v, move(dt), freed});
        }
    };
    rebuild_clusters();

    int nt = (int)best_dag.targets.size();
    if (nt <= 1) return res;

    double t0 = now_sec(), deadline = t0 + max(0.5, P.lns_sec);
    int stall = 0;
    static const double la_ws[4] = {0.30, 0.45, 0.65, 0.85};
    static const double jits[4] = {0.0, 0.20, 0.50, 0.90};

    while (now_sec() < deadline) {
        int cur_dag_len = (int)cur_dag.nodes.size() - cur_dag.ks;
        int best_dag_len = (int)best_dag.nodes.size() - best_dag.ks;
        int n = (int)cur_dag.nodes.size();

        int roll = (int)(rng() % 100);
        int mv = (roll < 55 && !clusters.empty()) ? 1 : ((roll < 85 || cur_dag_len > 135) ? 2 : 3);
        if (nt <= 3 && cur_dag_len <= 135 && roll >= 55) mv = 3;

        vector<char> rem_t(nt, 0);
        unordered_set<u64> banned;
        if (mv == 1) {
            const auto &cl = clusters[rng() % clusters.size()];
            for (int ti : cl.dep_t) rem_t[ti] = 1;
            banned.insert(cl.u_val);
            if (rng() % 2 == 0) {
                int ca = cur_dag.nodes[cl.u_idx].a, cb = cur_dag.nodes[cl.u_idx].b;
                if (ca >= cur_dag.ks) banned.insert(cur_dag.nodes[ca].v);
                if (cb >= cur_dag.ks) banned.insert(cur_dag.nodes[cb].v);
            }
        } else if (mv == 2) {
            int k = 3 + (int)(rng() % min(12, max(1, nt - 3)));
            vector<int> seeds;
            int mode2 = (int)(rng() % 3);
            if (mode2 == 0) {
                // Contiguous window along cur_ord
                vector<pair<int, int>> ord_t;
                for (int ti = 0; ti < nt; ti++) ord_t.emplace_back(cur_dag.tnode[ti], ti);
                sort(ord_t.begin(), ord_t.end());
                int st = (int)(rng() % max(1, nt - k + 1));
                for (int j = 0; j < k && st + j < nt; j++) seeds.push_back(ord_t[st + j].second);
            } else {
                vector<int> perm(nt);
                iota(perm.begin(), perm.end(), 0);
                shuffle(perm.begin(), perm.end(), rng);
                for (int j = 0; j < k && j < nt; j++) seeds.push_back(perm[j]);
            }
            for (int ti : seeds) {
                if ((int)target_desc_t[ti].size() <= 14) {
                    for (int dt : target_desc_t[ti]) rem_t[dt] = 1;
                }
            }
            int cnt_r = 0;
            for (int ti = 0; ti < nt; ti++) if (rem_t[ti]) cnt_r++;
            if (cnt_r == 0 || cnt_r >= nt) {
                for (int ti = 0; ti < nt; ti++) rem_t[ti] = 0;
                int picked = 0;
                for (int ti = nt - 1; ti >= 0 && picked < min(nt - 1, 6); ti--) {
                    if ((int)target_desc_t[ti].size() == 1) { rem_t[ti] = 1; picked++; }
                }
                if (picked == 0) continue;
            }
        } else {
            for (int ti = 0; ti < nt; ti++) rem_t[ti] = 1;
            if (rng() % 2 == 0) {
                vector<pair<int, u64>> shs;
                for (int u = cur_dag.ks; u < n; u++)
                    if ((int)cur_dag.users[u].size() >= 3) shs.emplace_back((int)cur_dag.users[u].size(), cur_dag.nodes[u].v);
                sort(shs.rbegin(), shs.rend());
                for (size_t i = 0; i < shs.size() && i < 2; i++) banned.insert(shs[i].second);
            }
        }

        // Compute G_rest = Anc(T \ T_rem) preserving cur_ord
        vector<char> keep(n, 0);
        vector<int> st;
        for (int s = 0; s < cur_dag.ks; s++) keep[s] = 1;
        for (int ti = 0; ti < nt; ti++) if (!rem_t[ti]) st.push_back(cur_dag.tnode[ti]);
        while (!st.empty()) {
            int x = st.back(); st.pop_back();
            if (keep[x]) continue;
            keep[x] = 1;
            if (x >= cur_dag.ks) {
                st.push_back(cur_dag.nodes[x].a);
                st.push_back(cur_dag.nodes[x].b);
            }
        }
        if (mv == 2 && (rng() % 2 == 0)) {
            for (int u = cur_dag.ks; u < n; u++) {
                if (!keep[u] && cur_dag.users[u].size() >= 2) banned.insert(cur_dag.nodes[u].v);
            }
        }

        unordered_set<u64> init_have(starts.begin(), starts.end());
        vector<u64> init_built(starts.begin(), starts.end());
        map<u64, Step> init_defs;
        vector<u64> init_order;
        unordered_map<u64, int> init_fanout;
        for (int u = cur_dag.ks; u < n; u++) {
            if (!keep[u]) continue;
            u64 v = cur_dag.nodes[u].v, va = cur_dag.nodes[cur_dag.nodes[u].a].v, vb = cur_dag.nodes[cur_dag.nodes[u].b].v;
            init_have.insert(v);
            init_built.push_back(v);
            init_defs[v] = {cur_dag.nodes[u].op, va, vb, v};
            init_order.push_back(v);
            init_fanout[va]++;
            if (vb != va) init_fanout[vb]++;
        }

        int n_trials = 1;
        for (int tr = 0; tr < n_trials && now_sec() < deadline; tr++) {
            res.iters++;
            double law = la_ws[(res.iters + tr) % 4];
            double jit = jits[(rng() >> 3) % 4];
            map<u64, Step> out_defs;
            vector<u64> out_ord;
            int new_dag_len = 0;
            bool ok = construct_subprogram(L, starts, all_targets, init_have, init_built, init_defs,
                                           init_order, init_fanout, banned, P, rng, res.stats,
                                           law, jit, best_dag_len + 4, out_defs, out_ord, new_dag_len);
            if (!ok) continue;

            res.best_delta = min(res.best_delta, new_dag_len - best_dag_len);
            Dag td;
            td.init(L, starts, all_targets);
            if (!td.from_defs(out_defs, out_ord)) continue;
            td.analyse();
            u64 sig = dag_sig(td);

            if (tr == 0 && new_dag_len <= best_dag_len + 3) n_trials = 2;

            if (new_dag_len < best_dag_len) {
                res.dag_wins++;
                vector<Instr> nprog;
                if (td.compile(nprog, max(P.reorder_tries, 64), rng, &best_ord, nullptr, 15000) &&
                    verify_prog(L, starts, all_targets, nprog).empty() &&
                    nprog.size() < res.prog.size()) {
                    res.cmp_wins++;
                    res.prog = move(nprog);
                    best_dag = td;
                    best_ord = extract_prog_order(starts, res.prog);
                    cur_dag = td;
                    cur_ord = best_ord;
                    rebuild_clusters();
                    stall = 0;
                    if (arch_sigs.insert(sig).second && res.archive.size() < 16)
                        res.archive.push_back(res.prog);
                    break;
                } else {
                    res.spill_drops++;
                }
            } else if (new_dag_len <= best_dag_len + 2 && res.archive.size() < 16 && !arch_sigs.count(sig)) {
                vector<Instr> nprog;
                if (td.compile(nprog, 16, rng, &cur_ord, nullptr, 3000) &&
                    (int)nprog.size() <= (int)res.prog.size() + 2 &&
                    verify_prog(L, starts, all_targets, nprog).empty()) {
                    arch_sigs.insert(sig);
                    res.archive.push_back(move(nprog));
                }
            }

            // Macro-SA state transition
            double prog_frac = min(1.0, (now_sec() - t0) / max(0.5, P.lns_sec));
            double phase = fmod(prog_frac * 3.0, 1.0);
            double temp = max(0.05, 1.2 * (1.0 - phase));
            int dE = new_dag_len - cur_dag_len;
            if (dE <= 0 || (new_dag_len <= best_dag_len + 2 &&
                            ((double)(rng() % 10000) / 10000.0) < exp(-(double)dE / temp))) {
                cur_dag = move(td);
                cur_ord = move(out_ord);
                cur_dag.analyse();
                rebuild_clusters();
                stall++;
            } else {
                stall++;
            }
            if (stall >= 12) {
                cur_dag = best_dag;
                cur_ord = best_ord;
                cur_dag.analyse();
                rebuild_clusters();
                stall = 0;
            }
        }
    }
    return res;
}

static int run_macro_mode(const vector<PuzzleIn> &puzzles, const Params &P) {
    auto in_progs = load_progs_file(P.macro_path);
    vector<string> out_lines(puzzles.size());
    vector<string> pool_lines(puzzles.size());
    atomic<int> nextp{0};
    mutex mu;
    long long tot_old = 0, tot_new = 0, tot_dw = 0, tot_cw = 0, tot_sp = 0, tot_it = 0, tot_arch = 0;
    double t0 = now_sec();

    auto worker = [&]() {
        while (true) {
            int k = nextp++;
            if (k >= (int)puzzles.size()) break;
            const auto &pz = puzzles[k];
            auto it = in_progs.find(pz.id);
            if (it == in_progs.end()) continue;
            int olen = (int)it->second.size();
            mt19937_64 rng((u64)P.seed * 1000003ULL + hash<string>()(pz.id));

            MacroResult mr;
            double tp0 = now_sec();
            if (olen >= P.min_len && olen <= P.max_len) {
                mr = macro_optimize(pz.L, pz.s, pz.t, it->second, P, rng);
            } else {
                mr.prog = it->second;
                mr.archive.push_back(it->second);
            }
            string why = verify_prog(pz.L, pz.s, pz.t, mr.prog);
            if (!why.empty()) {
                mr.prog = it->second;
            }
            out_lines[k] = pz.id + " " + to_string(mr.prog.size()) + " " + to_json(mr.prog);
            if (!P.macro_pool_path.empty()) {
                vector<string> vlines;
                for (const auto &ap : mr.archive) {
                    if (verify_prog(pz.L, pz.s, pz.t, ap).empty())
                        vlines.push_back(to_string(ap.size()) + " " + to_json(ap) + "\n");
                }
                string pl = "PUZ " + pz.id + " " + to_string(mr.prog.size()) + " " + to_string(vlines.size()) + "\n";
                for (const auto &vl : vlines) pl += vl;
                pool_lines[k] = move(pl);
            }
            double dt = now_sec() - tp0;

            lock_guard<mutex> g(mu);
            tot_old += olen;
            tot_new += (long long)mr.prog.size();
            tot_dw += mr.dag_wins;
            tot_cw += mr.cmp_wins;
            tot_sp += mr.spill_drops;
            tot_it += mr.iters;
            tot_arch += (long long)mr.archive.size();
            if (mr.cmp_wins > 0 || mr.dag_wins > 0 || P.verbose || puzzles.size() <= 20 || (k + 1) % 20 == 0) {
                fprintf(stderr, "%s: %d -> %zu (it=%d dw=%d cw=%d sp=%d bdel=%+d arch=%zu %.1fs)\n",
                        pz.id.c_str(), olen, mr.prog.size(), mr.iters, mr.dag_wins,
                        mr.cmp_wins, mr.spill_drops, mr.best_delta, mr.archive.size(), dt);
            }
        }
    };

    vector<thread> th;
    int nt = max(1, min(P.threads, (int)puzzles.size()));
    for (int i = 0; i < nt; i++) th.emplace_back(worker);
    for (auto &t : th) t.join();

    if (!P.out_path.empty()) {
        FILE *fo = fopen(P.out_path.c_str(), "w");
        if (!fo) return 1;
        for (const auto &s : out_lines) if (!s.empty()) fprintf(fo, "%s\n", s.c_str());
        fclose(fo);
    } else {
        for (const auto &s : out_lines) if (!s.empty()) printf("%s\n", s.c_str());
    }
    if (!P.macro_pool_path.empty()) {
        FILE *fp = fopen(P.macro_pool_path.c_str(), "w");
        if (fp) {
            for (const auto &s : pool_lines) if (!s.empty()) fputs(s.c_str(), fp);
            fclose(fp);
        }
    }
    fprintf(stderr, "MACRO TOTAL: old=%lld new=%lld (%+lld) iters=%lld dw=%lld cw=%lld spills=%lld arch=%lld time=%.1fs\n",
            tot_old, tot_new, tot_new - tot_old, tot_it, tot_dw, tot_cw, tot_sp, tot_arch, now_sec() - t0);
    return 0;
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
        else if (a == "-multi") P.multi = stoi(nx());
        else if (a == "-beam") P.beam_width = stoi(nx());
        else if (a == "-beam_pool") P.beam_pool_path = nx();
        else if (a == "-la") P.lookahead = stoi(nx());
        else if (a == "-la_weight" || a == "-lw") P.la_weight = stod(nx());
        else if (a == "-lalt") P.la_alts = stoi(nx());
        else if (a == "-lns") P.lns_path = nx();
        else if (a == "-sa") P.sa_path = nx();
        else if (a == "-macro") P.macro_path = nx();
        else if (a == "-macro_pool") P.macro_pool_path = nx();
        else if (a == "-min_len") P.min_len = stoi(nx());
        else if (a == "-max_len") P.max_len = stoi(nx());
        else if (a == "-o") P.out_path = nx();
        else if (a == "-pool") P.pool_path = nx();
        else if (a == "-dag") P.dag_path = nx();
        else if (a == "-pref") P.pref_path = nx();
        else if (a == "-dag_sa") P.dag_sa_iters = stoi(nx());
        else if (a == "-dump_recipes") P.dump_recipes_path = nx();
        else if (a == "-sched-test") P.sched_test_path = nx();
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
        fprintf(stderr, "usage: elf2 PUZZLES.txt [-construct 1] [-multi N] [-lns PROGS.txt] [-sa PROGS.txt] [-pool OUT] [-dag DAG] [-sched-test PROGS] [-lsec S] ...\n");
        return 1;
    }
    if (P.lns_sec < 0) P.lns_sec = ((!P.lns_path.empty() || !P.sa_path.empty() || !P.macro_path.empty()) && !P.construct) ? 15.0 : 0.0;

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

    if (!P.sched_test_path.empty()) return run_sched_test(puzzles, P);
    if (!P.pool_path.empty()) return run_export_pool(puzzles, P);
    if (!P.dag_path.empty()) return run_dag_mode(puzzles, P);
    if (!P.macro_path.empty()) return run_macro_mode(puzzles, P);

    unordered_map<string, vector<Instr>> lns_in;
    if (!P.sa_path.empty()) lns_in = load_progs_file(P.sa_path);
    else if (!P.lns_path.empty()) lns_in = load_progs_file(P.lns_path);

    vector<string> out(puzzles.size());
    vector<string> rec_dump(puzzles.size());
    vector<string> beam_pool_dump(puzzles.size());
    atomic<int> nextp{0};
    mutex mu;
    long long total_new = 0, total_old = 0;
    long long tot_rejs = 0, tot_reje = 0;
    array<u64, 6> tot_mwins{};
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
                int s_acc = 0, e_acc = 0, iters = 0, rejs = 0, reje = 0;
                array<u64, 6> mwins{};

                if (!P.sa_path.empty()) {
                    auto it = lns_in.find(pz.id);
                    if (it == lns_in.end()) continue;
                    old_len = it->second.size();
                    auto sr = sa_optimize(pz.L, pz.s, pz.t, it->second, P, pz.id, rng);
                    prog = move(sr.prog);
                    s_acc = sr.strict_acc; e_acc = sr.eq_acc; iters = sr.iters;
                    rejs = sr.rej_strict; reje = sr.rej_eq; mwins = sr.move_wins;
                    st = sr.stats;
                    if (!P.dump_recipes_path.empty() && !sr.harvested_recipes.empty()) {
                        string rd = pz.id + " " + to_string(sr.harvested_recipes.size()) + "\n";
                        for (const auto &hr : sr.harvested_recipes) {
                            rd += to_string(hr.r) + " ";
                            rd.push_back(hr.op);
                            rd += " " + to_string(hr.a) + " " + to_string(hr.b) + "\n";
                        }
                        rec_dump[k] = move(rd);
                    }
                } else if (P.beam_width > 0) {
                    vector<vector<Instr>> all_bp;
                    vector<Step> brecs;
                    prog = construct_beam_puzzle(pz.L, pz.s, pz.t, P, pz.id, rng, st,
                                                 P.beam_pool_path.empty() ? nullptr : &all_bp,
                                                 P.dump_recipes_path.empty() ? nullptr : &brecs);
                    old_len = prog.size();
                    if (!P.beam_pool_path.empty() && !all_bp.empty()) {
                        string bp = "PUZ " + pz.id + " " + to_string(all_bp[0].size()) + " " + to_string(all_bp.size()) + "\n";
                        for (const auto &pr : all_bp) {
                            bp += to_string(pr.size()) + " " + to_json(pr) + "\n";
                        }
                        beam_pool_dump[k] = move(bp);
                    }
                    if (!P.dump_recipes_path.empty() && !brecs.empty()) {
                        string rd = pz.id + " " + to_string(brecs.size()) + "\n";
                        for (const auto &hr : brecs) {
                            rd += to_string(hr.r) + " ";
                            rd.push_back(hr.op);
                            rd += " " + to_string(hr.a) + " " + to_string(hr.b) + "\n";
                        }
                        rec_dump[k] = move(rd);
                    }
                } else if (P.construct || P.lns_path.empty()) {
                    // Multi-start constructive portfolio
                    int runs = max(1, P.multi);
                    for (int r = 0; r < runs; r++) {
                        Params rp = P;
                        if (r > 0) {
                            static const double lws[] = {0.35, 0.25, 0.45, 0.30};
                            static const double als[] = {0.16, 0.13, 0.17, 0.15};
                            static const int nls[] = {0, 20, -16, 8};
                            rp.la_weight = lws[r % 4];
                            rp.alpha = als[r % 4];
                            rp.n_leaves = max(48, P.n_leaves + nls[r % 4]);
                        }
                        mt19937_64 rrng((u64)(P.seed + r * 7919) * 1000003ULL + hash<string>()(pz.id));
                        auto cand = construct_puzzle(pz.L, pz.s, pz.t, rp, pz.id, rrng, st, r == 0 ? 0.0 : 0.25);
                        if (prog.empty() || cand.size() < prog.size()) prog = move(cand);
                    }
                    old_len = prog.size();
                    if (P.lns_sec > 0) {
                        auto lr = sa_optimize(pz.L, pz.s, pz.t, prog, P, pz.id, rng);
                        prog = move(lr.prog);
                        s_acc = lr.strict_acc; e_acc = lr.eq_acc; iters = lr.iters;
                        rejs = lr.rej_strict; reje = lr.rej_eq; mwins = lr.move_wins;
                        for (int e = 0; e <= 5; e++) { st.hits[e] += lr.stats.hits[e]; st.wins[e] += lr.stats.wins[e]; st.sec[e] += lr.stats.sec[e]; }
                    }
                } else {
                    auto it = lns_in.find(pz.id);
                    if (it == lns_in.end()) continue;
                    old_len = it->second.size();
                    auto lr = lns_optimize(pz.L, pz.s, pz.t, it->second, P, pz.id, rng);
                    prog = move(lr.prog);
                    s_acc = lr.strict_acc; e_acc = lr.eq_acc; iters = lr.iters;
                    rejs = lr.rej_strict; reje = lr.rej_eq; mwins = lr.move_wins;
                    st = lr.stats;
                }

                string why = verify_prog(pz.L, pz.s, pz.t, prog);
                if (!why.empty()) throw runtime_error("verify failed: " + why);
                out[k] = pz.id + " " + to_string(prog.size()) + " " + to_json(prog);

                lock_guard<mutex> g(mu);
                total_new += (long long)prog.size();
                total_old += (long long)old_len;
                tot_rejs += rejs;
                tot_reje += reje;
                tot_stats.calls += st.calls;
                for (int e = 0; e <= 5; e++) {
                    tot_stats.hits[e] += st.hits[e];
                    tot_stats.wins[e] += st.wins[e];
                    tot_stats.sec[e] += st.sec[e];
                }
                for (int m = 0; m < 6; m++) tot_mwins[m] += mwins[m];
                if (P.verbose || !P.lns_path.empty() || !P.sa_path.empty()) {
                    fprintf(stderr,
                            "%s: %zu -> %zu (%+lld) | sacc=%d eq=%d it=%d rejp=%d/%d | mv=[%llu,%llu,%llu,%llu,%llu,%llu] hits=[%llu,%llu,%llu,%llu,%llu] wins=[%llu,%llu,%llu,%llu,%llu] t=[%.1f|%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                            pz.id.c_str(), old_len, prog.size(),
                            (long long)prog.size() - (long long)old_len,
                            s_acc, e_acc, iters, rejs, reje,
                            (unsigned long long)mwins[0], (unsigned long long)mwins[1], (unsigned long long)mwins[2], (unsigned long long)mwins[3], (unsigned long long)mwins[4], (unsigned long long)mwins[5],
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

    if (!P.dump_recipes_path.empty()) {
        FILE *fr = fopen(P.dump_recipes_path.c_str(), "w");
        if (fr) {
            for (const auto &rs : rec_dump) if (!rs.empty()) fwrite(rs.data(), 1, rs.size(), fr);
            fclose(fr);
        }
    }
    if (!P.beam_pool_path.empty()) {
        FILE *fb = fopen(P.beam_pool_path.c_str(), "w");
        if (fb) {
            for (const auto &bs : beam_pool_dump) if (!bs.empty()) fwrite(bs.data(), 1, bs.size(), fb);
            fclose(fb);
        }
    }
    if (!P.out_path.empty()) {
        FILE *fo = fopen(P.out_path.c_str(), "w");
        if (!fo) { fprintf(stderr, "cannot open %s\n", P.out_path.c_str()); return 1; }
        for (const auto &s : out) if (!s.empty()) fprintf(fo, "%s\n", s.c_str());
        fclose(fo);
    } else {
        for (const auto &s : out) if (!s.empty()) printf("%s\n", s.c_str());
    }
    fprintf(stderr,
            "TOTAL: old=%lld new=%lld (%+lld) time=%.1fs | rejp=%lld/%lld mv=[%llu,%llu,%llu,%llu,%llu,%llu] calls=%llu hits=[%llu,%llu,%llu,%llu,%llu] wins=[%llu,%llu,%llu,%llu,%llu] t=[%.1f|%.1f,%.1f,%.1f,%.1f,%.1f]\n",
            total_old, total_new, total_new - total_old, now_sec() - t0,
            tot_rejs, tot_reje,
            (unsigned long long)tot_mwins[0], (unsigned long long)tot_mwins[1], (unsigned long long)tot_mwins[2], (unsigned long long)tot_mwins[3], (unsigned long long)tot_mwins[4], (unsigned long long)tot_mwins[5],
            (unsigned long long)tot_stats.calls,
            (unsigned long long)tot_stats.hits[1], (unsigned long long)tot_stats.hits[2],
            (unsigned long long)tot_stats.hits[3], (unsigned long long)tot_stats.hits[4], (unsigned long long)tot_stats.hits[5],
            (unsigned long long)tot_stats.wins[1], (unsigned long long)tot_stats.wins[2],
            (unsigned long long)tot_stats.wins[3], (unsigned long long)tot_stats.wins[4], (unsigned long long)tot_stats.wins[5],
            tot_stats.sec[0], tot_stats.sec[1], tot_stats.sec[2], tot_stats.sec[3], tot_stats.sec[4], tot_stats.sec[5]);
    return 0;
}
