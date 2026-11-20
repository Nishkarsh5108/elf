// mitm_probe: given register snapshots and a target, find the minimum tree cost (<= 7)
// of building the target from the snapshot values alone.
//   S0 = registers, S1 = one op over S0, S2 = chains of two ops (exact hash),
//   S3 = (S2 op S0) U (S1 op S1) kept in a Bloom filter, hits verified exactly.
// Input (stdin), one query per line:  limit target n v1 ... vn
// Output: one line per query: target bits mindepth (or -1 if > maxdepth)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
using namespace std;
typedef uint64_t u64;
typedef unsigned __int128 u128;

static u64 L;  // value limit

static inline u64 mix(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

struct HashSet {  // open addressing, value -> cost (uint8)
    vector<u64> keys; vector<uint8_t> cost; u64 mask; size_t n = 0;
    void init(int bits) { keys.assign(1ULL << bits, 0); cost.assign(1ULL << bits, 0); mask = (1ULL << bits) - 1; n = 0; }
    inline int get(u64 v) const {
        u64 i = mix(v) & mask;
        while (keys[i]) { if (keys[i] == v) return cost[i]; i = (i + 1) & mask; }
        return -1;
    }
    inline bool insert(u64 v, int c) {  // returns true if new
        u64 i = mix(v) & mask;
        while (keys[i]) { if (keys[i] == v) return false; i = (i + 1) & mask; }
        keys[i] = v; cost[i] = (uint8_t)c; n++; return true;
    }
};

struct Bloom {
    vector<u64> bits; u64 mask;
    void init(int lg) { bits.assign((1ULL << lg) / 64, 0); mask = (1ULL << lg) - 1; }
    inline void add(u64 v) {
        u64 h = mix(v), h2 = mix(v ^ 0x9e3779b97f4a7c15ULL);
        u64 a = h & mask, b = (h >> 32 ^ h2) & mask, c = h2 & mask;
        bits[a >> 6] |= 1ULL << (a & 63); bits[b >> 6] |= 1ULL << (b & 63); bits[c >> 6] |= 1ULL << (c & 63);
    }
    inline bool has(u64 v) const {
        u64 h = mix(v), h2 = mix(v ^ 0x9e3779b97f4a7c15ULL);
        u64 a = h & mask, b = (h >> 32 ^ h2) & mask, c = h2 & mask;
        return (bits[a >> 6] >> (a & 63) & 1) && (bits[b >> 6] >> (b & 63) & 1) && (bits[c >> 6] >> (c & 63) & 1);
    }
};

// all results of a op b (both orders for - and /), within (0, L]
static inline int combos(u64 a, u64 b, u64 *out) {
    int n = 0;
    u64 s = a + b; if (s <= L) out[n++] = s;
    if (a > b) out[n++] = a - b; else if (b > a) out[n++] = b - a;
    if ((u128)a * b <= L) out[n++] = a * b;
    if (a % b == 0) out[n++] = a / b;
    if (b % a == 0 && a != b) out[n++] = b / a;
    return n;
}
// all A with A op B == t or B op A == t
static inline int inverses(u64 t, u64 B, u64 *out) {
    int n = 0;
    if (t > B) out[n++] = t - B;
    if (t + B <= L) out[n++] = t + B;
    if (B > t) out[n++] = B - t;
    if (t % B == 0) out[n++] = t / B;
    if ((u128)t * B <= L) out[n++] = t * B;
    if (B % t == 0) out[n++] = B / t;
    return n;
}

static vector<u64> S0, S1, S2;
static HashSet H;   // S0..S2 with costs
static Bloom BF;    // S3

static bool inS3exact(u64 A) {  // verify A in S3 (tree cost exactly <= 3)
    u64 inv[8];
    for (u64 c : S0) { int k = inverses(A, c, inv); for (int i = 0; i < k; i++) { int q = H.get(inv[i]); if (q >= 0 && q <= 2) return true; } }
    for (u64 y : S1) { int k = inverses(A, y, inv); for (int i = 0; i < k; i++) { int q = H.get(inv[i]); if (q >= 0 && q <= 1) return true; } }
    // x op x forms: A = x+x or x*x with x in S2
    if (A % 2 == 0 && H.get(A / 2) >= 0) return true;
    return false;
}

int main(int argc, char **argv) {
    int maxd = argc > 1 ? atoi(argv[1]) : 6;
    int bloomlg = argc > 2 ? atoi(argv[2]) : 33;
    char buf[1 << 16];
    BF.init(bloomlg);
    while (true) {
        unsigned long long lim, t; int n;
        if (scanf("%llu %llu %d", &lim, &t, &n) != 3) break;
        L = lim;
        S0.clear(); S1.clear(); S2.clear();
        H.init(23);
        for (int i = 0; i < n; i++) { unsigned long long v; scanf("%llu", &v); if (H.insert(v, 0)) S0.push_back(v); }
        auto t0 = chrono::steady_clock::now();
        u64 out[8];
        for (size_t i = 0; i < S0.size(); i++)
            for (size_t j = i; j < S0.size(); j++) {
                int k = combos(S0[i], S0[j], out);
                for (int q = 0; q < k; q++) if (H.insert(out[q], 1)) S1.push_back(out[q]);
            }
        for (u64 x : S1) {
            for (size_t j = 0; j <= S0.size(); j++) {
                u64 c = j < S0.size() ? S0[j] : x;
                int k = combos(x, c, out);
                for (int q = 0; q < k; q++) if (H.insert(out[q], 2)) S2.push_back(out[q]);
            }
        }
        int best = H.get(t);
        long long s3n = 0;
        if (best < 0 && maxd >= 4) {
            // build S3 bloom
            fill(BF.bits.begin(), BF.bits.end(), 0);
            for (u64 x : S2)
                for (size_t j = 0; j <= S0.size(); j++) {
                    u64 c = j < S0.size() ? S0[j] : x;
                    int k = combos(x, c, out);
                    for (int q = 0; q < k; q++) { BF.add(out[q]); s3n++; }
                }
            for (size_t i = 0; i < S1.size(); i++)
                for (size_t j = i; j < S1.size(); j++) {
                    int k = combos(S1[i], S1[j], out);
                    for (int q = 0; q < k; q++) { BF.add(out[q]); s3n++; }
                }
        }
        u64 inv[8];
        auto upd = [&](int c) { if (best < 0 || c < best) best = c; };
        if (best < 0) {
            // depth 3..: final op over (A,B) with B enumerated from S0,S1,S2 and A looked up
            // B in S0 or S1 -> A in S<=2 (exact) : depth <= 4
            for (int pass = 0; pass < 2; pass++) {
                const vector<u64> &BS = pass == 0 ? S0 : S1;
                for (u64 B : BS) {
                    int k = inverses(t, B, inv);
                    for (int i = 0; i < k; i++) { int q = H.get(inv[i]); if (q >= 0) upd(q + pass + 1); }
                }
            }
            if (best < 0 || best > 4) {  // A in S3 with B in S0 (depth 4), S1 (5), S2 (6)
                if (maxd >= 4) {
                    for (u64 B : S0) { int k = inverses(t, B, inv); for (int i = 0; i < k; i++) if (BF.has(inv[i]) && inS3exact(inv[i])) upd(4); }
                    if (BF.has(t) && inS3exact(t)) upd(3);
                }
            }
            if ((best < 0 || best > 5) && maxd >= 5) {
                for (u64 B : S2) { int k = inverses(t, B, inv); for (int i = 0; i < k; i++) { int q = H.get(inv[i]); if (q >= 0) upd(q + 3); } }
                for (u64 B : S1) { int k = inverses(t, B, inv); for (int i = 0; i < k; i++) if (BF.has(inv[i]) && inS3exact(inv[i])) upd(5); }
            }
            if ((best < 0 || best > 6) && maxd >= 6) {
                for (u64 B : S2) { int k = inverses(t, B, inv); for (int i = 0; i < k; i++) if (BF.has(inv[i]) && inS3exact(inv[i])) { upd(6); break; } }
            }
            if ((best < 0 || best > 7) && maxd >= 7) {
                // enumerate S3 on the fly as B, A in S3 via bloom
                bool found = false;
                for (u64 x : S2) {
                    if (found) break;
                    for (size_t j = 0; j <= S0.size() && !found; j++) {
                        u64 c = j < S0.size() ? S0[j] : x;
                        u64 bb[8]; int kb = combos(x, c, bb);
                        for (int qb = 0; qb < kb && !found; qb++) {
                            int k = inverses(t, bb[qb], inv);
                            for (int i = 0; i < k; i++) if (BF.has(inv[i]) && inS3exact(inv[i])) { found = true; break; }
                        }
                    }
                }
                for (size_t i2 = 0; i2 < S1.size() && !found; i2++)
                    for (size_t j2 = i2; j2 < S1.size() && !found; j2++) {
                        u64 bb[8]; int kb = combos(S1[i2], S1[j2], bb);
                        for (int qb = 0; qb < kb && !found; qb++) {
                            int k = inverses(t, bb[qb], inv);
                            for (int i = 0; i < k; i++) if (BF.has(inv[i]) && inS3exact(inv[i])) { found = true; break; }
                        }
                    }
                if (found) upd(7);
            }
        }
        double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
        int bits = 64 - __builtin_clzll(t);
        printf("%llu %d %d |S1|=%zu |S2|=%zu S3ins=%lld %.0fms\n", (unsigned long long)t, bits, best, S1.size(), S2.size(), s3n, ms);
        fflush(stdout);
    }
}
