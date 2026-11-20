// bases: discover hidden "base" values shared by several targets.
// For every pair of targets and small offsets da, db in [-D, D], g = gcd(t_a - da, t_b - db).
// A value g >= GMIN with g^2 <= 4*min(...) that divides offset versions of many targets is a
// candidate base. Candidates are ranked by the number of distinct targets they support; multiples
// of a stronger candidate are dropped.
// Input: puzzle lines (id L ns s.. nt t..). Output: id k b1 ... bk
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace std;
typedef uint64_t u64;

static inline u64 bgcd(u64 a, u64 b) {
    if (!a) return b; if (!b) return a;
    int sh = __builtin_ctzll(a | b);
    a >>= __builtin_ctzll(a);
    do { b >>= __builtin_ctzll(b); if (a > b) { u64 t = a; a = b; b = t; } b -= a; } while (b);
    return a << sh;
}

int main(int argc, char **argv) {
    string path = argv[1];
    int D = argc > 2 ? atoi(argv[2]) : 100;
    int minsup = argc > 3 ? atoi(argv[3]) : 4;
    int maxk = argc > 4 ? atoi(argv[4]) : 6;
    u64 GMIN = argc > 5 ? strtoull(argv[5], 0, 10) : 1000;
    int threads = argc > 6 ? atoi(argv[6]) : 8;
    struct Pz { string id; vector<u64> t; };
    vector<Pz> pz;
    { ifstream f(path); string line;
      while (getline(f, line)) { istringstream is(line); Pz p; u64 L; int ns, nt; is >> p.id >> L >> ns; for (int i = 0; i < ns; i++) { u64 x; is >> x; } is >> nt; p.t.resize(nt); for (auto &x : p.t) is >> x; if (!p.id.empty()) pz.push_back(p); } }
    vector<string> out(pz.size());
    atomic<int> nx{0};
    auto work = [&]() {
        while (true) {
            int k = nx++; if (k >= (int)pz.size()) break;
            auto &T = pz[k].t; int n = T.size();
            unordered_map<u64, u64> sup;  // g -> bitmask of supporting targets
            unordered_map<u64, int> npair;
            for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++)
                for (int da = -D; da <= D; da++) {
                    if ((long long)T[a] - da <= 0) continue;
                    u64 x = T[a] - da;
                    for (int db = -D; db <= D; db++) {
                        if ((long long)T[b] - db <= 0) continue;
                        u64 y = T[b] - db;
                        u64 g = bgcd(x, y);
                        if (g < GMIN) continue;
                        u64 m = min(x, y);
                        if ((__int128)g * g > (__int128)m * 4) continue;
                        sup[g] |= (1ULL << a) | (1ULL << b);
                        npair[g]++;
                    }
                }
            vector<pair<int, u64>> c;
            double pairs = n * (n - 1) / 2.0, w = 2.0 * D + 1;
            for (auto &kv : sup) {
                int s = __builtin_popcountll(kv.second);
                double expct = pairs * w * w / ((double)kv.first * (double)kv.first);
                // binomial tail C(n,s) p^s with p = (2D+1)/g, corrected for ~g candidate values of this size
                double p = w / (double)kv.first, lp = lgamma(n + 1.0) - lgamma(s + 1.0) - lgamma(n - s + 1.0) + s * log(p) + log((double)kv.first);
                (void)expct;
                if (s >= minsup && lp < log(0.01)) c.push_back({s, kv.first});
            }
            sort(c.begin(), c.end(), [](auto &x, auto &y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
            vector<u64> keep;
            for (auto &[s, g] : c) {
                bool redundant = false;
                for (u64 h : keep) if (g % h == 0 || h % g == 0) { redundant = true; break; }
                if (!redundant) keep.push_back(g);
                if ((int)keep.size() >= maxk) break;
            }
            string o = pz[k].id + " " + to_string(keep.size());
            for (u64 g : keep) o += " " + to_string(g) + ":" + to_string(__builtin_popcountll(sup[g]));
            out[k] = o;
        }
    };
    vector<thread> th; for (int i = 0; i < threads; i++) th.emplace_back(work); for (auto &t : th) t.join();
    for (auto &s : out) printf("%s\n", s.c_str());
}
