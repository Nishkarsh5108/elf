// elf3.cpp — Free-Order Bitset Knockout, Quartet Bridge Closure, Neutral Backbone Walk, & Multi-Donor Crossover
// Build: .venv/Scripts/python.exe -m ziglang c++ -O3 -march=native -std=c++20 cpp/elf3.cpp -o elf3.exe
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
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;
using i64 = int64_t;
using u128 = unsigned __int128;

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
            return "bad register";
        if (!def[in.i] || !def[in.j]) return "undefined source";
        u64 r = 0;
        if (!apply_op(in.op, reg[in.i], reg[in.j], L, r)) return "illegal op";
        reg[in.d] = r;
        def[in.d] = true;
        seen.insert(r);
    }
    for (u64 t : targets) if (!seen.count(t)) return "missing target";
    if (to_json(prog).size() > 131072) return "json > 131072";
    return "";
}

struct Puzzle {
    string id;
    u64 L = 0;
    vector<u64> s, t;
};

static vector<Puzzle> load_puzzles(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    vector<Puzzle> out;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        Puzzle p;
        int ns, nt;
        ss >> p.id >> p.L >> ns;
        p.s.resize(ns);
        for (int i = 0; i < ns; i++) ss >> p.s[i];
        ss >> nt;
        p.t.resize(nt);
        for (int i = 0; i < nt; i++) ss >> p.t[i];
        out.push_back(move(p));
    }
    return out;
}

static vector<Instr> parse_prog_json(const string &js) {
    vector<Instr> out;
    size_t i = 0, n = js.size();
    while (i < n) {
        if (js[i] == '"' && i + 2 < n && js[i + 2] == '"') {
            char op = js[i + 1];
            i += 3;
            int nums[3]{};
            for (int k = 0; k < 3; k++) {
                while (i < n && (js[i] < '0' || js[i] > '9')) i++;
                int v = 0;
                while (i < n && js[i] >= '0' && js[i] <= '9') { v = v * 10 + (js[i] - '0'); i++; }
                nums[k] = v;
            }
            out.push_back({op, nums[0], nums[1], nums[2]});
        } else {
            i++;
        }
    }
    return out;
}

static unordered_map<string, vector<vector<Instr>>> load_donors(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    unordered_map<string, vector<vector<Instr>>> out;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (line.rfind("PUZ ", 0) == 0) {
            istringstream ss(line);
            string tag, pid;
            int blen = 0, nd = 0;
            ss >> tag >> pid >> blen >> nd;
            auto &vec = out[pid];
            vec.reserve(nd);
            for (int k = 0; k < nd; k++) {
                string dline;
                if (!getline(in, dline)) break;
                size_t sp = dline.find(' ');
                if (sp != string::npos) vec.push_back(parse_prog_json(dline.substr(sp + 1)));
            }
        } else {
            // Fallback single-program txt format (`id len json`)
            size_t p1 = line.find(' ');
            if (p1 == string::npos) continue;
            size_t p2 = line.find(' ', p1 + 1);
            if (p2 == string::npos) continue;
            string pid = line.substr(0, p1);
            out[pid].push_back(parse_prog_json(line.substr(p2 + 1)));
        }
    }
    return out;
}

// ---------------------------------------------------------------- DAG & Peak-Liveness 48-Reg Compiler
struct DagNode { u64 v; char op; int a, b; };

struct Dag {
    u64 L = 0;
    vector<u64> starts;
    vector<u64> targets; // non-start targets
    vector<DagNode> nodes;
    int ks = 0;
    unordered_map<u64, int> val2idx;
    vector<int> tnode;

    void init(u64 limit, const vector<u64> &st, const vector<u64> &tg) {
        L = limit;
        starts = st;
        unordered_set<u64> sset(st.begin(), st.end());
        targets.clear();
        for (u64 t : tg) if (!sset.count(t)) targets.push_back(t);
    }

    int len() const { return (int)nodes.size() - ks; }

    bool from_defs(const unordered_map<u64, Step> &defs, const vector<u64> &pref_order = {}) {
        nodes.clear();
        val2idx.clear();
        for (u64 s : starts) {
            if (val2idx.count(s)) continue;
            val2idx[s] = (int)nodes.size();
            nodes.push_back({s, 0, -1, -1});
        }
        ks = (int)nodes.size();
        unordered_set<u64> needed;
        function<bool(u64)> mark = [&](u64 v) -> bool {
            if (val2idx.count(v)) return true;
            if (!needed.insert(v).second) return true;
            auto it = defs.find(v);
            if (it == defs.end()) return false;
            return mark(it->second.a) && mark(it->second.b);
        };
        for (u64 t : targets) if (!mark(t)) return false;

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
            if (d == 0) ready.push_back({prio.count(v) ? prio[v] : 1000000, v});
        }
        make_heap(ready.begin(), ready.end(), greater<PQItem>());
        size_t cnt = 0;
        while (!ready.empty()) {
            pop_heap(ready.begin(), ready.end(), greater<PQItem>());
            u64 v = ready.back().second;
            ready.pop_back();
            const Step &s = defs.at(v);
            val2idx[v] = (int)nodes.size();
            nodes.push_back({v, s.op, val2idx.at(s.a), val2idx.at(s.b)});
            cnt++;
            for (u64 w : adj[v]) {
                if (--indeg[w] == 0) {
                    ready.push_back({prio.count(w) ? prio[w] : 1000000 + (int)cnt, w});
                    push_heap(ready.begin(), ready.end(), greater<PQItem>());
                }
            }
        }
        if (cnt != needed.size()) return false;
        tnode.resize(targets.size());
        for (size_t i = 0; i < targets.size(); i++) tnode[i] = val2idx.at(targets[i]);
        return true;
    }

    bool from_prog(const vector<Instr> &prog, vector<u64> *ord_out = nullptr, unordered_map<u64, Step> *defs_out = nullptr) {
        array<u64, 48> reg{};
        array<bool, 48> def{};
        for (size_t k = 0; k < starts.size(); k++) { reg[k] = starts[k]; def[k] = true; }
        unordered_map<u64, Step> defs;
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
        if (ord_out) *ord_out = ord;
        if (defs_out) *defs_out = defs;
        return from_defs(defs, ord);
    }

    unordered_map<u64, Step> to_defs() const {
        unordered_map<u64, Step> d;
        for (int u = ks; u < (int)nodes.size(); u++) {
            d[nodes[u].v] = {nodes[u].op, nodes[nodes[u].a].v, nodes[nodes[u].b].v, nodes[u].v};
        }
        return d;
    }

    vector<u64> node_order() const {
        vector<u64> ord;
        ord.reserve(nodes.size() - ks);
        for (int u = ks; u < (int)nodes.size(); u++) ord.push_back(nodes[u].v);
        return ord;
    }

    u64 hash_dag() const {
        vector<u64> hs;
        hs.reserve(nodes.size() - ks);
        for (int u = ks; u < (int)nodes.size(); u++) {
            u64 va = nodes[nodes[u].a].v, vb = nodes[nodes[u].b].v;
            if ((nodes[u].op == '+' || nodes[u].op == '*') && va > vb) swap(va, vb);
            u64 h = mix64(nodes[u].v ^ ((u64)nodes[u].op << 56));
            h ^= mix64(va + 0x9e3779b97f4a7c15ULL);
            h ^= mix64(vb + 0xbf58476d1ce4e5b9ULL);
            hs.push_back(h);
        }
        sort(hs.begin(), hs.end());
        u64 tot = 0;
        for (u64 h : hs) tot = mix64(tot ^ h);
        return tot;
    }

    struct PeakInfo { int max_peak; int over48; int worst_idx; };

    PeakInfo eval_order(const vector<int> &seq, vector<int> *lu_out = nullptr) const {
        int n = (int)nodes.size(), M = (int)seq.size();
        vector<int> lu_local;
        vector<int> &lu = lu_out ? *lu_out : lu_local;
        lu.assign(n, -1);
        for (int idx = 0; idx < M; idx++) {
            int u = seq[idx];
            lu[nodes[u].a] = max(lu[nodes[u].a], idx);
            lu[nodes[u].b] = max(lu[nodes[u].b], idx);
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
        for (int s = 0; s < ks; s++) { reg_of[s] = s; occ[s] = (rem[s] > 0); }
        vector<Instr> prog;
        prog.reserve(seq.size());
        for (int u : seq) {
            int a = nodes[u].a, b = nodes[u].b;
            int ra = reg_of[a], rb = reg_of[b];
            if (ra < 0 || rb < 0) return false;
            rem[a]--; if (b != a) rem[b]--;
            if (rem[a] == 0) occ[ra] = false;
            if (b != a && rem[b] == 0) occ[rb] = false;
            int rd = -1;
            if (!occ[ra]) rd = ra;
            else if (!occ[rb]) rd = rb;
            else { for (int r = 0; r < 48; r++) if (!occ[r]) { rd = r; break; } }
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
        int n = (int)nodes.size(), M = (int)seq.size();
        if (M <= 1) return eval_order(seq).max_peak <= 48;
        vector<vector<int>> deps(n);
        for (int u = ks; u < n; u++) {
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
        }
        vector<int> lu;
        PeakInfo cur = eval_order(seq, &lu);
        if (cur.max_peak <= 48) return true;
        vector<int> best_seq = seq;
        PeakInfo best_info = cur;
        vector<int> pos(n, -1);
        for (int i = 0; i < M; i++) pos[seq[i]] = i;
        auto score_pi = [](const PeakInfo &pi) { return pi.max_peak * 100.0 + pi.over48 * 2.5; };
        double curE = score_pi(cur);
        vector<int> cand_u, trial;
        for (int it = 0; it < max_iters; it++) {
            if (best_info.max_peak <= 48) break;
            int idx = -1;
            if (rng() % 100 < 75) {
                cand_u.clear();
                int w = cur.worst_idx;
                for (int i = 0; i < M; i++) {
                    int u = seq[i];
                    if ((i <= w && lu[u] >= w) || abs(i - w) <= 3 || (lu[u] >= w && lu[u] <= w + 6))
                        cand_u.push_back(i);
                }
                if (!cand_u.empty()) idx = cand_u[rng() % cand_u.size()];
            }
            if (idx < 0) idx = (int)(rng() % M);
            int u = seq[idx];
            int lo = 0;
            if (nodes[u].a >= ks) lo = max(lo, pos[nodes[u].a] + 1);
            if (nodes[u].b >= ks) lo = max(lo, pos[nodes[u].b] + 1);
            int hi = M - 1;
            for (int ch : deps[u]) hi = min(hi, pos[ch] - 1);
            if (lo >= hi) continue;
            int opts[4], nopt = 0;
            auto add_opt = [&](int p) {
                if (p == idx || p < lo || p > hi) return;
                for (int k = 0; k < nopt; k++) if (opts[k] == p) return;
                opts[nopt++] = p;
            };
            add_opt(hi); add_opt(lo);
            add_opt(max(lo, min(hi, cur.worst_idx + (cur.worst_idx < idx ? -1 : 1))));
            add_opt(lo + (int)(rng() % (hi - lo + 1)));
            if (nopt == 0) continue;
            int chosen_p = -1;
            PeakInfo chosen_pi{999, 999999, 0};
            double chosen_E = 1e18;
            for (int k = 0; k < nopt; k++) {
                int np = opts[k];
                trial = seq;
                trial.erase(trial.begin() + idx);
                trial.insert(trial.begin() + np, u);
                PeakInfo pi = eval_order(trial);
                double E = score_pi(pi);
                if (E < chosen_E || (E == chosen_E && np > chosen_p)) {
                    chosen_E = E; chosen_p = np; chosen_pi = pi;
                }
            }
            double temp = max(0.05, 4.0 * (1.0 - (double)it / (double)max_iters));
            double dE = chosen_E - curE;
            if (dE <= 0.0 || ((double)(rng() % 10000) / 10000.0) < exp(-dE / temp)) {
                seq.erase(seq.begin() + idx);
                seq.insert(seq.begin() + chosen_p, u);
                for (int k = min(idx, chosen_p); k <= max(idx, chosen_p); k++) pos[seq[k]] = k;
                cur = eval_order(seq, &lu);
                curE = score_pi(cur);
                if (cur.max_peak < best_info.max_peak ||
                    (cur.max_peak == best_info.max_peak && cur.over48 < best_info.over48)) {
                    best_info = cur;
                    best_seq = seq;
                }
            }
        }
        seq = move(best_seq);
        return best_info.max_peak <= 48;
    }

    vector<int> build_pref_order_with_insert(const vector<u64> &kept_ord) const {
        int n = (int)nodes.size(), M = n - ks;
        if (M <= 0) return {};
        vector<vector<int>> deps(n);
        for (int u = ks; u < n; u++) {
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
        }
        vector<int> prio(n, -1);
        for (size_t i = 0; i < kept_ord.size(); i++) {
            auto it = val2idx.find(kept_ord[i]);
            if (it != val2idx.end() && it->second >= ks && prio[it->second] < 0)
                prio[it->second] = (int)i;
        }
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
        vector<int> seq; seq.reserve(M);
        vector<char> in_seq(n, 0);
        while (!ready_k.empty()) {
            pop_heap(ready_k.begin(), ready_k.end(), greater<pair<int, int>>());
            int u = ready_k.back().second;
            ready_k.pop_back();
            seq.push_back(u); in_seq[u] = 1;
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
                vector<int> eff_prio(n, M + n);
                for (int x = ks; x < n; x++) if (prio[x] >= 0) eff_prio[x] = prio[x];
                for (int x = n - 1; x >= ks; --x)
                    for (int ch : deps[x]) eff_prio[x] = min(eff_prio[x], eff_prio[ch]);
                vector<int> indeg_all(n, 0);
                for (int x = ks; x < n; x++) {
                    if (nodes[x].a >= ks) indeg_all[x]++;
                    if (nodes[x].b >= ks && nodes[x].b != nodes[x].a) indeg_all[x]++;
                }
                using Key = tuple<int, int, int, int>;
                priority_queue<Key, vector<Key>, greater<Key>> pq;
                for (int x = ks; x < n; x++)
                    if (indeg_all[x] == 0) pq.emplace(eff_prio[x], prio[x] < 0 ? 1 : 0, prio[x] >= 0 ? prio[x] : x, x);
                seq.clear();
                while (!pq.empty()) {
                    auto [ep, isn, sp, x] = pq.top(); pq.pop();
                    seq.push_back(x);
                    for (int ch : deps[x]) if (--indeg_all[ch] == 0)
                        pq.emplace(eff_prio[ch], prio[ch] < 0 ? 1 : 0, prio[ch] >= 0 ? prio[ch] : ch, ch);
                }
                break;
            }
            if (lo == hi) { seq.insert(seq.begin() + lo, u); continue; }
            vector<int> cands;
            auto add_c = [&](int p) {
                if (p < lo || p > hi) return;
                for (int x : cands) if (x == p) return;
                cands.push_back(p);
            };
            if (hi - lo <= 24) { for (int p = lo; p <= hi; p++) cands.push_back(p); }
            else {
                for (int d = 0; d <= 3; d++) { add_c(lo + d); add_c(hi - d); }
                for (int k = 1; k < 20; k++) add_c(lo + (int)((long long)(hi - lo) * k / 20));
            }
            int best_p = hi; PeakInfo best_pi{999, 999999, 0};
            for (int p : cands) {
                trial = seq; trial.insert(trial.begin() + p, u);
                PeakInfo pi = eval_order(trial);
                if (pi.max_peak < best_pi.max_peak ||
                    (pi.max_peak == best_pi.max_peak && (pi.over48 < best_pi.over48 || (pi.over48 == best_pi.over48 && p > best_p)))) {
                    best_pi = pi; best_p = p;
                }
            }
            seq.insert(seq.begin() + best_p, u);
        }
        return seq;
    }

    bool schedule_min_live(vector<Instr> &prog_out, int restarts, mt19937_64 &rng) const {
        int n = (int)nodes.size();
        if (n - ks > 4096) return false;
        vector<int> orig_users(n, 0), orig_indeg(n, 0);
        vector<vector<int>> deps(n);
        for (int u = ks; u < n; u++) {
            orig_users[nodes[u].a]++;
            if (nodes[u].b != nodes[u].a) orig_users[nodes[u].b]++;
            if (nodes[u].a >= ks) orig_indeg[u]++;
            if (nodes[u].b >= ks && nodes[u].b != nodes[u].a) orig_indeg[u]++;
            deps[nodes[u].a].push_back(u);
            if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
        }
        vector<int> ready0;
        for (int u = ks; u < n; u++) if (orig_indeg[u] == 0) ready0.push_back(u);
        int live0 = 0;
        for (int s = 0; s < ks; s++) if (orig_users[s] > 0) live0++;
        for (int rep = 0; rep < restarts; rep++) {
            vector<int> rem = orig_users, indeg = orig_indeg, ready = ready0, seq;
            seq.reserve(n - ks);
            int live = live0, max_peak = live;
            bool ok = true;
            while (!ready.empty()) {
                int best_pos = -1; double best_sc = 1e18;
                for (int idx = 0; idx < (int)ready.size(); idx++) {
                    int u = ready[idx], a = nodes[u].a, b = nodes[u].b;
                    int freed = (rem[a] == 1 ? 1 : 0) + (b != a && rem[b] == 1 ? 1 : 0);
                    int peak = live + (freed == 0 ? 1 : 0);
                    int delta = (rem[u] > 0 ? 1 : 0) - freed;
                    double sc = (peak > 48 ? 1e6 : 0.0) + delta * 100.0 + (rem[u] > 0 ? 15.0 : 0.0) - freed * 4.0
                                + 0.2 * (rem[a] + (b != a ? rem[b] : 0))
                                + (rep > 0 ? 2.5 * ((double)(rng() % 1000) / 1000.0) : 0.0);
                    if (sc < best_sc) { best_sc = sc; best_pos = idx; }
                }
                int u = ready[best_pos]; ready[best_pos] = ready.back(); ready.pop_back();
                int a = nodes[u].a, b = nodes[u].b;
                int freed = (rem[a] == 1 ? 1 : 0) + (b != a && rem[b] == 1 ? 1 : 0);
                max_peak = max(max_peak, live + (freed == 0 ? 1 : 0));
                if (max_peak > 48) { ok = false; break; }
                rem[a]--; if (b != a) rem[b]--;
                live += (rem[u] > 0 ? 1 : 0) - freed;
                seq.push_back(u);
                for (int w : deps[u]) if (--indeg[w] == 0) ready.push_back(w);
            }
            if (ok && (int)seq.size() == n - ks && emit_exact_order(seq, prog_out)) return true;
        }
        return false;
    }

    bool compile_zero_spill(vector<Instr> &prog_out, mt19937_64 &rng,
                            const vector<u64> *kept_ord = nullptr, int sa_iters = 6000) const {
        auto consider_seq = [&](vector<int> &seq, int siters) -> bool {
            PeakInfo pi = eval_order(seq);
            if (pi.max_peak <= 48 && emit_exact_order(seq, prog_out)) return true;
            if (siters > 0 && fix_pressure_sa(seq, siters, rng)) {
                if (emit_exact_order(seq, prog_out)) return true;
            }
            return false;
        };
        if (kept_ord && !kept_ord->empty()) {
            vector<int> seq_k_init = build_pref_order_with_insert(*kept_ord);
            vector<int> seq_k = seq_k_init;
            if (consider_seq(seq_k, sa_iters)) return true;
            if (sa_iters >= 6000) {
                for (int rep = 0; rep < 2; rep++) {
                    vector<int> seq_rep = seq_k_init;
                    if (consider_seq(seq_rep, sa_iters)) return true;
                }
            }
        }
        {
            vector<int> seq_self;
            for (int u = ks; u < (int)nodes.size(); u++) seq_self.push_back(u);
            if (consider_seq(seq_self, sa_iters)) return true;
        }
        if (schedule_min_live(prog_out, 24, rng)) return true;
        // LIFO-DFS fallback + SA
        {
            int n = (int)nodes.size();
            vector<vector<int>> deps(n);
            vector<int> indeg0(n, 0);
            for (int u = ks; u < n; u++) {
                deps[nodes[u].a].push_back(u);
                if (nodes[u].b != nodes[u].a) deps[nodes[u].b].push_back(u);
                if (nodes[u].a >= ks) indeg0[u]++;
                if (nodes[u].b >= ks && nodes[u].b != nodes[u].a) indeg0[u]++;
            }
            for (int rep = 0; rep < 3; rep++) {
                vector<int> indeg = indeg0, ready;
                for (int u = ks; u < n; u++) if (indeg[u] == 0) ready.push_back(u);
                if (rep > 0) shuffle(ready.begin(), ready.end(), rng);
                vector<int> seq_l;
                seq_l.reserve(n - ks);
                while (!ready.empty()) {
                    int u = ready.back(); ready.pop_back();
                    seq_l.push_back(u);
                    for (int ch : deps[u]) if (--indeg[ch] == 0) ready.push_back(ch);
                }
                if (consider_seq(seq_l, max(3000, sa_iters))) return true;
            }
        }
        return false;
    }
};

// ---------------------------------------------------------------- Pass 1: Multi-Donor Bitset Crossover
struct RawEdge { u16 a, b; char op; };

static bool run_crossover(const Puzzle &pz, const vector<Dag> &pool_dags,
                          const vector<u64> &pref_ord, int trials,
                          int max_allowed_len, mt19937_64 &rng,
                          Dag &out_dag, vector<Instr> &out_prog) {
    if (pool_dags.empty()) return false;
    const Dag &base = pool_dags[0];
    int ks = base.ks;
    vector<u64> vals;
    unordered_map<u64, int> v2i;
    for (int i = 0; i < ks; i++) {
        v2i[base.nodes[i].v] = (int)vals.size();
        vals.push_back(base.nodes[i].v);
    }
    auto add_val = [&](u64 v) -> int {
        auto it = v2i.find(v);
        if (it != v2i.end()) return it->second;
        if (vals.size() >= 1530) return -1;
        int idx = (int)vals.size();
        v2i[v] = idx;
        vals.push_back(v);
        return idx;
    };
    for (size_t di = 0; di < pool_dags.size(); di++) {
        for (int u = pool_dags[di].ks; u < (int)pool_dags[di].nodes.size(); u++) {
            add_val(pool_dags[di].nodes[u].v);
        }
    }
    int N = (int)vals.size();
    int orig_N = (int)base.nodes.size();
    int W = (N + 63) / 64;
    if (W > 24) return false;

    vector<int> targets_idx;
    vector<u8> is_target(N, 0);
    for (u64 tv : base.targets) {
        int ti = v2i.at(tv);
        targets_idx.push_back(ti);
        is_target[ti] = 1;
    }

    // Gather canonical edges: donor edges + ALL 1-op cross edges within V_u
    vector<vector<RawEdge>> edges_of(N);
    auto push_edge = [&](int vi, int ai, int bi, char op) {
        if (vi < ks || ai == vi || bi == vi) return;
        if ((op == '+' || op == '*') && ai > bi) swap(ai, bi);
        for (const auto &e : edges_of[vi]) {
            if (e.op == op && e.a == ai && e.b == bi) return;
        }
        edges_of[vi].push_back({(u16)ai, (u16)bi, op});
    };
    for (const auto &d : pool_dags) {
        for (int u = d.ks; u < (int)d.nodes.size(); u++) {
            auto itv = v2i.find(d.nodes[u].v);
            auto ita = v2i.find(d.nodes[d.nodes[u].a].v);
            auto itb = v2i.find(d.nodes[d.nodes[u].b].v);
            if (itv != v2i.end() && ita != v2i.end() && itb != v2i.end()) {
                push_edge(itv->second, ita->second, itb->second, d.nodes[u].op);
            }
        }
    }
    // Cross-recipes: a op b == v in V_u
    for (int i = 0; i < N; i++) {
        u64 a = vals[i];
        for (int j = i; j < N; j++) {
            u64 b = vals[j];
            u64 r = 0;
            if (apply_op('+', a, b, pz.L, r)) { auto it = v2i.find(r); if (it != v2i.end()) push_edge(it->second, i, j, '+'); }
            if (apply_op('*', a, b, pz.L, r)) { auto it = v2i.find(r); if (it != v2i.end()) push_edge(it->second, i, j, '*'); }
            if (b > a) { auto it = v2i.find(b - a); if (it != v2i.end()) push_edge(it->second, j, i, '-'); }
            if (a > 0 && b % a == 0 && b > a) { auto it = v2i.find(b / a); if (it != v2i.end()) push_edge(it->second, j, i, '/'); }
        }
    }

    auto covers = [&](const array<u64, 24> &allow) -> bool {
        array<u64, 24> seen{};
        for (int s = 0; s < ks; s++) seen[s >> 6] |= (1ULL << (s & 63));
        bool prog = true;
        while (prog) {
            prog = false;
            for (int v = ks; v < N; v++) {
                u64 mask = 1ULL << (v & 63);
                int w = v >> 6;
                if (!(allow[w] & mask) || (seen[w] & mask)) continue;
                for (const auto &e : edges_of[v]) {
                    if ((seen[e.a >> 6] & (1ULL << (e.a & 63))) &&
                        (seen[e.b >> 6] & (1ULL << (e.b & 63)))) {
                        seen[w] |= mask;
                        prog = true;
                        break;
                    }
                }
            }
            bool all_ok = true;
            for (int t : targets_idx) {
                if (!(seen[t >> 6] & (1ULL << (t & 63)))) { all_ok = false; break; }
            }
            if (all_ok) return true;
        }
        return false;
    };

    // Mandatory check (single deletion over all V_u)
    array<u64, 24> all_mask{};
    for (int i = 0; i < N; i++) all_mask[i >> 6] |= (1ULL << (i & 63));
    vector<int> non_mand;
    for (int v = ks; v < N; v++) {
        if (is_target[v]) continue;
        auto tmp = all_mask;
        tmp[v >> 6] &= ~(1ULL << (v & 63));
        if (covers(tmp)) non_mand.push_back(v);
    }

    int best_cnt = max_allowed_len;
    array<u64, 24> best_allow{};
    bool found = false;

    vector<int> perm = non_mand;
    for (int tr = 0; tr < trials; tr++) {
        array<u64, 24> allow = all_mask;
        int cnt = N - ks;
        int m = tr % 7;
        if (m > 0) {
            int keep_pct = 15 + m * 12; // 27% .. 87% of non-base optional nodes retained initially
            for (int v : non_mand) {
                if (v >= orig_N && (int)(rng() % 100) >= keep_pct) {
                    allow[v >> 6] &= ~(1ULL << (v & 63));
                    cnt--;
                }
            }
            if (!covers(allow)) continue;
        }
        shuffle(perm.begin(), perm.end(), rng);
        for (int v : perm) {
            u64 bit = 1ULL << (v & 63);
            int w = v >> 6;
            if (!(allow[w] & bit)) continue;
            allow[w] &= ~bit;
            if (covers(allow)) {
                cnt--;
            } else {
                allow[w] |= bit;
            }
        }
        if (cnt < best_cnt) {
            best_cnt = cnt;
            best_allow = allow;
            found = true;
        }
    }
    if (!found) return false;

    // Reconstruct acyclic DAG in firing order + prune unused via from_defs
    array<u64, 24> seen{};
    for (int s = 0; s < ks; s++) seen[s >> 6] |= (1ULL << (s & 63));
    unordered_map<u64, Step> defs;
    bool prog = true;
    while (prog) {
        prog = false;
        for (int v = ks; v < N; v++) {
            u64 mask = 1ULL << (v & 63);
            int w = v >> 6;
            if (!(best_allow[w] & mask) || (seen[w] & mask)) continue;
            for (const auto &e : edges_of[v]) {
                if ((seen[e.a >> 6] & (1ULL << (e.a & 63))) &&
                    (seen[e.b >> 6] & (1ULL << (e.b & 63)))) {
                    seen[w] |= mask;
                    defs[vals[v]] = {e.op, vals[e.a], vals[e.b], vals[v]};
                    prog = true;
                    break;
                }
            }
        }
    }
    Dag td;
    td.init(pz.L, pz.s, pz.t);
    if (!td.from_defs(defs, pref_ord) || td.len() >= max_allowed_len) return false;
    vector<Instr> nprog;
    if (!td.compile_zero_spill(nprog, rng, &pref_ord, 8000)) return false;
    if ((int)nprog.size() >= max_allowed_len) return false;
    out_dag = move(td);
    out_prog = move(nprog);
    return true;
}

// ---------------------------------------------------------------- Pass 2 & 3: Knockout, Bridges, & Neutral Walk
struct BridgeWitness { u16 a, b; char op; };

struct ViaRule {
    int bridge_idx;
    u16 target_v;
    u16 other_c;
    char op;
    u8 bridge_on_left; // 1: v = h op c, 0: v = c op h
};

struct KnockoutEnv {
    const Puzzle &pz;
    const Dag &dag;
    int n = 0, ks = 0, W = 0;
    vector<u64> vals;
    vector<u8> is_target;
    vector<int> targets_idx;
    vector<vector<RawEdge>> dir_edges; // per node v: existing + 1-op cross recipes within V
    vector<int> fanout;
    vector<vector<int>> users;
    vector<vector<int>> dom_cut; // dominated subtree per non-start node

    // Bridges S1 \ V
    struct Bridge { u64 val; array<BridgeWitness, 4> wits; u8 nw = 0; };
    vector<Bridge> bridges;
    unordered_map<u64, int> bridge_map;
    vector<vector<ViaRule>> via_for_v;

    KnockoutEnv(const Puzzle &p, const Dag &d) : pz(p), dag(d) {
        n = (int)d.nodes.size();
        ks = d.ks;
        W = (n + 63) / 64;
        vals.resize(n);
        for (int i = 0; i < n; i++) vals[i] = d.nodes[i].v;
        is_target.assign(n, 0);
        for (int ti : d.tnode) { is_target[ti] = 1; targets_idx.push_back(ti); }
        dir_edges.resize(n);
        fanout.assign(n, 0);
        users.resize(n);
        for (int u = ks; u < n; u++) {
            int a = d.nodes[u].a, b = d.nodes[u].b;
            dir_edges[u].push_back({(u16)a, (u16)b, d.nodes[u].op});
            users[a].push_back(u);
            fanout[a]++;
            if (b != a) { users[b].push_back(u); fanout[b]++; }
        }
        // Dominated subtree cut for each non-start node (reverse reach without u from targets)
        dom_cut.resize(n);
        for (int u = ks; u < n; u++) {
            vector<u8> vis(n, 0);
            vector<int> st;
            for (int ti : targets_idx) if (ti != u) st.push_back(ti);
            while (!st.empty()) {
                int x = st.back(); st.pop_back();
                if (x < ks || x == u || vis[x]) continue;
                vis[x] = 1;
                st.push_back(d.nodes[x].a);
                st.push_back(d.nodes[x].b);
            }
            for (int x = ks; x < n; x++) if (!vis[x]) dom_cut[u].push_back(x);
        }
        // Add ALL 1-op cross recipes within V & gather S1 bridges
        for (int i = 0; i < n; i++) {
            u64 a = vals[i];
            for (int j = i; j < n; j++) {
                u64 b = vals[j];
                auto add_cand = [&](u64 r, int ia, int ib, char op) {
                    auto it = d.val2idx.find(r);
                    if (it != d.val2idx.end()) {
                        int vi = it->second;
                        if (vi >= ks && vi != ia && vi != ib) {
                            if ((op == '+' || op == '*') && ia > ib) swap(ia, ib);
                            for (const auto &e : dir_edges[vi])
                                if (e.op == op && e.a == ia && e.b == ib) return;
                            dir_edges[vi].push_back({(u16)ia, (u16)ib, op});
                        }
                    } else {
                        auto bi = bridge_map.find(r);
                        int id;
                        if (bi == bridge_map.end()) {
                            id = (int)bridges.size();
                            bridge_map[r] = id;
                            bridges.push_back({r, {}, 0});
                        } else id = bi->second;
                        Bridge &br = bridges[id];
                        if (br.nw < 4) {
                            bool dup = false;
                            for (int w = 0; w < br.nw; w++)
                                if (br.wits[w].a == ia && br.wits[w].b == ib && br.wits[w].op == op) dup = true;
                            if (!dup) br.wits[br.nw++] = {(u16)ia, (u16)ib, op};
                        }
                    }
                };
                u64 r = 0;
                if (apply_op('+', a, b, pz.L, r)) add_cand(r, i, j, '+');
                if (apply_op('*', a, b, pz.L, r)) add_cand(r, i, j, '*');
                if (b > a) add_cand(b - a, j, i, '-');
                if (a > 0 && b % a == 0 && b > a) add_cand(b / a, j, i, '/');
            }
        }
        // Build via_for_v: for each non-start node v and c in V, inverses(v, c, L) in bridge_map
        via_for_v.resize(n);
        for (int vi = ks; vi < n; vi++) {
            u64 T = vals[vi];
            for (int ci = 0; ci < n; ci++) {
                if (ci == vi) continue;
                u64 x = vals[ci];
                auto add_h = [&](u64 h, char op, u8 hleft) {
                    auto it = bridge_map.find(h);
                    if (it != bridge_map.end()) {
                        via_for_v[vi].push_back({it->second, (u16)vi, (u16)ci, op, hleft});
                    }
                };
                if (T > x && T - x <= pz.L) add_h(T - x, '+', 1);
                if (x <= pz.L - T) add_h(T + x, '-', 1);
                if (x > T && x - T <= pz.L) add_h(x - T, '-', 0);
                if (x > 1 && T % x == 0 && T / x <= pz.L) add_h(T / x, '*', 1);
                if (x > 1 && (u128)T * x <= (u128)pz.L) add_h(T * x, '/', 1);
                if (x > T && x % T == 0 && x / T <= pz.L) add_h(x / T, '/', 0);
            }
        }
    }

    // Fast bitset closure over `allowed` (up to 384 nodes -> 6 u64s)
    bool close_bitset(const array<u64, 6> &allowed, array<u64, 6> &seen,
                      const ViaRule *extra1 = nullptr, const ViaRule *extra2 = nullptr) const {
        seen = {};
        for (int s = 0; s < ks; s++) seen[s >> 6] |= (1ULL << (s & 63));
        bool prog = true;
        while (prog) {
            prog = false;
            for (int v = ks; v < n; v++) {
                u64 mask = 1ULL << (v & 63);
                int w = v >> 6;
                if (!(allowed[w] & mask) || (seen[w] & mask)) continue;
                if (extra1 && extra1->target_v == v && (seen[extra1->other_c >> 6] & (1ULL << (extra1->other_c & 63)))) {
                    seen[w] |= mask; prog = true; continue;
                }
                if (extra2 && extra2->target_v == v && (seen[extra2->other_c >> 6] & (1ULL << (extra2->other_c & 63)))) {
                    seen[w] |= mask; prog = true; continue;
                }
                for (const auto &e : dir_edges[v]) {
                    if ((seen[e.a >> 6] & (1ULL << (e.a & 63))) &&
                        (seen[e.b >> 6] & (1ULL << (e.b & 63)))) {
                        seen[w] |= mask; prog = true; break;
                    }
                }
            }
            bool all_ok = true;
            for (int t : targets_idx) {
                if (!(seen[t >> 6] & (1ULL << (t & 63)))) { all_ok = false; break; }
            }
            if (all_ok) return true;
        }
        return false;
    }

    // Reconstruct DAG after winning ban_set + 0..2 ViaRules
    bool build_dag(const array<u64, 6> &allowed, const vector<pair<int, BridgeWitness>> &new_b,
                   const vector<ViaRule> &vias, const vector<u64> &pref_ord, Dag &out) const {
        array<u64, 6> seen{};
        for (int s = 0; s < ks; s++) seen[s >> 6] |= (1ULL << (s & 63));
        unordered_map<u64, Step> defs;
        for (const auto &[bi, bw] : new_b) {
            u64 hv = bridges[bi].val;
            defs[hv] = {bw.op, vals[bw.a], vals[bw.b], hv};
        }
        bool prog = true;
        while (prog) {
            prog = false;
            for (int v = ks; v < n; v++) {
                u64 mask = 1ULL << (v & 63);
                int w = v >> 6;
                if (!(allowed[w] & mask) || (seen[w] & mask)) continue;
                bool fired = false;
                for (const auto &e : dir_edges[v]) {
                    if ((seen[e.a >> 6] & (1ULL << (e.a & 63))) &&
                        (seen[e.b >> 6] & (1ULL << (e.b & 63)))) {
                        seen[w] |= mask;
                        defs[vals[v]] = {e.op, vals[e.a], vals[e.b], vals[v]};
                        prog = true; fired = true; break;
                    }
                }
                if (!fired) {
                    for (const auto &vr : vias) {
                        if (vr.target_v == v && (seen[vr.other_c >> 6] & (1ULL << (vr.other_c & 63)))) {
                            u64 hv = bridges[vr.bridge_idx].val, cv = vals[vr.other_c];
                            seen[w] |= mask;
                            defs[vals[v]] = vr.bridge_on_left ? Step{vr.op, hv, cv, vals[v]} : Step{vr.op, cv, hv, vals[v]};
                            prog = true; break;
                        }
                    }
                }
            }
        }
        out.init(pz.L, pz.s, pz.t);
        return out.from_defs(defs, pref_ord);
    }

    inline int active_witness(int bi, const array<u64, 6> &seen) const {
        const Bridge &br = bridges[bi];
        for (int w = 0; w < br.nw; w++) {
            const auto &bw = br.wits[w];
            if ((seen[bw.a >> 6] & (1ULL << (bw.a & 63))) &&
                (seen[bw.b >> 6] & (1ULL << (bw.b & 63)))) return w;
        }
        return -1;
    }

    // Try 0 or 1 bridge to repair `allowed` with banned nodes; also collect neutral swaps if `neut_out` non-null
    bool repair_ban(const array<u64, 6> &allowed, int ban_cnt, const vector<u64> &pref_ord,
                    Dag &win_dag, vector<Dag> *neut_out = nullptr, int max_neut = 4) const {
        array<u64, 6> seen{};
        if (close_bitset(allowed, seen)) {
            Dag td;
            if (build_dag(allowed, {}, {}, pref_ord, td) && td.len() < dag.len()) {
                win_dag = move(td);
                return true;
            }
            return false;
        }
        if (ban_cnt < 1) return false;
        // Collect candidate ViaRules on unreached nodes whose `other_c` IS in `seen` and bridge witness IS in `seen`
        // Notice we prioritize unreached nodes that are targets or have highest fanout!
        unordered_map<int, vector<const ViaRule *>> by_bridge;
        for (int v = ks; v < n; v++) {
            if (!(allowed[v >> 6] & (1ULL << (v & 63)))) continue;
            if (seen[v >> 6] & (1ULL << (v & 63))) continue;
            for (const auto &vr : via_for_v[v]) {
                if (!(seen[vr.other_c >> 6] & (1ULL << (vr.other_c & 63)))) continue;
                if (active_witness(vr.bridge_idx, seen) < 0) continue;
                by_bridge[vr.bridge_idx].push_back(&vr);
            }
        }
        // Try unlocking with 1 bridge! (Either 1 ViaRule or multiple ViaRules sharing the SAME bridge_idx!)
        // Wait: when 1 bridge h unlocks multiple unreached v's, pass the two highest-priority ViaRules or allow h in close!
        vector<pair<int, vector<const ViaRule *>>> cands;
        cands.reserve(by_bridge.size());
        for (auto &[bi, vrs] : by_bridge) cands.emplace_back(bi, move(vrs));
        sort(cands.begin(), cands.end(), [](const auto &A, const auto &B) { return A.second.size() > B.second.size(); });

        for (const auto &[bi, vrs] : cands) {
            int wi = active_witness(bi, seen);
            if (wi < 0) continue;
            // Test with vrs[0] (and vrs[1] if >=2 unlocked nodes)
            const ViaRule *e1 = vrs[0];
            const ViaRule *e2 = (vrs.size() >= 2 ? vrs[1] : nullptr);
            array<u64, 6> seen2{};
            if (close_bitset(allowed, seen2, e1, e2)) {
                vector<ViaRule> used_vrs;
                for (const ViaRule *vp : vrs) used_vrs.push_back(*vp);
                Dag td;
                if (build_dag(allowed, {{bi, bridges[bi].wits[wi]}}, used_vrs, pref_ord, td)) {
                    if (td.len() < dag.len()) {
                        win_dag = move(td);
                        return true;
                    }
                    if (neut_out && ban_cnt == 1 && td.len() == dag.len() && (int)neut_out->size() < max_neut) {
                        if (td.hash_dag() != dag.hash_dag()) neut_out->push_back(move(td));
                    }
                }
            }
        }
        return false;
    }
};

static bool solve_knockout_once(const Puzzle &pz, Dag &cur, vector<Instr> &best_prog,
                                vector<u64> &pref_ord, mt19937_64 &rng,
                                vector<Dag> *neut_archive = nullptr) {
    if ((int)cur.nodes.size() > 380) return false;
    KnockoutEnv env(pz, cur);
    array<u64, 6> all_ok{};
    for (int i = 0; i < env.n; i++) all_ok[i >> 6] |= (1ULL << (i & 63));
    auto accept_dag = [&](const Dag &td, const char *tag) -> bool {
        if (td.len() >= cur.len()) return false;
        vector<Instr> nprog;
        if (td.compile_zero_spill(nprog, rng, &pref_ord, 8000) && (int)nprog.size() < (int)best_prog.size()) {
            cur = td;
            best_prog = move(nprog);
            cur.from_prog(best_prog, &pref_ord);
            return true;
        }
        return false;
    };

    Dag win_dag;
    // 1. Single-node bans (both 0-bridge reductions AND neutral swaps) & dominated-subtree cuts
    for (int u = env.ks; u < env.n; u++) {
        if (!env.is_target[u]) {
            auto al = all_ok;
            al[u >> 6] &= ~(1ULL << (u & 63));
            if (env.repair_ban(al, 1, pref_ord, win_dag, neut_archive, 8)) {
                if (accept_dag(win_dag, "1ban")) return true;
            }
        }
        if (env.dom_cut[u].size() >= 2 && env.dom_cut[u].size() <= 16) {
            bool has_t = false;
            for (int x : env.dom_cut[u]) if (env.is_target[x]) { has_t = true; break; }
            if (!has_t) {
                auto al = all_ok;
                for (int x : env.dom_cut[u]) al[x >> 6] &= ~(1ULL << (x & 63));
                if (env.repair_ban(al, (int)env.dom_cut[u].size(), pref_ord, win_dag)) {
                    if (accept_dag(win_dag, "cut")) return true;
                }
            }
        }
    }

    // 2. Pair bans (2-ban + <=1 bridge): connected pairs (shared parent/consumer or edge) + top shared nodes
    vector<pair<int, int>> pairs;
    vector<int> non_ts;
    for (int u = env.ks; u < env.n; u++) if (!env.is_target[u]) non_ts.push_back(u);
    if (non_ts.size() <= 120) {
        for (size_t i = 0; i < non_ts.size(); i++)
            for (size_t j = i + 1; j < non_ts.size(); j++)
                pairs.emplace_back(non_ts[i], non_ts[j]);
    } else {
        unordered_set<u64> seen_p;
        auto add_p = [&](int a, int b) {
            if (a == b || env.is_target[a] || env.is_target[b] || a < env.ks || b < env.ks) return;
            if (a > b) swap(a, b);
            u64 k = ((u64)a << 16) | (u64)b;
            if (seen_p.insert(k).second) pairs.emplace_back(a, b);
        };
        for (int u = env.ks; u < env.n; u++) {
            add_p(u, cur.nodes[u].a);
            add_p(u, cur.nodes[u].b);
            for (size_t x = 0; x < env.users[u].size() && x < 16; x++)
                for (size_t y = x + 1; y < env.users[u].size() && y < 16; y++)
                    add_p(env.users[u][x], env.users[u][y]);
        }
        // Also all pairs among top-40 fanout<=4 recent nodes
        vector<int> f_nodes = non_ts;
        sort(f_nodes.begin(), f_nodes.end(), [&](int a, int b) { return env.fanout[a] < env.fanout[b]; });
        size_t lim = min<size_t>(f_nodes.size(), 45);
        for (size_t i = 0; i < lim; i++)
            for (size_t j = i + 1; j < lim; j++)
                add_p(f_nodes[i], f_nodes[j]);
    }
    for (auto [u1, u2] : pairs) {
        auto al = all_ok;
        al[u1 >> 6] &= ~(1ULL << (u1 & 63));
        al[u2 >> 6] &= ~(1ULL << (u2 & 63));
        if (env.repair_ban(al, 2, pref_ord, win_dag)) {
            if (accept_dag(win_dag, "2ban")) return true;
        }
    }

    // 3. Connected triple bans (3-ban + <=1 bridge)
    vector<int> low_fan;
    for (int u : non_ts) if (env.fanout[u] <= 3) low_fan.push_back(u);
    for (int u : low_fan) {
        vector<int> nbrs;
        auto push_n = [&](int x) {
            if (x >= env.ks && !env.is_target[x] && x != u) nbrs.push_back(x);
        };
        push_n(cur.nodes[u].a); push_n(cur.nodes[u].b);
        for (int w : env.users[u]) push_n(w);
        sort(nbrs.begin(), nbrs.end());
        nbrs.erase(unique(nbrs.begin(), nbrs.end()), nbrs.end());
        for (size_t i = 0; i < nbrs.size(); i++) {
            for (size_t j = i + 1; j < nbrs.size(); j++) {
                auto al = all_ok;
                al[u >> 6] &= ~(1ULL << (u & 63));
                al[nbrs[i] >> 6] &= ~(1ULL << (nbrs[i] & 63));
                al[nbrs[j] >> 6] &= ~(1ULL << (nbrs[j] & 63));
                if (env.repair_ban(al, 3, pref_ord, win_dag)) {
                    if (accept_dag(win_dag, "3ban")) return true;
                }
            }
        }
    }
    return false;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: elf3 puzzles.txt donors.txt -o out.txt [-t 11] [-sec 6] [-min_len 0] [-max_len 999] [-selftest 1]\n";
        return 1;
    }
    string puz_path = argv[1], donor_path = argv[2], out_path = "elf3_out.txt", ids_path = "";
    int threads = 11, min_len = 0, max_len = 999, selftest = 0;
    double sec_per_puz = 5.0;
    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "-t" && i + 1 < argc) threads = stoi(argv[++i]);
        else if (a == "-sec" && i + 1 < argc) sec_per_puz = stod(argv[++i]);
        else if (a == "-min_len" && i + 1 < argc) min_len = stoi(argv[++i]);
        else if (a == "-max_len" && i + 1 < argc) max_len = stoi(argv[++i]);
        else if (a == "-only_ids" && i + 1 < argc) ids_path = argv[++i];
        else if (a == "-selftest" && i + 1 < argc) selftest = stoi(argv[++i]);
    }

    auto puzzles = load_puzzles(puz_path);
    auto donors_map = load_donors(donor_path);
    unordered_set<string> filter_ids;
    if (!ids_path.empty()) {
        ifstream fi(ids_path);
        string id;
        while (fi >> id) filter_ids.insert(id);
    }

    vector<int> work_idxs;
    for (int i = 0; i < (int)puzzles.size(); i++) {
        if (!donors_map.count(puzzles[i].id) || donors_map.at(puzzles[i].id).empty()) continue;
        if (!filter_ids.empty() && !filter_ids.count(puzzles[i].id)) continue;
        work_idxs.push_back(i);
    }

    if (selftest) {
        int ok = 0, exact = 0;
        mt19937_64 rng(42);
        for (int idx : work_idxs) {
            const auto &pz = puzzles[idx];
            const auto &bp = donors_map.at(pz.id)[0];
            Dag d; d.init(pz.L, pz.s, pz.t);
            vector<u64> ord;
            if (!d.from_prog(bp, &ord)) { cerr << "SELFTEST FAIL parse " << pz.id << "\n"; continue; }
            vector<Instr> p2;
            if (d.compile_zero_spill(p2, rng, &ord, 8000)) {
                ok++;
                if ((int)p2.size() <= (int)bp.size()) exact++;
            } else {
                cerr << "SELFTEST SPILL on " << pz.id << " (len=" << bp.size() << ")\n";
            }
        }
        cout << "SELFTEST compile_zero_spill: ok=" << ok << "/" << work_idxs.size()
             << " exact=" << exact << "/" << work_idxs.size() << "\n";
        return ok == (int)work_idxs.size() ? 0 : 2;
    }

    double t_start = now_sec();
    atomic<size_t> next_job{0};
    atomic<int> done_cnt{0}, imp_cnt{0}, old_tot{0}, new_tot{0};
    atomic<int> wins_xover{0}, wins_knock{0}, wins_walk{0};
    mutex out_mtx, log_mtx;
    ofstream out_f(out_path);

    auto worker = [&](int tid) {
        while (true) {
            size_t wi = next_job.fetch_add(1);
            if (wi >= work_idxs.size()) break;
            const Puzzle &pz = puzzles[work_idxs[wi]];
            const auto &dprogs = donors_map.at(pz.id);
            vector<Instr> best_prog = dprogs[0];
            int init_len = (int)best_prog.size();
            old_tot.fetch_add(init_len);

            if (init_len < min_len || init_len > max_len) {
                new_tot.fetch_add(init_len);
                done_cnt.fetch_add(1);
                continue;
            }

            double tp0 = now_sec(), deadline = tp0 + sec_per_puz;
            mt19937_64 rng(mix64(hash<string>{}(pz.id) ^ (u64)tid ^ 0x517cc1b727220a95ULL));

            Dag best_dag;
            best_dag.init(pz.L, pz.s, pz.t);
            vector<u64> best_ord;
            best_dag.from_prog(best_prog, &best_ord);

            // Convert all unique donor programs into DAGs
            vector<Dag> archive;
            unordered_set<u64> seen_hashes;
            archive.push_back(best_dag);
            seen_hashes.insert(best_dag.hash_dag());
            for (size_t k = 1; k < dprogs.size(); k++) {
                Dag dd; dd.init(pz.L, pz.s, pz.t);
                if (dd.from_prog(dprogs[k])) {
                    u64 h = dd.hash_dag();
                    if (seen_hashes.insert(h).second) archive.push_back(move(dd));
                }
            }

            int p_x = 0, p_k = 0, p_w = 0;

            // Pass 1: Multi-donor bitset crossover
            if (archive.size() >= 2 && now_sec() < deadline) {
                Dag xd; vector<Instr> xp;
                if (run_crossover(pz, archive, best_ord, 800, (int)best_prog.size(), rng, xd, xp)) {
                    best_dag = move(xd);
                    best_prog = move(xp);
                    best_dag.from_prog(best_prog, &best_ord);
                    archive.insert(archive.begin(), best_dag);
                    seen_hashes.insert(best_dag.hash_dag());
                    p_x++; wins_xover++;
                }
            }

            // Pass 2: Direct knockout on current best (greedy descent + harvest neutral backbones)
            vector<Dag> work_q;
            while (now_sec() < deadline) {
                vector<Dag> neuts;
                if (solve_knockout_once(pz, best_dag, best_prog, best_ord, rng, &neuts)) {
                    p_k++; wins_knock++;
                    u64 h = best_dag.hash_dag();
                    if (seen_hashes.insert(h).second && archive.size() < 32) archive.push_back(best_dag);
                } else {
                    for (auto &nd : neuts) {
                        u64 h = nd.hash_dag();
                        if (seen_hashes.insert(h).second && work_q.size() < 24) {
                            work_q.push_back(move(nd));
                        }
                    }
                    break;
                }
            }

            // Pass 3: Neutral backbone walk + knockout on neighbors + archive crossover
            int walk_step = 0;
            while (now_sec() < deadline && (!work_q.empty() || walk_step < 12)) {
                walk_step++;
                if (work_q.empty()) break;
                Dag cur_w = work_q[rng() % work_q.size()];
                vector<u64> w_ord = cur_w.node_order();
                vector<Instr> dummy_prog(best_prog.size(), Instr{'+', 0, 0, 0});
                vector<Dag> more_neuts;
                if (solve_knockout_once(pz, cur_w, dummy_prog, w_ord, rng, &more_neuts)) {
                    // Knockout found shorter DAG on neutral neighbor! Check zero-spill compile!
                    vector<Instr> cprog;
                    if (cur_w.compile_zero_spill(cprog, rng, &w_ord, 8000) && (int)cprog.size() < (int)best_prog.size()) {
                        best_dag = move(cur_w);
                        best_prog = move(cprog);
                        best_dag.from_prog(best_prog, &best_ord);
                        p_w++; wins_walk++;
                        archive.insert(archive.begin(), best_dag);
                        work_q.clear();
                        // Immediately descend again
                        solve_knockout_once(pz, best_dag, best_prog, best_ord, rng, &more_neuts);
                    }
                }
                for (auto &nd : more_neuts) {
                    u64 h = nd.hash_dag();
                    if (seen_hashes.insert(h).second) {
                        if (archive.size() < 28) archive.push_back(nd);
                        if (work_q.size() < 24) work_q.push_back(move(nd));
                    }
                }
                if (walk_step % 4 == 0 && archive.size() >= 3 && now_sec() < deadline) {
                    vector<Dag> sub_pool = {best_dag};
                    for (size_t k = 1; k < archive.size() && sub_pool.size() < 12; k++) {
                        if (archive[k].len() <= best_dag.len() + 3) sub_pool.push_back(archive[k]);
                    }
                    if (sub_pool.size() >= 2) {
                        Dag xd; vector<Instr> xp;
                        if (run_crossover(pz, sub_pool, best_ord, 500, (int)best_prog.size(), rng, xd, xp)) {
                            best_dag = move(xd);
                            best_prog = move(xp);
                            best_dag.from_prog(best_prog, &best_ord);
                            p_x++; wins_xover++;
                            archive.insert(archive.begin(), best_dag);
                        }
                    }
                }
            }

            int fin_len = (int)best_prog.size();
            new_tot.fetch_add(fin_len);
            int d = done_cnt.fetch_add(1) + 1;
            if (fin_len < init_len) {
                imp_cnt.fetch_add(1);
                {
                    lock_guard<mutex> lk(out_mtx);
                    out_f << pz.id << " " << fin_len << " " << to_json(best_prog) << "\n" << flush;
                }
                {
                    lock_guard<mutex> lk(log_mtx);
                    cout << "[WIN] " << pz.id << ": " << init_len << " -> " << fin_len
                         << " (-" << (init_len - fin_len) << ", xover=" << p_x
                         << " knock=" << p_k << " walk=" << p_w << " arch=" << archive.size() << ")\n" << flush;
                }
            } else if (d % 25 == 0) {
                lock_guard<mutex> lk(log_mtx);
                cout << "progress " << d << "/" << work_idxs.size()
                     << " imp=" << imp_cnt.load()
                     << " tot=" << old_tot.load() << " -> " << new_tot.load()
                     << " (" << (new_tot.load() - old_tot.load()) << ") "
                     << (now_sec() - t_start) << "s\n" << flush;
            }
        }
    };

    int nt = max(1, min(threads, (int)work_idxs.size()));
    vector<thread> pool;
    for (int t = 0; t < nt; t++) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();

    cout << "DONE puzzles=" << done_cnt.load() << " improved=" << imp_cnt.load()
         << " total: " << old_tot.load() << " -> " << new_tot.load()
         << " (diff=" << (new_tot.load() - old_tot.load()) << ")"
         << " [xover=" << wins_xover.load() << " knock=" << wins_knock.load() << " walk=" << wins_walk.load() << "]"
         << " elapsed=" << (now_sec() - t_start) << "s\n";
    return 0;
}
