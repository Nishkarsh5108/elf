// s2_mitm.cpp — Full-DAG S2 Quotient-Pair & Large-Residual Meet-in-the-Middle Search
// Build: .venv/Scripts/python.exe -m ziglang c++ -O3 -march=native -std=c++20 cpp/s2_mitm.cpp -o s2_mitm.exe
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

static inline u64 isqrt64(u64 n) {
    u64 s = (u64)sqrtl((long double)n);
    while ((u128)(s + 1) * (s + 1) <= n) s++;
    while ((u128)s * s > n) s--;
    return s;
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

static unordered_map<string, vector<Instr>> load_progs_txt(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    unordered_map<string, vector<Instr>> out;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find(' ');
        if (p1 == string::npos) continue;
        size_t p2 = line.find(' ', p1 + 1);
        if (p2 == string::npos) continue;
        string pid = line.substr(0, p1);
        out[pid] = parse_prog_json(line.substr(p2 + 1));
    }
    return out;
}

// ---------------------------------------------------------------- DAG & 48-Reg Compiler
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

    bool from_prog(const vector<Instr> &prog, vector<u64> *ord_out = nullptr) {
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
        return from_defs(defs, ord);
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
            for (int rep = 0; rep < 2; rep++) {
                vector<int> seq_rep = seq_k_init;
                if (consider_seq(seq_rep, sa_iters)) return true;
            }
        }
        {
            vector<int> seq_self;
            for (int u = ks; u < (int)nodes.size(); u++) seq_self.push_back(u);
            if (consider_seq(seq_self, sa_iters)) return true;
        }
        if (schedule_min_live(prog_out, 20, rng)) return true;
        return false;
    }
};

// ---------------------------------------------------------------- S1 & Sharded S2 Hash Tables
struct S1Entry {
    u64 v;
    u16 a, b;
    u8 op;
};

struct S1Slot {
    u64 key;
    u32 idx0;
    u32 idx1; // optional 2nd witness with different operands!
};

struct S1Map {
    vector<S1Slot> slots;
    u32 mask = 0;

    void build(const vector<S1Entry> &entries) {
        u32 cap = 16;
        while (cap < entries.size() * 2 + 16) cap <<= 1;
        slots.assign(cap, {0ULL, UINT32_MAX, UINT32_MAX});
        mask = cap - 1;
        for (u32 i = 0; i < entries.size(); i++) {
            u64 v = entries[i].v;
            u32 pos = (u32)mix64(v) & mask;
            while (true) {
                if (slots[pos].key == 0ULL) {
                    slots[pos].key = v;
                    slots[pos].idx0 = i;
                    break;
                }
                if (slots[pos].key == v) {
                    if (slots[pos].idx1 == UINT32_MAX) {
                        const auto &e0 = entries[slots[pos].idx0];
                        const auto &ei = entries[i];
                        if (ei.a != e0.a || ei.b != e0.b) slots[pos].idx1 = i;
                    }
                    break;
                }
                pos = (pos + 1) & mask;
            }
        }
    }

    inline int find_clean(u64 v, const vector<S1Entry> &entries, const vector<u8> &forb) const {
        if (!v) return -1;
        u32 pos = (u32)mix64(v) & mask;
        while (true) {
            u64 k = slots[pos].key;
            if (k == 0ULL) return -1;
            if (k == v) {
                u32 i0 = slots[pos].idx0;
                if (!forb[entries[i0].a] && !forb[entries[i0].b]) return (int)i0;
                u32 i1 = slots[pos].idx1;
                if (i1 != UINT32_MAX && !forb[entries[i1].a] && !forb[entries[i1].b]) return (int)i1;
                return -1;
            }
            pos = (pos + 1) & mask;
        }
    }
};

// Sharded S2 open-addressed hash table: 64 independent L3-cache-friendly shards!
struct alignas(16) S2Slot {
    u64 key;
    u32 s1_idx;
    u16 v_idx;
    u8 op;
    u8 flags; // bit0: valid, remaining unused
};

struct S2Table {
    static constexpr int SHARDS = 64;
    vector<S2Slot> table;
    u32 shard_size = 0;
    u32 shard_mask = 0;
    size_t count = 0;

    void init(int s2bits) {
        size_t total = 1ULL << s2bits;
        shard_size = (u32)(total / SHARDS);
        shard_mask = shard_size - 1;
        table.assign(total, {0ULL, 0, 0, 0, 0});
        count = 0;
    }

    static inline int shard_of(u64 x) {
        return (int)((mix64(x) >> 58) & 63);
    }

    inline const S2Slot *lookup(u64 x) const {
        if (!x) return nullptr;
        u64 h = mix64(x);
        u32 sh = (u32)((h >> 58) & 63);
        u32 base = sh * shard_size;
        u32 pos = (u32)h & shard_mask;
        for (u32 p = 0; p < 128; p++) {
            const S2Slot &s = table[base + ((pos + p) & shard_mask)];
            if (s.key == x) return &s;
            if (s.key == 0ULL) return nullptr;
        }
        return nullptr;
    }

    inline bool is_clean(const S2Slot *s, const vector<S1Entry> &s1, const vector<u8> &forb) const {
        if (!s) return false;
        if (forb[s->v_idx]) return false;
        const auto &e = s1[s->s1_idx];
        return !forb[e.a] && !forb[e.b];
    }
};

struct S2Cand {
    u64 key;
    u32 s1_idx;
    u16 v_idx;
    u8 op;
    u8 pad;
};

static inline char op_from_id(u8 id) {
    static const char map_ops[4] = {'+', '-', '*', '/'};
    return map_ops[id];
}

// Lookup clean value in V, S1, or S2 returning minimal cost (0, 1, 2) or 99
struct ResolvedVal {
    int cost = 99; // 0=V, 1=S1, 2=S2
    int s1_idx = -1;
    const S2Slot *s2 = nullptr;
};

static ResolvedVal resolve_clean(u64 val,
                                 const unordered_map<u64, int> &vmap,
                                 const vector<u8> &forb,
                                 const S1Map &s1m, const vector<S1Entry> &s1_list,
                                 const S2Table &s2) {
    auto it = vmap.find(val);
    if (it != vmap.end() && !forb[it->second]) return {0, -1, nullptr};
    int i1 = s1m.find_clean(val, s1_list, forb);
    if (i1 >= 0) return {1, i1, nullptr};
    const S2Slot *p2 = s2.lookup(val);
    if (s2.is_clean(p2, s1_list, forb)) return {2, -1, p2};
    return {99, -1, nullptr};
}

// Reconstruct all newly added Steps for a clean value (`cost 0, 1, or 2`) and emit dependencies.
static void add_resolved_steps(u64 val, const ResolvedVal &rv,
                               const vector<DagNode> &nodes,
                               const vector<S1Entry> &s1_list,
                               unordered_map<u64, Step> &new_defs) {
    if (rv.cost == 0) return;
    if (rv.cost == 1) {
        const auto &e = s1_list[rv.s1_idx];
        new_defs[val] = {op_from_id(e.op), nodes[e.a].v, nodes[e.b].v, val};
        return;
    }
    if (rv.cost == 2 && rv.s2) {
        const auto &e = s1_list[rv.s2->s1_idx];
        new_defs[e.v] = {op_from_id(e.op), nodes[e.a].v, nodes[e.b].v, e.v};
        u64 x = e.v, y = nodes[rv.s2->v_idx].v;
        // s2 op encodings: 0: x+y, 1: x-y, 2: y-x, 3: x*y, 4: x/y, 5: y/x
        Step st{};
        switch (rv.s2->op) {
            case 0: st = {'+', x, y, val}; break;
            case 1: st = {'-', x, y, val}; break;
            case 2: st = {'-', y, x, val}; break;
            case 3: st = {'*', x, y, val}; break;
            case 4: st = {'/', x, y, val}; break;
            case 5: st = {'/', y, x, val}; break;
        }
        new_defs[val] = st;
    }
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: s2_mitm puzzles.txt progs.txt -o out.txt [-t 11] [-minp 5] [-s2bits 25] [-only_ids ids.txt]\n";
        return 1;
    }
    string puz_path = argv[1], prog_path = argv[2], out_path = "s2_out.txt", ids_path = "";
    int threads = 11, minp = 5, s2bits = 25;
    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "-t" && i + 1 < argc) threads = stoi(argv[++i]);
        else if (a == "-minp" && i + 1 < argc) minp = stoi(argv[++i]);
        else if (a == "-s2bits" && i + 1 < argc) s2bits = stoi(argv[++i]);
        else if (a == "-only_ids" && i + 1 < argc) ids_path = argv[++i];
    }

    auto puzzles = load_puzzles(puz_path);
    auto progs = load_progs_txt(prog_path);
    unordered_set<string> filter_ids;
    if (!ids_path.empty()) {
        ifstream fi(ids_path);
        string id;
        while (fi >> id) filter_ids.insert(id);
    }

    S2Table s2;
    s2.init(s2bits);

    ofstream out_f(out_path);
    int total_old = 0, total_new = 0, puz_done = 0, puz_imp = 0, dag_wins = 0, compile_wins = 0;
    double t_all = now_sec();

    for (const auto &pz : puzzles) {
        if (!progs.count(pz.id)) continue;
        if (!filter_ids.empty() && !filter_ids.count(pz.id)) continue;
        vector<Instr> best_prog = progs.at(pz.id);
        int old_len = (int)best_prog.size();
        total_old += old_len;

        Dag cur;
        cur.init(pz.L, pz.s, pz.t);
        vector<u64> ref_order;
        if (!cur.from_prog(best_prog, &ref_order)) {
            cerr << "BAD INPUT PROG " << pz.id << "\n";
            total_new += old_len;
            continue;
        }
        if ((int)cur.nodes.size() - cur.ks < old_len) {
            mt19937_64 rng(123);
            vector<Instr> p2;
            if (cur.compile_zero_spill(p2, rng, &ref_order, 6000) && (int)p2.size() < (int)best_prog.size())
                best_prog = move(p2);
        }

        // First scan if any target has priv >= minp
        auto analyze_priv = [&](const Dag &d, vector<int> &rc, vector<vector<int>> &cones,
                                vector<vector<u8>> &desc_incl) {
            int n = (int)d.nodes.size();
            rc.assign(n, 0);
            cones.assign(d.tnode.size(), {});
            vector<vector<int>> users(n);
            for (int u = d.ks; u < n; u++) {
                users[d.nodes[u].a].push_back(u);
                if (d.nodes[u].b != d.nodes[u].a) users[d.nodes[u].b].push_back(u);
            }
            for (size_t ti = 0; ti < d.tnode.size(); ti++) {
                vector<u8> vis(n, 0);
                vector<int> st = {d.tnode[ti]};
                while (!st.empty()) {
                    int u = st.back(); st.pop_back();
                    if (u < d.ks || vis[u]) continue;
                    vis[u] = 1;
                    cones[ti].push_back(u);
                    st.push_back(d.nodes[u].a);
                    st.push_back(d.nodes[u].b);
                }
                for (int u : cones[ti]) rc[u]++;
            }
            desc_incl.assign(d.tnode.size(), vector<u8>(n, 0));
            for (size_t ti = 0; ti < d.tnode.size(); ti++) {
                vector<int> st = {d.tnode[ti]};
                while (!st.empty()) {
                    int u = st.back(); st.pop_back();
                    if (desc_incl[ti][u]) continue;
                    desc_incl[ti][u] = 1;
                    for (int w : users[u]) st.push_back(w);
                }
            }
        };

        vector<int> rc0;
        vector<vector<int>> cones0;
        vector<vector<u8>> desc0;
        analyze_priv(cur, rc0, cones0, desc0);
        int max_p = 0;
        for (size_t ti = 0; ti < cur.tnode.size(); ti++) {
            int pc = 0;
            for (int u : cones0[ti]) if (rc0[u] == 1) pc++;
            max_p = max(max_p, pc);
        }
        if (max_p < minp) {
            total_new += (int)best_prog.size();
            puz_done++;
            continue;
        }

        double tp0 = now_sec();
        int pz_dag_wins = 0, pz_cmp_wins = 0;
        mt19937_64 rng(mix64(hash<string>{}(pz.id) ^ 0x9e3779b97f4a7c15ULL));

        for (int round = 0; round < 3; round++) {
            int n = (int)cur.nodes.size();
            vector<int> rc;
            vector<vector<int>> cones;
            vector<vector<u8>> desc_incl;
            analyze_priv(cur, rc, cones, desc_incl);

            // Identify targets with priv >= minp
            vector<int> hard_ts;
            for (size_t ti = 0; ti < cur.tnode.size(); ti++) {
                int pc = 0;
                for (int u : cones[ti]) if (rc[u] == 1) pc++;
                if (pc >= minp) hard_ts.push_back((int)ti);
            }
            if (hard_ts.empty()) break;

            // Build S1 across ALL nodes in cur
            vector<S1Entry> s1_list;
            s1_list.reserve(n * n * 2);
            for (int i = 0; i < n; i++) {
                u64 a = cur.nodes[i].v;
                for (int j = i; j < n; j++) {
                    u64 b = cur.nodes[j].v;
                    u64 r = 0;
                    if (apply_op('+', a, b, pz.L, r) && !cur.val2idx.count(r))
                        s1_list.push_back({r, (u16)i, (u16)j, 0});
                    if (apply_op('*', a, b, pz.L, r) && !cur.val2idx.count(r))
                        s1_list.push_back({r, (u16)i, (u16)j, 2});
                    if (b > a && !cur.val2idx.count(b - a))
                        s1_list.push_back({b - a, (u16)j, (u16)i, 1});
                    if (a > 0 && b % a == 0 && b > a && !cur.val2idx.count(b / a))
                        s1_list.push_back({b / a, (u16)j, (u16)i, 3});
                }
            }
            S1Map s1m;
            s1m.build(s1_list);

            // Build compact list of unique S1 representatives for outer S2 scan + M1/M2/M6
            vector<u32> uniq_s1;
            uniq_s1.reserve(s1_list.size());
            for (const auto &sl : s1m.slots) if (sl.key) uniq_s1.push_back(sl.idx0);
            sort(uniq_s1.begin(), uniq_s1.end(), [&](u32 i, u32 j) { return s1_list[i].v < s1_list[j].v; });

            // Cap S2 generator seed count so S2 build & M1/M2 stay fast (< 2s/puzzle) while dense
            vector<u32> s1_for_s2 = uniq_s1;
            if (s1_for_s2.size() > 36000) {
                // Keep all <= (1<<26) + near sqrt(L) + uniform stride of the rest
                u64 sqL = isqrt64(pz.L);
                u64 mid_lo = max<u64>(1, sqL >> 8), mid_hi = (sqL <= pz.L >> 8) ? (sqL << 8) : pz.L;
                vector<u32> keep, rest;
                for (u32 idx : uniq_s1) {
                    u64 v = s1_list[idx].v;
                    if (v <= (1ULL << 22) || (v >= mid_lo && v <= mid_hi)) keep.push_back(idx);
                    else rest.push_back(idx);
                }
                if (keep.size() < 36000 && !rest.empty()) {
                    size_t need = 36000 - keep.size();
                    double step = max(1.0, (double)rest.size() / (double)need);
                    for (double p = 0; p < (double)rest.size() && keep.size() < 36000; p += step)
                        keep.push_back(rest[(size_t)p]);
                } else if (keep.size() > 36000) {
                    vector<u32> sub;
                    double step = (double)keep.size() / 36000.0;
                    for (double p = 0; p < (double)keep.size() && sub.size() < 36000; p += step)
                        sub.push_back(keep[(size_t)p]);
                    keep = move(sub);
                }
                s1_for_s2 = move(keep);
            }

            // Build sharded S2 in parallel across `threads`
            fill(s2.table.begin(), s2.table.end(), S2Slot{0ULL, 0, 0, 0, 0});
            s2.count = 0;
            int nt_use = max(1, min(threads, 12));
            vector<vector<vector<S2Cand>>> t_buckets(nt_use, vector<vector<S2Cand>>(S2Table::SHARDS));
            size_t max_per_shard_slot = (size_t)(s2.shard_size * 0.65);

            {
                vector<thread> pool;
                for (int tid = 0; tid < nt_use; tid++) {
                    pool.emplace_back([&, tid]() {
                        auto &bk = t_buckets[tid];
                        for (size_t ii = tid; ii < s1_for_s2.size(); ii += nt_use) {
                            u32 s1i = s1_for_s2[ii];
                            u64 x = s1_list[s1i].v;
                            for (int vi = 0; vi < n; vi++) {
                                u64 y = cur.nodes[vi].v;
                                auto push_c = [&](u64 val, u8 op_id) {
                                    if (!val || cur.val2idx.count(val)) return;
                                    int sh = S2Table::shard_of(val);
                                    if (bk[sh].size() < max_per_shard_slot * 2 / nt_use + 4096)
                                        bk[sh].push_back({val, s1i, (u16)vi, op_id, 0});
                                };
                                if (x <= pz.L - y) push_c(x + y, 0);
                                if (x > y) push_c(x - y, 1);
                                else if (y > x) push_c(y - x, 2);
                                if ((u128)x * y <= (u128)pz.L) push_c(x * y, 3);
                                if (y > 1 && x % y == 0) push_c(x / y, 4);
                                if (x > 1 && y % x == 0) push_c(y / x, 5);
                            }
                        }
                    });
                }
                for (auto &th : pool) th.join();
            }

            // Parallel lock-free shard insertion into S2
            atomic<size_t> total_ins{0};
            {
                atomic<int> next_sh{0};
                vector<thread> pool;
                for (int tid = 0; tid < nt_use; tid++) {
                    pool.emplace_back([&]() {
                        while (true) {
                            int sh = next_sh.fetch_add(1);
                            if (sh >= S2Table::SHARDS) break;
                            u32 base = (u32)sh * s2.shard_size;
                            size_t sh_cnt = 0;
                            for (int p = 0; p < nt_use; p++) {
                                const auto &vec = t_buckets[p][sh];
                                for (const auto &c : vec) {
                                    if (sh_cnt >= max_per_shard_slot) break;
                                    u64 h = mix64(c.key);
                                    u32 pos = (u32)h & s2.shard_mask;
                                    for (u32 k = 0; k < 128; k++) {
                                        S2Slot &sl = s2.table[base + ((pos + k) & s2.shard_mask)];
                                        if (sl.key == c.key) break;
                                        if (sl.key == 0ULL) {
                                            sl.key = c.key;
                                            sl.s1_idx = c.s1_idx;
                                            sl.v_idx = c.v_idx;
                                            sl.op = c.op;
                                            sl.flags = 1;
                                            sh_cnt++;
                                            break;
                                        }
                                    }
                                }
                            }
                            total_ins.fetch_add(sh_cnt);
                        }
                    });
                }
                for (auto &th : pool) th.join();
            }
            s2.count = total_ins.load();

            // Extract s2_mid for Shape M3 (S2 * S2 +- C)
            u64 tmin = *min_element(cur.targets.begin(), cur.targets.end());
            u64 tmax = *max_element(cur.targets.begin(), cur.targets.end());
            u64 m3_lo = max<u64>(2, isqrt64(tmin) / 32);
            u64 m3_hi = isqrt64(tmax) * 32;
            vector<u32> s2_mid_pos;
            s2_mid_pos.reserve(min<size_t>(s2.count, 3000000));
            for (u32 p = 0; p < (u32)s2.table.size(); p++) {
                u64 k = s2.table[p].key;
                if (k >= m3_lo && k <= m3_hi) {
                    s2_mid_pos.push_back(p);
                    if (s2_mid_pos.size() >= 3000000) break;
                }
            }

            // Combined V U uniq_s1 sorted list for M1/M2 A-scan
            struct ACand { u64 v; int cost; int s1i; };
            vector<ACand> all_A;
            all_A.reserve(n + uniq_s1.size());
            for (int i = 0; i < n; i++) if (cur.nodes[i].v >= 2) all_A.push_back({cur.nodes[i].v, 0, -1});
            for (u32 s1i : uniq_s1) if (s1_list[s1i].v >= 2) all_A.push_back({s1_list[s1i].v, 1, (int)s1i});

            bool improved_this_round = false;

            // Search each hard target ti
            for (int ti : hard_ts) {
                int tn = cur.tnode[ti];
                u64 T = cur.nodes[tn].v;
                int pc = 0;
                for (int u : cones[ti]) if (rc[u] == 1) pc++;
                int old_budget = pc + (rc[tn] >= 2 ? 1 : 0);

                vector<u8> forb(n, 0);
                for (int u : cones[ti]) if (rc[u] == 1) forb[u] = 1;
                for (int u = 0; u < n; u++) if (desc_incl[ti][u]) forb[u] = 1;

                // Clean V-values sorted by closeness to T (for M2)
                vector<pair<u64, int>> clean_U;
                for (int u = 0; u < n; u++) {
                    if (!forb[u]) {
                        u64 uv = cur.nodes[u].v;
                        clean_U.push_back({T > uv ? T - uv : uv - T, u});
                    }
                }
                sort(clean_U.begin(), clean_U.end());
                vector<int> clean_V_idxs;
                for (int u = 0; u < n; u++) if (!forb[u] && cur.nodes[u].v >= 2) clean_V_idxs.push_back(u);

                struct Hit {
                    int est_cost;
                    unordered_map<u64, Step> extra;
                };
                mutex hit_mtx;
                bool found_hit = false;
                Hit best_hit;
                best_hit.est_cost = old_budget;

                auto try_accept_extra = [&](int est_cost, unordered_map<u64, Step> &&ext) {
                    if (est_cost >= old_budget) return;
                    // Check acyclicity vs desc_incl: none of new steps != T may match a node in desc_incl
                    for (const auto &[nv, st] : ext) {
                        auto it = cur.val2idx.find(nv);
                        if (it != cur.val2idx.end() && nv != T && desc_incl[ti][it->second]) return;
                    }
                    lock_guard<mutex> lk(hit_mtx);
                    if (!found_hit || est_cost < best_hit.est_cost) {
                        found_hit = true;
                        best_hit.est_cost = est_cost;
                        best_hit.extra = move(ext);
                    }
                };

                // Helper to emit a product-diff solution: T' = A * q +- r, optionally wrapped by T = T' op U or k*T = T'
                auto record_mul = [&](u64 A, const ResolvedVal &ra,
                                      u64 q, const ResolvedVal &rq,
                                      bool has_r, u64 rval, bool r_add, const ResolvedVal &rr,
                                      int wrap_kind, u64 Uval, u64 Tp) {
                    // wrap_kind: 0 => Tp==T; 1 => T = Tp + U; 2 => T = U - Tp; 3 => T = Tp - U; 4 => T = Tp / U (k*T=Tp)
                    u64 P = A * q;
                    if (P == 0 || P > pz.L) return;
                    if (wrap_kind == 0) Tp = T;
                    unordered_map<u64, Step> ext;
                    add_resolved_steps(A, ra, cur.nodes, s1_list, ext);
                    add_resolved_steps(q, rq, cur.nodes, s1_list, ext);
                    if (!cur.val2idx.count(P) || forb[cur.val2idx.at(P)]) {
                        if (P != A && P != q) ext[P] = {'*', A, q, P};
                    }
                    if (has_r) {
                        add_resolved_steps(rval, rr, cur.nodes, s1_list, ext);
                        if (!cur.val2idx.count(Tp) || forb[cur.val2idx.at(Tp)]) {
                            ext[Tp] = r_add ? Step{'+', P, rval, Tp} : Step{'-', P, rval, Tp};
                        }
                    } else {
                        // P == Tp already
                    }
                    if (wrap_kind == 1) ext[T] = {'+', Tp, Uval, T};
                    else if (wrap_kind == 2) ext[T] = {'-', Uval, Tp, T};
                    else if (wrap_kind == 3) ext[T] = {'-', Tp, Uval, T};
                    else if (wrap_kind == 4) ext[T] = {'/', Tp, Uval, T};

                    int real_cost = (int)ext.size();
                    if (real_cost < old_budget) try_accept_extra(real_cost, move(ext));
                };

                // Check single T' across A-list slice [beg, end)
                auto scan_mul_range = [&](u64 Tp, int wrap_kind, u64 Uval, int base_wrap_cost,
                                          size_t a_limit, int tid_off, int tid_step) {
                    for (size_t idx = tid_off; idx < a_limit; idx += tid_step) {
                        const auto &ac = all_A[idx];
                        if (ac.s1i >= 0) {
                            const auto &e = s1_list[ac.s1i];
                            if (forb[e.a] || forb[e.b]) continue;
                        } else {
                            // V entry already clean check via map? wait: filter clean when building? let's check
                            if (forb[cur.val2idx.at(ac.v)]) continue;
                        }
                        u64 A = ac.v;
                        if (A > Tp) continue;
                        int rem_budget = old_budget - 1 - base_wrap_cost - ac.cost;
                        if (rem_budget <= 0) continue; // need at least A*q (1 op) + q

                        u64 q0 = Tp / A;
                        ResolvedVal ra;
                        ra.cost = ac.cost;
                        ra.s1_idx = ac.s1i;

                        for (int dq = 0; dq < 2; dq++) {
                            u64 q = q0 + dq;
                            if (q < 2 || (u128)A * q > (u128)pz.L) continue;
                            u64 P = A * q;
                            u64 rval = (P <= Tp) ? (Tp - P) : (P - Tp);
                            bool r_add = (P <= Tp);
                            int min_need = 1 + (rval > 0 ? 1 : 0);
                            if (min_need > rem_budget) continue;

                            auto itq = cur.val2idx.find(q);
                            ResolvedVal rq;
                            if (itq != cur.val2idx.end() && !forb[itq->second]) rq = {0, -1, nullptr};
                            else {
                                int iq1 = s1m.find_clean(q, s1_list, forb);
                                if (iq1 >= 0) rq = {1, iq1, nullptr};
                                else {
                                    const S2Slot *pq2 = s2.lookup(q);
                                    if (s2.is_clean(pq2, s1_list, forb)) rq = {2, -1, pq2};
                                    else continue;
                                }
                            }
                            int after_q = rem_budget - 1 - rq.cost; // budget for ±rval
                            if (rval == 0) {
                                if (after_q >= 0) {
                                    record_mul(A, ra, q, rq, false, 0, true, {}, wrap_kind, Uval, Tp);
                                }
                            } else if (after_q >= 1) {
                                ResolvedVal rr = resolve_clean(rval, cur.val2idx, forb, s1m, s1_list, s2);
                                if (rr.cost <= after_q - 1) {
                                    record_mul(A, ra, q, rq, true, rval, r_add, rr, wrap_kind, Uval, Tp);
                                }
                            }
                        }
                    }
                };

                // Run M1, M2, M3, M4, M5, M6 in parallel across nt_use threads
                {
                    vector<thread> pool;
                    for (int tid = 0; tid < nt_use; tid++) {
                        pool.emplace_back([&, tid]() {
                            // M1: T = A * B +- C (all A in V U S1)
                            scan_mul_range(T, 0, 0, 0, all_A.size(), tid, nt_use);

                            // M4: k * T = A * B +- C for clean k in V (2..65536)
                            for (int ui : clean_V_idxs) {
                                u64 kv = cur.nodes[ui].v;
                                if (kv >= 2 && kv <= 65536 && (u128)kv * T <= (u128)pz.L) {
                                    scan_mul_range(kv * T, 4, kv, 1, all_A.size(), tid, nt_use);
                                }
                            }

                            // M2: T +- U = A * B +- C
                            // Full all_A for top-24 closest U (when old_budget >= 6); clean V A for ALL U!
                            for (size_t ui = 0; ui < clean_U.size(); ui++) {
                                u64 uv = cur.nodes[clean_U[ui].second].v;
                                size_t alim = (ui < 32 && old_budget >= 6) ? all_A.size() : min<size_t>(n, all_A.size());
                                if (T > uv) scan_mul_range(T - uv, 1, uv, 1, alim, tid, nt_use);
                                if (uv > T) scan_mul_range(uv - T, 2, uv, 1, alim, tid, nt_use);
                                if (uv <= pz.L - T) scan_mul_range(T + uv, 3, uv, 1, alim, tid, nt_use);
                            }

                            // M3: T = A * B +- C with A in s2_mid, B in S2/S1/V
                            if (old_budget >= 6) {
                                for (size_t i = tid; i < s2_mid_pos.size(); i += nt_use) {
                                    const S2Slot &sa = s2.table[s2_mid_pos[i]];
                                    if (!s2.is_clean(&sa, s1_list, forb)) continue;
                                    u64 A = sa.key;
                                    if (A < 2 || A > T) continue;
                                    u64 q0 = T / A;
                                    for (int dq = 0; dq < 2; dq++) {
                                        u64 q = q0 + dq;
                                        if (q < 2 || (u128)A * q > (u128)pz.L) continue;
                                        u64 P = A * q;
                                        u64 rval = (P <= T) ? (T - P) : (P - T);
                                        bool r_add = (P <= T);
                                        int rem_b = old_budget - 3 - (rval > 0 ? 1 : 0); // 2(A) + 1(*) + (r?1:0)
                                        if (rem_b < 0) continue;
                                        ResolvedVal rq = resolve_clean(q, cur.val2idx, forb, s1m, s1_list, s2);
                                        if (rq.cost > rem_b) continue;
                                        ResolvedVal rr;
                                        if (rval > 0) {
                                            rr = resolve_clean(rval, cur.val2idx, forb, s1m, s1_list, s2);
                                            if (rr.cost > rem_b - rq.cost) continue;
                                        }
                                        ResolvedVal ra{2, -1, &sa};
                                        record_mul(A, ra, q, rq, rval > 0, rval, r_add, rr, 0, 0, T);
                                    }
                                }
                            }

                            // M5: S2 additive/subtractive 4..5 op match: (T [+- U]) +- A in S2
                            for (size_t i = tid; i < uniq_s1.size(); i += nt_use) {
                                u32 s1i = uniq_s1[i];
                                const auto &e = s1_list[s1i];
                                if (forb[e.a] || forb[e.b]) continue;
                                u64 Av = e.v;
                                auto check_s2_wrap = [&](u64 Tp, int wk, u64 Uv, int wc) {
                                    if (old_budget <= 4 + wc) return;
                                    auto try_w = [&](u64 W, char top, bool t_left) {
                                        const S2Slot *ps2 = s2.lookup(W);
                                        if (!s2.is_clean(ps2, s1_list, forb)) return;
                                        unordered_map<u64, Step> ext;
                                        ext[Av] = {op_from_id(e.op), cur.nodes[e.a].v, cur.nodes[e.b].v, Av};
                                        add_resolved_steps(W, ResolvedVal{2, -1, ps2}, cur.nodes, s1_list, ext);
                                        u64 res = wk ? Tp : T;
                                        ext[res] = t_left ? Step{top, W, Av, res} : Step{top, Av, W, res};
                                        if (wk == 1) ext[T] = {'+', Tp, Uv, T};
                                        else if (wk == 2) ext[T] = {'-', Uv, Tp, T};
                                        else if (wk == 3) ext[T] = {'-', Tp, Uv, T};
                                        if ((int)ext.size() < old_budget) try_accept_extra((int)ext.size(), move(ext));
                                    };
                                    if (Tp > Av) try_w(Tp - Av, '+', true);
                                    if (Av <= pz.L - Tp) try_w(Tp + Av, '-', true);
                                    if (Av > Tp) try_w(Av - Tp, '-', false);
                                };
                                check_s2_wrap(T, 0, 0, 0);

                                // M6: T +- C_s1 = A_V * B_S2 (exact division by clean V!)
                                if (old_budget >= 5) {
                                    auto check_div = [&](u64 Tp, bool c_add, bool t_sub_from_c) {
                                        for (int vi : clean_V_idxs) {
                                            u64 A = cur.nodes[vi].v;
                                            if (A < 2 || Tp % A != 0) continue;
                                            u64 B = Tp / A;
                                            if (B < 2) continue;
                                            ResolvedVal rb = resolve_clean(B, cur.val2idx, forb, s1m, s1_list, s2);
                                            if (rb.cost > old_budget - 4) continue;
                                            unordered_map<u64, Step> ext;
                                            ext[Av] = {op_from_id(e.op), cur.nodes[e.a].v, cur.nodes[e.b].v, Av};
                                            add_resolved_steps(B, rb, cur.nodes, s1_list, ext);
                                            ext[Tp] = {'*', A, B, Tp};
                                            ext[T] = t_sub_from_c ? Step{'-', Av, Tp, T} : (c_add ? Step{'+', Tp, Av, T} : Step{'-', Tp, Av, T});
                                            if ((int)ext.size() < old_budget) try_accept_extra((int)ext.size(), move(ext));
                                        }
                                    };
                                    if (T > Av) check_div(T - Av, true, false);
                                    if (Av <= pz.L - T) check_div(T + Av, false, false);
                                    if (Av > T) check_div(Av - T, false, true);
                                }
                            }
                        });
                    }
                    for (auto &th : pool) th.join();
                }

                if (found_hit) {
                    // Build new DAG cleanly!
                    unordered_map<u64, Step> defs;
                    vector<u64> kept_ord;
                    for (int u = cur.ks; u < n; u++) {
                        u64 v = cur.nodes[u].v;
                        if (!forb[u]) {
                            defs[v] = {cur.nodes[u].op, cur.nodes[cur.nodes[u].a].v, cur.nodes[cur.nodes[u].b].v, v};
                            kept_ord.push_back(v);
                        }
                    }
                    for (const auto &[nv, st] : best_hit.extra) defs[nv] = st;
                    Dag trial;
                    trial.init(pz.L, pz.s, pz.t);
                    if (trial.from_defs(defs, kept_ord)) {
                        int dlen = (int)trial.nodes.size() - trial.ks;
                        int clen = (int)cur.nodes.size() - cur.ks;
                        if (dlen < clen) {
                            pz_dag_wins++;
                            dag_wins++;
                            vector<Instr> nprog;
                            if (trial.compile_zero_spill(nprog, rng, &kept_ord, 10000) &&
                                (int)nprog.size() < (int)best_prog.size()) {
                                cur = move(trial);
                                best_prog = move(nprog);
                                cur.from_prog(best_prog, &ref_order);
                                pz_cmp_wins++;
                                compile_wins++;
                                improved_this_round = true;
                                cout << "  [WIN] " << pz.id << " T_idx=" << ti << " (" << (int)log2((double)T) + 1 << "b): -> "
                                     << best_prog.size() << " (was " << old_len << ")\n" << flush;
                                break; // restart round on updated DAG
                            } else {
                                cout << "  [SPILL_DROP] " << pz.id << " T_idx=" << ti << " dag=" << dlen << " (was " << clen << ")\n" << flush;
                            }
                        }
                    }
                }
            }
            if (!improved_this_round) break;
        }

        total_new += (int)best_prog.size();
        puz_done++;
        if ((int)best_prog.size() < old_len) {
            puz_imp++;
            out_f << pz.id << " " << best_prog.size() << " " << to_json(best_prog) << "\n" << flush;
        }
        if (pz_dag_wins > 0 || puz_done % 10 == 0 || (now_sec() - tp0) > 2.0) {
            cout << pz.id << ": " << old_len << " -> " << best_prog.size()
                 << " (dag_wins=" << pz_dag_wins << ", cmp_wins=" << pz_cmp_wins
                 << ", s2=" << s2.count << ", dt=" << (now_sec() - tp0) << "s)\n" << flush;
        }
    }
    cout << "DONE puzzles=" << puz_done << " improved=" << puz_imp
         << " dag_wins=" << dag_wins << " cmp_wins=" << compile_wins
         << " total: " << total_old << " -> " << total_new
         << " (diff=" << (total_new - total_old) << ") elapsed=" << (now_sec() - t_all) << "s\n";
    return 0;
}
