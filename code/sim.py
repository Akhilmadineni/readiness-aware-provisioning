"""Plan-instance construction, event-driven provisioning simulator, baselines and Differential Evolution (A5P3)."""
import heapq
import math
import numpy as np
from twin import REGIONS, BACKOFF, TYPES

# dependency template of one platform stack inside a region (prerequisite counts match the dataset)
DEPS = {"vpc_network": [], "subnet": ["vpc_network"], "iam_role": [], "dns_zone": [], "object_bucket": [],
        "k8s_cluster": ["subnet", "iam_role"], "node_pool": ["k8s_cluster"], "managed_db": ["subnet", "iam_role"],
        "cache_cluster": ["subnet"], "iot_hub": ["iam_role"], "stream_topic": ["iot_hub"],
        "load_balancer": ["subnet", "dns_zone"]}
GATED = [t for t in TYPES if DEPS[t]]                       # 8 types that have prerequisites
POWER_KW = {t: 0.5 for t in ["k8s_cluster", "node_pool", "managed_db", "cache_cluster"]}
POWER_KW.update({t: 0.1 for t in ["iot_hub", "load_balancer", "stream_topic"]})
POWER_KW.update({t: 0.02 for t in ["vpc_network", "subnet", "iam_role", "dns_zone", "object_bucket"]})
RUNNER_KW = 0.2          # orchestration runner (central-1) for the whole makespan
FIFO_PARALLELISM = 10    # default parallelism of common IaC engines
TAU_MAX = 6.0            # maximum plan deferral (h)
N_GENES = 1 + len(TYPES) + len(GATED) + len(REGIONS) + 2   # rank weight, type bias, gates, budgets, gcap, tau


RATE = {0: 10, 1: 8, 2: 20, 3: 20, 4: 5, 5: 12}   # region rate limits as recorded in the dataset (req/s)


class Plan:
    """One multi-region IaC plan built from held-out task rows."""

    def __init__(self, rows, rng, twin, tenant_hour_factor, bucket_model, pid):
        self.pid = pid
        nreg = rng.integers(3, 7)
        regs = rng.choice(len(REGIONS), nreg, replace=False)
        stacks = []
        for r in regs:
            stacks += [r] * rng.integers(1, 3)
        rng.shuffle(stacks)
        self.h0 = int(rng.integers(0, 24))
        T = []
        for si, r in enumerate(stacks):
            ids = {}
            for ty in TYPES:
                cand = rows[(rows.region == REGIONS[r]) & (rows.resource_type == ty)]
                row = cand.iloc[rng.integers(0, len(cand))]
                ids[ty] = len(T)
                T.append(dict(region=r, type=TYPES.index(ty), tname=ty, stack=si,
                              a=float(row.api_calls_required), ready=float(row.typical_ready_time_s),
                              k=BACKOFF.index(row.backoff_policy), poll=float(row.readiness_poll_interval_s),
                              L0=float(row.other_tenant_load_rps) / tenant_hour_factor[int(row.hour)],
                              bres=float(row.token_bucket_fill - (bucket_model[0] - bucket_model[1]
                                                                  * row.concurrent_tasks_in_region)),
                              pre=[ids[p] for p in DEPS[ty]]))
        self.T = T
        self.n = len(T)
        self.reg = np.array([t["region"] for t in T]); self.typ = np.array([t["type"] for t in T])
        self.a = np.array([t["a"] for t in T]); self.ready = np.array([t["ready"] for t in T])
        self.k = np.array([t["k"] for t in T]); self.poll = np.array([t["poll"] for t in T])
        self.L0 = np.array([t["L0"] for t in T]); self.bres = np.array([t["bres"] for t in T])
        self.pre = [t["pre"] for t in T]
        self.child = [[] for _ in T]
        for i, p in enumerate(self.pre):
            for j in p:
                self.child[j].append(i)
        self.power = np.array([POWER_KW[t["tname"]] for t in T])
        self.R = np.array([RATE[int(r)] for r in self.reg], float)
        self.lists = [self.reg.tolist(), self.a.tolist(), self.ready.tolist(), self.k.tolist(), self.poll.tolist(),
                      self.L0.tolist(), self.bres.tolist(), self.R.tolist()]
        # base duration (no throttling, ready prerequisites) and upward rank (critical path to plan end)
        base = twin.ttr(self.ready, self.reg, self.k, np.zeros(self.n), self.poll, np.zeros(self.n), np.zeros(self.n))
        self.base = base
        rank = base.copy()
        for i in reversed(range(self.n)):          # template order is topological
            if self.child[i]:
                rank[i] = base[i] + max(rank[c] for c in self.child[i])
        self.rank = rank / rank.max()
        self.regions_used = sorted(set(self.reg.tolist()))


def simulate(plan, twin, thf, bm, prio, gate, budget, gcap, tau, rng=None, trace=False):
    """Event-driven list scheduling (Section V).
    prio: per-task priority (higher first); gate: per-task readiness threshold in [0,1] on prerequisite progress
    (0 = start as soon as the prerequisites are created); budget: per-region admission threshold on the demand
    ratio D (a task is admitted if its D <= budget[r] or the region is idle; inf = no rate awareness);
    gcap: global parallelism; tau: plan deferral in hours; rng=None -> expected-value mode."""
    n = plan.n
    tp = twin.throttle_params
    th0, thD, th1, kap, eps0 = tp["theta0"], tp["thetaD"], tp["theta1"], tp["kappa"], tp["eps0"]
    mk = [tp["m_none"], tp["m_fixed"], 1.0, tp["m_token"]]
    w = [float(v) for v in twin.wt]
    fw = [float(v) for v in twin.fw]
    sig, mu, fm = twin.sig_ln, twin.mu_ln, twin.fail_mult
    emean = math.exp(mu + 0.5 * sig ** 2)
    reg, a, ready, k, poll, L0, bres, R = plan.lists
    pre, child = plan.pre, plan.child
    prio = [float(v) for v in prio]; gate = [float(v) for v in gate]; budget = [float(v) for v in budget]
    thf = [float(v) for v in thf]; b0, b1 = float(bm[0]), float(bm[1])
    start = [-1.0] * n; dur = [0.0] * n
    thr = [0.0] * n; fail = [0.0] * n; hour_s = [0] * n; unr = [0.0] * n
    inflight = [0] * 6; g_in = 0
    ncp = [0] * n
    ev = []
    released = sorted((-prio[i], i) for i in range(n) if not pre[i])
    t = 0.0
    trace_rows = [] if trace else None
    t0h = plan.h0 + tau
    while True:
        if released and g_in < gcap:
            keep = []
            h = int(t0h + t / 3600.0) % 24
            fh = thf[h]
            for item in released:
                i = item[1]; r = reg[i]
                nr = inflight[r] + 1
                D = (a[i] * nr / 60.0 + L0[i] * fh) / R[i]
                if g_in < gcap and (D <= budget[r] or nr == 1):
                    inflight[r] = nr; g_in += 1
                    b = min(1.0, max(0.0, b0 - b1 * nr + bres[i]))
                    x = D - thD
                    lam = (th0 * x ** th1 if x > 0 else 0.0) * mk[k[i]] * max(kap - b, 0.01) + eps0
                    u = 0.0
                    if pre[i]:
                        for j in pre[i]:
                            p = (t - start[j]) / dur[j]
                            u += 1.0 - (p if p < 1.0 else 1.0)
                        u /= len(pre[i])
                    th = lam if rng is None else float(rng.poisson(lam))
                    kk = k[i]
                    z = fw[0] + (fw[1] * th if kk == 0 else 0.0) + fw[2] * th + fw[3] * u \
                        + (fw[4] if ready[i] > 200 else 0.0) + (fw[4 + r] if r >= 1 else 0.0)
                    pf = 1.0 / (1.0 + math.exp(-z))
                    T = w[r] * ready[i] + w[6 + kk] * th + w[10] * poll[i] + w[11] * u \
                        + (w[12] * (D - 1.0) * ready[i] if D > 1.0 else 0.0)
                    if T < 1.0:
                        T = 1.0
                    if rng is None:
                        T = T * emean * (1 + pf * (fm - 1)); f = pf
                    else:
                        T = T * math.exp(rng.normal(mu, sig)); f = 1.0 if rng.random() < pf else 0.0
                        if f:
                            T *= fm
                    start[i] = t; dur[i] = T; thr[i] = th; fail[i] = f; hour_s[i] = h; unr[i] = u
                    c = 5.0 + 2.0 * a[i]
                    heapq.heappush(ev, (t + (c if c < T else T), 1, i)); heapq.heappush(ev, (t + T, 2, i))
                    if trace:
                        trace_rows.append((t, r, D, nr))
                else:
                    keep.append(item)
            released = keep
        if not ev:
            break
        t, kind, i = heapq.heappop(ev)
        if kind == 2:
            inflight[reg[i]] -= 1; g_in -= 1
            if trace:
                trace_rows.append((t, reg[i], float("nan"), inflight[reg[i]]))
        elif kind == 1:
            for c in child[i]:
                ncp[c] += 1
                if ncp[c] == len(pre[c]):
                    rt = max(max(start[j] + min(dur[j], 5.0 + 2.0 * a[j]), start[j] + gate[c] * dur[j])
                             for j in pre[c])
                    if rt <= t + 1e-9:
                        released.append((-prio[c], c)); released.sort()
                    else:
                        heapq.heappush(ev, (rt, 0, c))
        else:
            released.append((-prio[i], i)); released.sort()
    start = np.array(start); dur = np.array(dur)
    cmax = float((start + dur).max())
    ci = twin.ci[plan.reg, np.array(hour_s)]
    co2 = float((plan.power * dur / 3600.0 * ci).sum() / 1000.0)
    hrs = np.arange(int(cmax // 3600) + 1)
    seg = np.minimum(cmax - hrs * 3600.0, 3600.0) / 3600.0
    co2 += float((RUNNER_KW * seg * twin.ci[2, (int(t0h) + hrs) % 24]).sum() / 1000.0)
    hp = np.array([bool(p) for p in pre])
    out = dict(makespan=cmax, throttles=float(sum(thr)), co2=co2, failures=float(sum(fail)),
               latency=cmax + tau * 3600.0, unready=float(np.array(unr)[hp].mean()), tau=float(tau))
    if trace:
        out["trace"] = trace_rows; out["start"] = start; out["dur"] = dur
    return out


# ---------------------------------------------------------------- policies
INF = 1e9


def policy_fifo(plan):
    return dict(prio=-np.arange(plan.n, dtype=float), gate=np.zeros(plan.n), budget=np.full(6, INF),
                gcap=FIFO_PARALLELISM, tau=0.0)


def policy_parmax(plan):
    return dict(prio=-np.arange(plan.n, dtype=float), gate=np.zeros(plan.n), budget=np.full(6, INF),
                gcap=10 ** 6, tau=0.0)


def policy_greedy(plan, theta=1.0, gcap=10 ** 6):
    """Rate-limit-aware greedy: critical-path priority, admit a task only while its demand ratio D <= theta
    (theta and the global parallelism gcap are tuned on training-period plans)."""
    return dict(prio=plan.rank.copy(), gate=np.zeros(plan.n), budget=np.full(6, float(theta)), gcap=int(gcap), tau=0.0)


def decode(x, plan, mask=None):
    """Genome x in [0,1]^N_GENES -> schedule policy (Eq. (7)). mask disables components for the ablation study."""
    mask = mask or {}
    wr = x[0]; tb = x[1:1 + len(TYPES)]
    gg = x[1 + len(TYPES):1 + len(TYPES) + len(GATED)]
    cz = x[1 + len(TYPES) + len(GATED):1 + len(TYPES) + len(GATED) + len(REGIONS)]
    tau = 0.0 if mask.get("no_tau") else float(x[-1] * TAU_MAX)
    gcap = 10 ** 6 if mask.get("no_budget") else max(1, int(round(2.0 ** (7 * x[-2]))))
    prio = wr * plan.rank + (1 - wr) * tb[plan.typ]
    gate = np.zeros(plan.n)
    if not mask.get("no_gate"):
        gmap = {TYPES.index(t): gg[q] for q, t in enumerate(GATED)}
        gate = np.array([gmap.get(ti, 0.0) for ti in plan.typ])
    budget = np.full(6, INF) if mask.get("no_budget") else 2.0 ** (3 * cz - 1.5)
    return dict(prio=prio, gate=gate, budget=budget, gcap=gcap, tau=tau)


def warm_genome(theta=1.0, gcap=10 ** 6):
    """Genome that reproduces the tuned rate-limit-aware greedy schedule."""
    x = np.full(N_GENES, 0.5)
    x[0] = 1.0
    x[1 + len(TYPES) + len(GATED):1 + len(TYPES) + len(GATED) + len(REGIONS)] = (np.log2(theta) + 1.5) / 3
    x[-2] = min(1.0, np.log2(min(gcap, 128)) / 7)
    x[1 + len(TYPES):1 + len(TYPES) + len(GATED)] = 0.0
    x[-1] = 0.0
    return x


def objective(m, ref, wts):
    """J(x), Eq. (6): normalised makespan, throttles, carbon, expected failures, plus deferral penalty."""
    return (m["makespan"] / ref["makespan"] + wts["beta"] * m["throttles"] / ref["throttles"]
            + wts["gamma"] * m["co2"] / ref["co2"] + wts["delta"] * m["failures"] / ref["failures"]
            + wts["omega"] * m["tau"])


def differential_evolution(f, dim, rng, NP=24, G=50, F=0.6, CR=0.9, warm=None, search="de"):
    """Classic DE/rand/1/bin (Storn & Price) on [0,1]^dim with bounce-back repair.
    search='random' replaces variation by uniform random sampling with the same evaluation budget."""
    P = rng.random((NP, dim))
    if warm is not None:
        P[0] = warm
    fit = np.array([f(p) for p in P])
    hist = [fit.min()]
    for g in range(G):
        trials = np.empty_like(P)
        for i in range(NP):
            if search == "random":
                trials[i] = rng.random(dim)
                continue
            a, b, c = rng.choice([j for j in range(NP) if j != i], 3, replace=False)
            v = P[a] + F * (P[b] - P[c])                                   # mutation, Eq. (8)
            v = np.where(v < 0, rng.random(dim) * P[a], v)                 # bounce-back repair
            v = np.where(v > 1, P[a] + rng.random(dim) * (1 - P[a]), v)
            cross = rng.random(dim) < CR                                   # binomial crossover, Eq. (9)
            cross[rng.integers(dim)] = True
            trials[i] = np.where(cross, v, P[i])
        ft = np.array([f(t) for t in trials])
        better = ft <= fit                                                 # greedy selection, Eq. (10)
        P[better] = trials[better]; fit[better] = ft[better]
        hist.append(fit.min())
    b = int(np.argmin(fit))
    return P[b], fit[b], np.array(hist)
