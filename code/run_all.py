"""A5P3 - Readiness-aware scheduling for multi-region infrastructure provisioning under API rate limits.

Runs everything: pre-processing, digital-twin calibration and fidelity check, plan-instance construction,
baselines (FIFO, parallel-max, rate-limit-aware greedy), the proposed Differential Evolution (DE) scheduler,
ablation, carbon-weight sensitivity, statistics, all result tables (results/*.csv, results/metrics.json)
and all figures (figures/*.png at 300 dpi + results/*.jpg).
Usage:  python3 run_all.py          (about 10-15 min on one CPU core)
"""
import os, sys, json, time, shutil, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin import load_and_clean, Twin, REGIONS, BACKOFF, TYPES, TARGETS   # noqa: E402
from sim import (Plan, simulate, policy_fifo, policy_parmax, policy_greedy, decode, warm_genome, objective,  # noqa
                 differential_evolution, N_GENES, GATED, TAU_MAX)

DATA = os.path.join(HERE, "..", "data", "A5P3_multiregion_provisioning.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_TEST, N_TRAIN_PLANS = 10, 6
NP, GEN, F_DE, CR_DE = 20, 50, 0.6, 0.9
MC = 30                                   # Monte-Carlo replicates per policy evaluation
WTS = dict(beta=0.5, gamma=0.5, delta=0.5, omega=0.15)
GAMMAS = [0.0, 0.5, 1.5, 3.0]
METHODS = ["FIFO", "Parallel-max", "RL-greedy", "DE (proposed)"]
LOG = []

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 100})
COL = {"FIFO": "#8c8c8c", "Parallel-max": "#b8b8b8", "RL-greedy": "#5a5a5a", "DE (proposed)": "#1f6fb4",
       "Random search": "#d08c2a"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


T0 = time.time()
# ============================================================ 1. data + pre-processing
d, train, clean_log = load_and_clean(DATA)
log("cleaning:", clean_log)
dtr, dte = d[train], d[~train]
hm = dtr.groupby("hour").other_tenant_load_rps.median()
THF = (hm / hm.mean()).to_numpy()                      # tenant-load hour factor
A = np.column_stack([np.ones(len(dtr)), dtr.concurrent_tasks_in_region])
bco = np.linalg.lstsq(A, dtr.token_bucket_fill, rcond=None)[0]
BM = (float(bco[0]), float(-bco[1]))                   # token bucket b = b0 - b1 * n_r (+ row residual)

N_COLS = len(pd.read_csv(DATA, nrows=1).columns)
summary = [("Tasks / columns", f"{len(d):,} / {N_COLS}"),
           ("Plans / workloads", f"{d.plan_id.nunique():,} / {d.workload.nunique()}"),
           ("Regions / resource types / backoff policies", f"{d.region.nunique()} / {d.resource_type.nunique()} / "
                                                           f"{d.backoff_policy.nunique()}"),
           ("Period (train / test)", "2025 / 2026 H1"),
           ("Rows train / test", f"{train.sum():,} / {(~train).sum():,}"),
           ("Rate limit range (req/s)", f"{d.region_rate_limit_rps.min()}-{d.region_rate_limit_rps.max()}"),
           ("Missing % (3 columns)", "1.3-2.2"),
           ("Throttled tasks (errors>0) %", f"{100 * (d.throttle_errors > 0).mean():.1f}"),
           ("Time-to-ready median [IQR] (s)", f"{d.time_to_ready_s.median():.0f} [{d.time_to_ready_s.quantile(.25):.0f}, "
                                              f"{d.time_to_ready_s.quantile(.75):.0f}]"),
           ("Failure rate %", f"{100 * d.provisioning_failed.mean():.1f}"),
           ("Carbon intensity range (gCO2/kWh)", f"{d.grid_carbon_gco2_kwh.min():.0f}-{d.grid_carbon_gco2_kwh.max():.0f}")]
pd.DataFrame(summary, columns=["item", "value"]).to_csv(os.path.join(RES, "table1_dataset_summary.csv"), index=False)
reg_tab = d.groupby("region").agg(rate_limit=("region_rate_limit_rps", "first"),
                                  carbon=("grid_carbon_gco2_kwh", "mean"), throttles=("throttle_errors", "mean"),
                                  fail_rate=("provisioning_failed", "mean"), ttr=("time_to_ready_s", "median"))
reg_tab.to_csv(os.path.join(RES, "region_summary.csv"))
log(reg_tab.round(3).to_string())

# ============================================================ 2. digital twin + fidelity on 2026 hold-out
tw = Twin().fit(dtr)
log("throttle params:", tw.throttle_params); log("ttr params:", tw.ttr_params); log("fail params:", tw.fail_params)
k_te = dte.backoff_policy.map(BACKOFF.index).to_numpy(); r_te = dte.region.map(REGIONS.index).to_numpy()
lam_te = tw.lam(dte.demand, k_te, dte.token_bucket_fill)
ttr_twin = tw.ttr(dte.typical_ready_time_s.to_numpy(), r_te, k_te, lam_te, dte.readiness_poll_interval_s.to_numpy(),
                  dte.unready.to_numpy(), dte.demand.to_numpy()) * np.exp(tw.mu_ln)
pf_twin = tw.pfail(lam_te, k_te, dte.unready.to_numpy(), dte.typical_ready_time_s.to_numpy(), r_te)
ttr_twin = ttr_twin * (1 + pf_twin * (tw.fail_mult - 1))
FEATS = ["prereq_count", "prereq_ready_fraction", "api_calls_required", "plan_batch_size",
         "concurrent_tasks_in_region", "region_rate_limit_rps", "other_tenant_load_rps", "token_bucket_fill",
         "readiness_poll_interval_s", "readiness_poll_calls", "typical_ready_time_s", "demand", "hour",
         "region", "resource_type", "backoff_policy", "workload"]
assert not set(FEATS) & set(TARGETS)


def lgb_X(x):
    X = x[FEATS].copy()
    for c in ["region", "resource_type", "backoff_policy", "workload"]:
        X[c] = pd.Categorical(X[c], categories=sorted(d[c].unique()))
    return X


lgbp = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, num_threads=1,
            verbose=-1, random_state=0)
keep = dtr.ttr_outlier == 0
m_ttr = lgb.LGBMRegressor(**lgbp).fit(lgb_X(dtr[keep]), np.log(dtr.time_to_ready_s[keep]))
m_thr = lgb.LGBMRegressor(objective="poisson", **lgbp).fit(lgb_X(dtr), dtr.throttle_errors)
m_fail = lgb.LGBMClassifier(**lgbp).fit(lgb_X(dtr), dtr.provisioning_failed)
y_l = np.log(dte.time_to_ready_s.to_numpy())
fid = []
for name, pl, pt, pfp in [("Type median (typical_ready_time)", np.log(dte.typical_ready_time_s.to_numpy()), None, None),
                          ("Digital twin (Eqs. 2-4)", np.log(ttr_twin), lam_te, pf_twin),
                          ("LightGBM (black box)", m_ttr.predict(lgb_X(dte)), m_thr.predict(lgb_X(dte)),
                           m_fail.predict_proba(lgb_X(dte))[:, 1])]:
    row = dict(model=name, r2_log_ttr=r2_score(y_l, pl), mae_log_ttr=mean_absolute_error(y_l, pl))
    if pt is not None:
        row.update(throttle_auc=roc_auc_score(dte.throttle_errors > 0, pt),
                   throttle_mae=mean_absolute_error(dte.throttle_errors, pt),
                   fail_pr_auc=average_precision_score(dte.provisioning_failed, pfp),
                   fail_roc_auc=roc_auc_score(dte.provisioning_failed, pfp))
    fid.append(row)
fid = pd.DataFrame(fid)
fid.to_csv(os.path.join(RES, "twin_fidelity.csv"), index=False)
log(fid.round(3).to_string())

# fig 2: twin fidelity
fig, ax = plt.subplots(2, 1, figsize=(3.3, 4.2))
hb = ax[0].hexbin(np.log10(ttr_twin), np.log10(dte.time_to_ready_s), gridsize=40, cmap="Blues", mincnt=1, bins="log")
lo, hi = 0, 4.6
ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
ax[0].set_xlabel("Twin prediction, log$_{10}$ T (s)"); ax[0].set_ylabel("Observed, log$_{10}$ T (s)")
ax[0].set_title(f"(a) Readiness time, 2026 rows ($R^2$={fid.r2_log_ttr[1]:.2f})")
bins = np.array([0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 50])
cb = pd.cut(dte.demand, bins)
g = pd.DataFrame({"b": cb, "obs": dte.throttle_errors, "twin": lam_te}).groupby("b", observed=True)
mid = [f"{iv.left:g}-{iv.right:g}" for iv in g.mean().index]
x = np.arange(len(mid))
ax[1].bar(x - 0.2, g.obs.mean(), 0.4, color="#b8b8b8", label="observed", yerr=g.obs.sem(), capsize=1.5,
          error_kw=dict(lw=0.6))
ax[1].bar(x + 0.2, g.twin.mean(), 0.4, color=COL["DE (proposed)"], label="twin $\\lambda$, Eq. (2)")
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(mid, rotation=40, ha="right")
ax[1].set_xlabel("Demand ratio $D$ (-)"); ax[1].set_ylabel("Throttle errors / task")
ax[1].set_title("(b) Throttling vs demand ratio"); ax[1].legend(frameon=False)
fig.tight_layout(); savefig(fig, "fig2_twin_fidelity")

# ============================================================ 3. plan instances + greedy tuning
rng_plan = np.random.default_rng(2026)
test_plans = [Plan(dte, rng_plan, tw, THF, BM, i) for i in range(N_TEST)]
rng_tr = np.random.default_rng(2025)
train_plans = [Plan(dtr, rng_tr, tw, THF, BM, i) for i in range(N_TRAIN_PLANS)]
pd.DataFrame([dict(plan=p.pid, tasks=p.n, regions=len(p.regions_used), submit_hour=p.h0,
                   heavy_tasks=int(np.isin(p.typ, [TYPES.index(t) for t in ["k8s_cluster", "node_pool", "managed_db",
                                                                                "cache_cluster"]]).sum()))
              for p in test_plans]).to_csv(os.path.join(RES, "test_plans.csv"), index=False)
log("test plan sizes:", [p.n for p in test_plans])


def run(p, pol, rng=None):
    return simulate(p, tw, THF, BM, **pol, rng=rng)


grid = []
for th in [0.75, 1.0, 1.5, 2.0, 2.5]:
    for gc in [5, 10, 20, 40, 128]:
        Js = []
        for p in train_plans:
            ref = run(p, policy_fifo(p))
            Js.append(objective(run(p, policy_greedy(p, th, gc)), ref, WTS))
        grid.append(dict(theta=th, gcap=gc, J=np.mean(Js)))
grid = pd.DataFrame(grid); grid.to_csv(os.path.join(RES, "greedy_tuning_train_plans.csv"), index=False)
best = grid.loc[grid.J.idxmin()]
G_TH, G_GC = float(best.theta), int(best.gcap)
log(f"tuned greedy: theta={G_TH} gcap={G_GC} J_train={best.J:.3f}")


# ============================================================ 4. evaluation helpers
def mc_eval(p, pol, seed):
    rng = np.random.default_rng(10_000 + 97 * seed + p.pid)
    ms = [run(p, pol, rng) for _ in range(MC)]
    return {k: float(np.mean([m[k] for m in ms])) for k in ["makespan", "throttles", "co2", "failures", "latency",
                                                            "unready", "tau"]}


def de_solve(p, seed, wts=WTS, mask=None, warm=True, search="de"):
    ref = run(p, policy_fifo(p))
    f = lambda x: objective(run(p, decode(x, p, mask)), ref, wts)  # noqa: E731
    rng = np.random.default_rng(1000 * seed + p.pid)
    xb, fb, hist = differential_evolution(f, N_GENES, rng, NP=NP, G=GEN, F=F_DE, CR=CR_DE,
                                          warm=warm_genome(G_TH, G_GC) if warm else None, search=search)
    return xb, fb, hist


rows, hists, genomes = [], {}, []
ref_mc = {}
t_de = []
for s in SEEDS:
    for p in test_plans:
        pols = {"FIFO": policy_fifo(p), "Parallel-max": policy_parmax(p), "RL-greedy": policy_greedy(p, G_TH, G_GC)}
        t1 = time.time()
        xb, fb, hist = de_solve(p, s)
        t_de.append(time.time() - t1)
        hists[("DE (proposed)", s, p.pid)] = hist
        pols["DE (proposed)"] = decode(xb, p)
        genomes.append(dict(seed=s, plan=p.pid, **{f"g{i}": v for i, v in enumerate(xb)}))
        for mth, pol in pols.items():
            m = mc_eval(p, pol, s)
            if mth == "FIFO":
                ref_mc[(s, p.pid)] = m
            m.update(method=mth, seed=s, plan=p.pid, tasks=p.n, J=objective(m, ref_mc[(s, p.pid)], WTS))
            rows.append(m)
    log(f"seed {s} done, elapsed {time.time() - T0:.0f}s")
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "main_per_plan_seed.csv"), index=False)
pd.DataFrame(genomes).to_csv(os.path.join(RES, "de_best_genomes.csv"), index=False)
for c in ["makespan", "latency"]:
    res[c + "_min"] = res[c] / 60
for c in ["makespan", "throttles", "co2", "failures"]:
    res[c + "_rel"] = res[c] / res.apply(lambda r: ref_mc[(r.seed, r.plan)][c], axis=1)

per_seed = res.groupby(["method", "seed"]).mean(numeric_only=True)
agg = per_seed.groupby("method").agg(["mean", "std"])
cols = ["makespan_min", "latency_min", "throttles", "failures", "co2", "unready", "J", "makespan_rel",
        "throttles_rel", "failures_rel", "co2_rel", "tau"]
main = pd.DataFrame({c: [f"{agg.loc[m, (c, 'mean')]:.3f} ± {agg.loc[m, (c, 'std')]:.3f}" for m in METHODS]
                     for c in cols}, index=METHODS)
main.to_csv(os.path.join(RES, "table3_main_comparison.csv"))
log(main.to_string())

# statistics: Wilcoxon signed-rank, DE vs each baseline on the 50 (plan, seed) pairs
stats = []
de = res[res.method == "DE (proposed)"].set_index(["seed", "plan"])
for b in METHODS[:-1]:
    bb = res[res.method == b].set_index(["seed", "plan"])
    for c in ["J", "makespan", "throttles", "failures", "co2", "latency"]:
        diff = de[c] - bb.loc[de.index, c]
        st = wilcoxon(de[c], bb.loc[de.index, c])
        stats.append(dict(baseline=b, metric=c, median_rel_change_pct=float(100 * np.median(diff / bb.loc[de.index, c])),
                          de_better_pairs=int((diff < 0).sum()), n=len(diff), p_value=float(st.pvalue)))
stats = pd.DataFrame(stats); stats.to_csv(os.path.join(RES, "wilcoxon_de_vs_baselines.csv"), index=False)
log(stats.to_string())

# ============================================================ 5. ablation (same budget, same seeds)
ABL = [("Full DE (proposed)", dict(mask=None)), ("w/o readiness gate", dict(mask={"no_gate": True})),
       ("w/o rate budget + parallelism", dict(mask={"no_budget": True})), ("w/o deferral", dict(mask={"no_tau": True})),
       ("w/o greedy warm start", dict(mask=None, warm=False)), ("Random search (same budget)", dict(mask=None,
                                                                                                    search="random"))]
arows = []
for s in SEEDS:
    for p in test_plans:
        for name, kw in ABL:
            if name == "Full DE (proposed)":
                m = res[(res.method == "DE (proposed)") & (res.seed == s) & (res.plan == p.pid)].iloc[0].to_dict()
            else:
                xb, fb, hist = de_solve(p, s, mask=kw.get("mask"), warm=kw.get("warm", True),
                                        search=kw.get("search", "de"))
                if name.startswith("Random"):
                    hists[("Random search", s, p.pid)] = hist
                m = mc_eval(p, decode(xb, p, kw.get("mask")), s)
                m["J"] = objective(m, ref_mc[(s, p.pid)], WTS)
            arows.append(dict(variant=name, seed=s, plan=p.pid, **{k: m[k] for k in ["J", "makespan", "throttles",
                                                                                      "failures", "co2", "latency",
                                                                                      "unready"]}))
    log(f"ablation seed {s} done, elapsed {time.time() - T0:.0f}s")
abl = pd.DataFrame(arows)
abl.to_csv(os.path.join(RES, "ablation_per_plan_seed.csv"), index=False)
abl["makespan_min"] = abl.makespan / 60; abl["latency_min"] = abl.latency / 60
aps = abl.groupby(["variant", "seed"]).mean(numeric_only=True).groupby("variant").agg(["mean", "std"])
order = [a[0] for a in ABL]
abl_tab = pd.DataFrame({c: [f"{aps.loc[v, (c, 'mean')]:.3f} ± {aps.loc[v, (c, 'std')]:.3f}" for v in order]
                        for c in ["J", "makespan_min", "throttles", "failures", "co2", "latency_min", "unready"]},
                       index=order)
abl_tab.to_csv(os.path.join(RES, "table4_ablation.csv"))
log(abl_tab.to_string())

# ============================================================ 6. carbon-weight sensitivity (seed 0)
srows = []
for gm in GAMMAS:
    w = dict(WTS, gamma=gm)
    for p in test_plans:
        if gm == WTS["gamma"]:
            m = res[(res.method == "DE (proposed)") & (res.seed == 0) & (res.plan == p.pid)].iloc[0].to_dict()
        else:
            xb, fb, _ = de_solve(p, 0, wts=w)
            m = mc_eval(p, decode(xb, p), 0)
        srows.append(dict(gamma=gm, plan=p.pid, **{k: m[k] for k in ["makespan", "throttles", "failures", "co2",
                                                                      "latency", "tau"]}))
sens = pd.DataFrame(srows); sens.to_csv(os.path.join(RES, "gamma_sensitivity_per_plan.csv"), index=False)
sens_m = sens.groupby("gamma").mean(numeric_only=True).drop(columns="plan")
sens_m.to_csv(os.path.join(RES, "gamma_sensitivity.csv"))
log(sens_m.round(3).to_string())
log(f"sensitivity done, elapsed {time.time() - T0:.0f}s")

# ============================================================ 7. figures
# fig 3: convergence (J normalised per plan by its own final DE value is not used: raw J, mean over plans, +-std seeds)
fig, ax = plt.subplots(figsize=(3.3, 2.5))
gx = np.arange(GEN + 1) * NP + NP
for nm in ["DE (proposed)", "Random search"]:
    H = np.array([[hists[(nm, s, p.pid)] for p in test_plans] for s in SEEDS]).mean(axis=1)
    ax.plot(gx, H.mean(0), color=COL[nm], lw=1.4, label=nm)
    ax.fill_between(gx, H.mean(0) - H.std(0), H.mean(0) + H.std(0), color=COL[nm], alpha=0.2, lw=0)
for b, ls in [("FIFO", ":"), ("RL-greedy", "--"), ("Parallel-max", "-.")]:
    Jb = np.mean([objective(run(p, pol(p)), run(p, policy_fifo(p)), WTS) for p in test_plans
                  for pol in [{"FIFO": policy_fifo, "Parallel-max": policy_parmax,
                               "RL-greedy": lambda q: policy_greedy(q, G_TH, G_GC)}[b]]])
    ax.axhline(Jb, color=COL[b] if b != "Parallel-max" else "#777777", ls=ls, lw=1, label=f"{b} ({Jb:.2f})")
ax.set_xlabel("Simulator evaluations"); ax.set_ylabel("Best objective $J$ (-), Eq. (6)")
ax.legend(frameon=False, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.62)); ax.set_xlim(gx[0], gx[-1])
fig.tight_layout(); savefig(fig, "fig3_convergence")

# fig 4: comparison bars (relative to FIFO) with error bars over seeds
fig, axs = plt.subplots(2, 2, figsize=(3.3, 3.3))
for a, (c, lab) in zip(axs.ravel(), [("makespan_min", "Makespan (min)"), ("throttles", "Throttle errors / plan"),
                                     ("failures", "Failed tasks / plan"), ("co2", "Carbon (kg CO$_2$ / plan)")]):
    mu = [agg.loc[m, (c, "mean")] for m in METHODS]; sd = [agg.loc[m, (c, "std")] for m in METHODS]
    a.bar(range(4), mu, yerr=sd, color=[COL[m] for m in METHODS], capsize=2, error_kw=dict(lw=0.7))
    a.set_xticks(range(4)); a.set_xticklabels(["FIFO", "PM", "RG", "DE"], fontsize=7.5, rotation=40, ha="right",
                      rotation_mode="anchor")
    a.set_ylabel(lab, fontsize=8)
    for i, v in enumerate(mu):
        a.text(i, v * 1.03, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    a.set_ylim(0, max(mu) * 1.3)
fig.tight_layout(); savefig(fig, "fig4_comparison")

# fig 5: admission demand ratio over time for one plan (deterministic run of seed-0 policies)
p5 = max(test_plans, key=lambda q: q.n)
xb5 = pd.DataFrame(genomes).query("seed == 0 and plan == @p5.pid").iloc[0, 2:].to_numpy(float)
fig, axs = plt.subplots(3, 1, figsize=(3.3, 3.5), sharex=True)
cmap = plt.get_cmap("tab10")
tmax = 0
for a, (nm, pol) in zip(axs, [("FIFO", policy_fifo(p5)), ("RL-greedy", policy_greedy(p5, G_TH, G_GC)),
                              ("DE (proposed)", decode(xb5, p5))]):
    m = simulate(p5, tw, THF, BM, **pol, trace=True)
    tr = np.array([r for r in m["trace"] if not np.isnan(r[2])])
    for r in p5.regions_used:
        q = tr[tr[:, 1] == r]
        a.scatter(q[:, 0] / 60, q[:, 2], s=5, color=cmap(r), label=REGIONS[r], alpha=0.8, lw=0)
    a.axhline(tw.throttle_params["thetaD"], color="k", ls="--", lw=0.7)
    a.set_yscale("log"); a.set_ylabel("$D$ at start (-)")
    a.set_title(f"{nm}: makespan {m['makespan'] / 60:.0f} min, {m['throttles']:.0f} throttles, "
                f"deferral {m['tau']:.1f} h", fontsize=7.5)
    tmax = max(tmax, m["makespan"] / 60)
axs[-1].set_xlabel("Time since plan start (min)")
fig.subplots_adjust(bottom=0.2)
h_, l_ = axs[-1].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7, ncol=3, loc="lower center", markerscale=1.8, handletextpad=0.1,
           columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); savefig(fig, "fig5_demand_trace")

# fig 6: gamma sensitivity: carbon vs makespan / latency, baselines as reference points
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(sens_m.makespan / 60, sens_m.co2, "o-", color=COL["DE (proposed)"], lw=1.2, ms=4, label="DE makespan")
ax.plot(sens_m.latency / 60, sens_m.co2, "s--", color=COL["DE (proposed)"], lw=1, ms=3.5, mfc="white",
        label="DE latency (incl. deferral)")
OFFS = {0.0: (-16, 6), 0.5: (-26, -11), 1.5: (-8, 6), 3.0: (-12, 6)}
for gm, r in sens_m.iterrows():
    ax.annotate(f"$\\gamma$={gm:g}", (r.latency / 60, r.co2), textcoords="offset points", xytext=OFFS.get(gm, (-6, 6)),
                fontsize=7)
r0 = res[res.seed == 0].groupby("method").mean(numeric_only=True)
for b, mk in [("FIFO", "^"), ("Parallel-max", "v"), ("RL-greedy", "D")]:
    ax.scatter(r0.loc[b, "makespan"] / 60, r0.loc[b, "co2"], marker=mk, color=COL[b], edgecolor="k", lw=0.4, s=28,
               label=b, zorder=3)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Carbon (kg CO$_2$ / plan)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); savefig(fig, "fig6_gamma_sensitivity")

# ============================================================ 8. flowchart (Graphviz)
dot_src = os.path.join(HERE, "fig1_flowchart.dot")
subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot_src, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
shutil.copy(dot_src, os.path.join(RES, "fig1_flowchart.dot"))
from PIL import Image  # noqa: E402
Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"),
                                                                        quality=92)

# ============================================================ 9. metrics.json
setup = [("Twin training / plan rows", "2025 rows / 2026 H1 rows"), ("Test plans", f"{N_TEST} ({min(p.n for p in test_plans)}"
                                                                                       f"-{max(p.n for p in test_plans)} tasks)"),
         ("DE", f"DE/rand/1/bin, NP={NP}, G={GEN}, F={F_DE}, CR={CR_DE}"), ("Genome", f"{N_GENES} genes in [0,1]"),
         ("Weights", "beta={beta}, gamma={gamma}, delta={delta}, omega={omega}/h".format(**WTS)),
         ("Deferral bound", f"tau <= {TAU_MAX} h"), ("Greedy (tuned)", f"theta={G_TH}, gcap={G_GC}"),
         ("Evaluation", f"{MC} Monte-Carlo runs x 5 seeds")]
pd.DataFrame(setup, columns=["item", "value"]).to_csv(os.path.join(RES, "table2_setup.csv"), index=False)
metrics = dict(cleaning=clean_log, throttle_params=tw.throttle_params, ttr_params=tw.ttr_params,
               fail_params=tw.fail_params, bucket_model=dict(b0=BM[0], b1=BM[1]),
               twin_fidelity=fid.to_dict(orient="records"), greedy_tuned=dict(theta=G_TH, gcap=G_GC),
               main={m: {c: dict(mean=float(agg.loc[m, (c, "mean")]), std=float(agg.loc[m, (c, "std")]))
                         for c in cols} for m in METHODS},
               wilcoxon=stats.to_dict(orient="records"),
               ablation={v: {c: dict(mean=float(aps.loc[v, (c, "mean")]), std=float(aps.loc[v, (c, "std")]))
                             for c in ["J", "makespan_min", "throttles", "failures", "co2", "latency_min", "unready"]}
                         for v in order},
               gamma_sensitivity=sens_m.reset_index().to_dict(orient="records"),
               de_time_per_plan_s=dict(mean=float(np.mean(t_de)), max=float(np.max(t_de))),
               test_plan_sizes=[p.n for p in test_plans], fig5_plan=int(p5.pid),
               config=dict(seeds=SEEDS, NP=NP, G=GEN, F=F_DE, CR=CR_DE, MC=MC, weights=WTS, tau_max=TAU_MAX),
               runtime_s=time.time() - T0)
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1)
log(f"total runtime {time.time() - T0:.0f}s")
open(os.path.join(RES, "run_log.txt"), "w").write("\n".join(LOG))
