"""Data pre-processing and calibration of the provisioning 'digital twin' (A5P3).

The twin consists of four sub-models fitted on the 2025 training rows only:
  (a) throttle model   lambda = theta0 * (D - thetaD)_+^theta1 * m_k * (kappa - b) + eps0   (Poisson MLE)
  (b) readiness model  E[T] = a_r * ready_s + beta_k * throttles + c * poll + d * (1 - rho) + e * (D - 1)_+ * ready_s
  (c) failure model    logistic regression on throttle / readiness / region terms
  (d) carbon profile   median grid intensity by region x hour-of-day
Target columns are never used as inputs of a model that predicts them.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor, LogisticRegression

REGIONS = ["green-north-1", "green-north-2", "central-1", "central-2", "south-1", "east-1"]
BACKOFF = ["none", "fixed", "exponential_jitter", "token_aware"]
TYPES = ["vpc_network", "subnet", "iam_role", "dns_zone", "object_bucket", "k8s_cluster", "node_pool",
         "managed_db", "cache_cluster", "iot_hub", "stream_topic", "load_balancer"]
TARGETS = ["throttle_errors", "time_to_ready_s", "provisioning_failed"]


def load_and_clean(path):
    d = pd.read_csv(path, parse_dates=["timestamp"])
    log = {}
    d["hour"] = d.timestamp.dt.hour
    train = d.timestamp < "2026-01-01"
    # outlier handling (thresholds from the training period only)
    hi = d.loc[train, "other_tenant_load_rps"].quantile(0.995)
    log["tenant_load_winsor_rps"] = float(hi)
    log["tenant_load_clipped"] = int((d.other_tenant_load_rps > hi).sum())
    d["other_tenant_load_rps"] = d.other_tenant_load_rps.clip(upper=hi)
    # imputation: tenant load by region x business-hours median, bucket fill by median, carbon by region x hour median
    d["biz"] = ((d.hour >= 9) & (d.hour <= 17)).astype(int)
    for c, keys in [("other_tenant_load_rps", ["region", "biz"]), ("token_bucket_fill", ["region"]),
                    ("grid_carbon_gco2_kwh", ["region", "hour"])]:
        log[f"missing_{c}"] = int(d[c].isna().sum())
        med = d[train].groupby(keys)[c].median()
        fill = d[keys].apply(tuple, axis=1).map(med) if len(keys) > 1 else d[keys[0]].map(med)
        d[c] = d[c].fillna(fill)
    d["token_bucket_fill"] = d.token_bucket_fill.clip(0, 1)
    d["prereq_ready_fraction"] = d.prereq_ready_fraction.clip(0, 1)
    d["demand"] = (d.api_calls_required * d.concurrent_tasks_in_region / 60 + d.other_tenant_load_rps) \
        / d.region_rate_limit_rps
    d["unready"] = (1 - d.prereq_ready_fraction) * (d.prereq_count > 0)
    # time_to_ready outliers: robust z on log residual vs type median (training thresholds)
    lr = np.log(d.time_to_ready_s) - np.log(d.typical_ready_time_s)
    med, mad = lr[train].median(), (lr[train] - lr[train].median()).abs().median()
    d["ttr_outlier"] = ((lr - med).abs() / (1.4826 * mad) > 4.5).astype(int)
    log["ttr_outliers_flagged"] = int(d.ttr_outlier.sum())
    return d, train.to_numpy(), log


class Twin:
    def fit(self, d):
        self._fit_throttle(d)
        self._fit_ttr(d)
        self._fit_fail(d)
        self.ci = d.groupby(["region", "hour"]).grid_carbon_gco2_kwh.median().unstack().loc[REGIONS].to_numpy()
        return self

    # ---------------- (a) throttle model ----------------
    def _lam(self, p, D, k, b):
        th0, thD, th1 = np.exp(p[0]), p[1], np.exp(p[2])
        m = np.concatenate([[np.exp(p[3]), np.exp(p[4]), 1.0, np.exp(p[5])]])[k]
        return th0 * np.clip(D - thD, 0, None) ** th1 * m * np.clip(p[6] - b, 0.01, None) + np.exp(p[7])

    def _fit_throttle(self, d):
        D, b = d.demand.to_numpy(), d.token_bucket_fill.to_numpy()
        k = d.backoff_policy.map({x: i for i, x in enumerate(BACKOFF)}).to_numpy()
        y = d.throttle_errors.to_numpy()

        def nll(p):
            lam = self._lam(p, D, k, b)
            return np.mean(lam - y * np.log(lam))
        p0 = np.array([np.log(5), 0.7, np.log(1.5), np.log(2), np.log(1.4), np.log(0.7), 1.3, np.log(0.05)])
        r = minimize(nll, p0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-7))
        r = minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-9))
        self.pt = r.x
        self.throttle_params = dict(theta0=float(np.exp(r.x[0])), thetaD=float(r.x[1]), theta1=float(np.exp(r.x[2])),
                                    m_none=float(np.exp(r.x[3])), m_fixed=float(np.exp(r.x[4])), m_expjit=1.0,
                                    m_token=float(np.exp(r.x[5])), kappa=float(r.x[6]), eps0=float(np.exp(r.x[7])))

    def lam(self, D, k, b):
        return self._lam(self.pt, np.asarray(D, float), np.asarray(k), np.asarray(b, float))

    # ---------------- (b) readiness-time model ----------------
    def _ttr_X(self, ready, reg, k, thr, poll, unready, D):
        n = len(ready)
        X = np.zeros((n, 6 + 4 + 3))
        X[np.arange(n), reg] = ready
        X[np.arange(n), 6 + k] = thr
        X[:, 10] = poll
        X[:, 11] = unready
        X[:, 12] = np.clip(D - 1, 0, None) * ready
        return X

    def _fit_ttr(self, d):
        m = (d.provisioning_failed == 0) & (d.ttr_outlier == 0)
        s = d[m]
        X = self._ttr_X(s.typical_ready_time_s.to_numpy(), s.region.map(REGIONS.index).to_numpy(),
                        s.backoff_policy.map(BACKOFF.index).to_numpy(), s.throttle_errors.to_numpy(),
                        s.readiness_poll_interval_s.to_numpy(), s.unready.to_numpy(), s.demand.to_numpy())
        h = HuberRegressor(epsilon=1.5, alpha=0.0, fit_intercept=False, max_iter=2000).fit(X, s.time_to_ready_s)
        self.wt = h.coef_
        pred = np.clip(X @ self.wt, 1, None)
        # multiplicative noise: robust spread of the log ratio
        lr = np.log(s.time_to_ready_s.to_numpy() / pred)
        self.mu_ln = float(np.median(lr))
        self.sig_ln = float(1.4826 * np.median(np.abs(lr - np.median(lr))))
        # failure slow-down multiplier estimated on failed (non-outlier) tasks
        f = d[(d.provisioning_failed == 1) & (d.ttr_outlier == 0)]
        Xf = self._ttr_X(f.typical_ready_time_s.to_numpy(), f.region.map(REGIONS.index).to_numpy(),
                         f.backoff_policy.map(BACKOFF.index).to_numpy(), f.throttle_errors.to_numpy(),
                         f.readiness_poll_interval_s.to_numpy(), f.unready.to_numpy(), f.demand.to_numpy())
        self.fail_mult = float(np.exp(np.median(np.log(f.time_to_ready_s.to_numpy() / np.clip(Xf @ self.wt, 1, None)))
                                      - self.mu_ln))
        self.ttr_params = dict(perf=dict(zip(REGIONS, map(float, self.wt[:6]))),
                               backoff_pen=dict(zip(BACKOFF, map(float, self.wt[6:10]))), poll=float(self.wt[10]),
                               unready=float(self.wt[11]), overload=float(self.wt[12]), mu_ln=self.mu_ln,
                               sigma_ln=self.sig_ln, fail_mult=self.fail_mult)

    def ttr(self, ready, reg, k, thr, poll, unready, D):
        X = self._ttr_X(np.atleast_1d(ready), np.atleast_1d(reg), np.atleast_1d(k), np.atleast_1d(thr),
                        np.atleast_1d(poll), np.atleast_1d(unready), np.atleast_1d(D))
        return np.clip(X @ self.wt, 1, None)

    # ---------------- (c) failure model ----------------
    def _fail_X(self, thr, k, unready, ready, reg):
        thr = np.atleast_1d(thr).astype(float)
        X = np.column_stack([thr * (np.atleast_1d(k) == 0), thr, np.atleast_1d(unready),
                             (np.atleast_1d(ready) > 200).astype(float)]
                            + [(np.atleast_1d(reg) == j).astype(float) for j in range(1, 6)])
        return X

    def _fit_fail(self, d):
        X = self._fail_X(d.throttle_errors.to_numpy(), d.backoff_policy.map(BACKOFF.index).to_numpy(),
                         d.unready.to_numpy(), d.typical_ready_time_s.to_numpy(), d.region.map(REGIONS.index).to_numpy())
        self.lrf = LogisticRegression(C=100.0, max_iter=2000).fit(X, d.provisioning_failed)
        self.fail_params = dict(intercept=float(self.lrf.intercept_[0]),
                                coef=dict(zip(["thr_x_none", "thr", "unready", "slow_type"] + REGIONS[1:],
                                              map(float, self.lrf.coef_[0]))))
        self.fw = np.concatenate([[self.lrf.intercept_[0]], self.lrf.coef_[0]])

    def pfail(self, thr, k, unready, ready, reg):
        X = self._fail_X(thr, k, unready, ready, reg)
        z = self.fw[0] + X @ self.fw[1:]
        return 1 / (1 + np.exp(-z))
