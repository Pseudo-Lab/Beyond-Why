아래는 플레이북(Backdoor/IPW/DR, IV, DiD, 진단·보고 템플릿)을 **바로 구현 가능한 Python 코드 블록**으로 정리한 것이다.

아래 코드는 실제 데이터 스키마/정책 맥락/식별 가정에 따라 수정이 필요하다.

## 1) `causal_toolkit.py` (실무용 최소 모듈 템플릿)

```python
# causal_toolkit.py
# 실무형 인과추론 플레이북에 대응하는 최소 유틸 모듈
# - Backdoor: PS/IPW + Balance + AIPW(이중강건)
# - IV: 2SLS + 진단(약한 도구)
# - DiD: TWFE + 이벤트 스터디용 리드/래그
# - 공통: 클러스터 SE, 리포트 텍스트 템플릿

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------
# 공통: 유틸
# ---------------------------

def _add_intercept(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    if "Intercept" not in X2.columns:
        X2.insert(0, "Intercept", 1.0)
    return X2

def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    """단순 윈저라이징(양끝 p 비율 절단)."""
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def check_binary(series: pd.Series, name: str) -> None:
    vals = set(pd.unique(series.dropna()))
    if not vals.issubset({0, 1, False, True}):
        raise ValueError(f"{name} must be binary 0/1. Got unique values: {sorted(list(vals))}")

def standardized_mean_difference(
    df: pd.DataFrame, treat_col: str, covariates: List[str], weight_col: Optional[str] = None
) -> pd.DataFrame:
    """
    공변량 균형 점검(SMD).
    weight_col=None이면 unweighted SMD.
    """
    check_binary(df[treat_col], treat_col)
    out = []
    for x in covariates:
        t = df[df[treat_col] == 1][x]
        c = df[df[treat_col] == 0][x]

        if weight_col is None:
            mt, mc = t.mean(), c.mean()
            vt, vc = t.var(ddof=1), c.var(ddof=1)
        else:
            wt = df.loc[df[treat_col] == 1, weight_col]
            wc = df.loc[df[treat_col] == 0, weight_col]
            mt = np.average(t, weights=wt)
            mc = np.average(c, weights=wc)
            # 가중 분산(단순 구현)
            vt = np.average((t - mt) ** 2, weights=wt)
            vc = np.average((c - mc) ** 2, weights=wc)

        pooled_sd = np.sqrt(0.5 * (vt + vc))
        smd = (mt - mc) / (pooled_sd + 1e-12)
        out.append({"covariate": x, "mean_treat": mt, "mean_ctrl": mc, "smd": smd})

    return pd.DataFrame(out).sort_values("smd", key=lambda s: np.abs(s), ascending=False)


# ---------------------------
# Backdoor: Propensity Score / IPW / AIPW
# ---------------------------

@dataclass
class PSModelResult:
    ps: pd.Series
    model: object

def fit_propensity_score_logit(
    df: pd.DataFrame,
    treat_col: str,
    covariates: List[str],
    clip: Tuple[float, float] = (0.01, 0.99),
    C: float = 1.0,
    max_iter: int = 2000
) -> PSModelResult:
    """
    성향점수(PS) 추정: sklearn 로지스틱 + 표준화 파이프라인.
    - clip: positivity 완화(극단 PS 클리핑)
    """
    check_binary(df[treat_col], treat_col)
    X = df[covariates].copy()
    y = df[treat_col].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"))
    ])
    pipe.fit(X, y)
    ps = pd.Series(pipe.predict_proba(X)[:, 1], index=df.index, name="ps")
    ps = ps.clip(clip[0], clip[1])
    return PSModelResult(ps=ps, model=pipe)

def make_ipw_weights(
    df: pd.DataFrame,
    treat_col: str,
    ps: pd.Series,
    estimand: str = "ATE",
    stabilize: bool = True,
    trim: Optional[Tuple[float, float]] = None
) -> pd.Series:
    """
    IPW 가중치 생성.
    - estimand: "ATE" or "ATT"
    - stabilize: 안정화 가중치
    - trim: 가중치 상/하위 절단(예: (0.0, 0.99) -> 상위 1% 절단)
    """
    check_binary(df[treat_col], treat_col)
    t = df[treat_col].astype(int)
    ps = ps.reindex(df.index)

    if estimand.upper() == "ATE":
        w = t / ps + (1 - t) / (1 - ps)
        if stabilize:
            p = t.mean()
            w = t * (p / ps) + (1 - t) * ((1 - p) / (1 - ps))
    elif estimand.upper() == "ATT":
        # ATT: treated weight 1, control weight ps/(1-ps)
        w = t * 1.0 + (1 - t) * (ps / (1 - ps))
        if stabilize:
            # 간단 안정화(컨트롤 가중치 평균 1 맞추기)
            w_ctrl = w[t == 0]
            w.loc[t == 0] = w_ctrl / (w_ctrl.mean() + 1e-12)
    else:
        raise ValueError("estimand must be 'ATE' or 'ATT'")

    if trim is not None:
        lo_q, hi_q = trim
        lo = w.quantile(lo_q) if lo_q > 0 else -np.inf
        hi = w.quantile(hi_q) if hi_q < 1 else np.inf
        w = w.clip(lower=lo, upper=hi)

    return pd.Series(w, index=df.index, name="ipw")

@dataclass
class IPWResult:
    effect: float
    se: float
    ci95: Tuple[float, float]
    model_summary: str

def estimate_ate_ipw(
    df: pd.DataFrame,
    outcome_col: str,
    treat_col: str,
    weight_col: str,
    cluster_col: Optional[str] = None
) -> IPWResult:
    """
    IPW로 평균 효과(단순 가중 평균 차이) + 회귀 기반(가중 WLS)로 SE 산출.
    - cluster_col 제공 시 군집-강건 표준오차.
    """
    check_binary(df[treat_col], treat_col)

    d = df[[outcome_col, treat_col, weight_col] + ([cluster_col] if cluster_col else [])].dropna()
    y = d[outcome_col]
    T = d[treat_col].astype(int)
    w = d[weight_col].astype(float)

    # 가중 WLS: y ~ T
    X = _add_intercept(pd.DataFrame({"T": T.values}, index=d.index))
    wls = sm.WLS(y, X, weights=w).fit()

    if cluster_col:
        cov = wls.get_robustcov_results(cov_type="cluster", groups=d[cluster_col]).cov_params()
        se = float(np.sqrt(cov.loc["T", "T"]))
    else:
        se = float(wls.bse["T"])

    eff = float(wls.params["T"])
    ci = (eff - 1.96 * se, eff + 1.96 * se)
    return IPWResult(effect=eff, se=se, ci95=ci, model_summary=str(wls.summary()))

@dataclass
class AIPWResult:
    effect: float
    se: float
    ci95: Tuple[float, float]

def estimate_aipw(
    df: pd.DataFrame,
    outcome_col: str,
    treat_col: str,
    covariates: List[str],
    ps: pd.Series,
    cluster_col: Optional[str] = None
) -> AIPWResult:
    """
    AIPW(이중강건) 추정기.
    - 결과 모형: 선형회귀(실무 최소 템플릿). 비선형은 필요 시 대체.
    - SE: influence function 기반 표준오차(클러스터는 간단 근사; 엄밀 구현은 별도 권장).
    """
    check_binary(df[treat_col], treat_col)
    d = df[[outcome_col, treat_col] + covariates + ([cluster_col] if cluster_col else [])].dropna().copy()
    y = d[outcome_col].astype(float).values
    t = d[treat_col].astype(int).values
    ps = ps.reindex(d.index).astype(float).values
    ps = np.clip(ps, 1e-3, 1 - 1e-3)

    # outcome model: E[Y|T,X] via separate regressions
    X = _add_intercept(d[covariates]).astype(float)

    m1 = sm.OLS(d.loc[t == 1, outcome_col], X.loc[t == 1]).fit()
    m0 = sm.OLS(d.loc[t == 0, outcome_col], X.loc[t == 0]).fit()

    mu1 = m1.predict(X).values
    mu0 = m0.predict(X).values

    # AIPW estimating equation
    phi = (mu1 - mu0) + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    tau_hat = float(np.mean(phi))

    # SE (IID 기준)
    if cluster_col is None:
        se = float(np.std(phi, ddof=1) / np.sqrt(len(phi)))
    else:
        # 간단 클러스터-샌드위치(근사)
        g = d[cluster_col].values
        df_phi = pd.DataFrame({"g": g, "phi": phi})
        cluster_means = df_phi.groupby("g")["phi"].sum()
        se = float(np.sqrt(np.var(cluster_means, ddof=1)) / len(cluster_means))

    ci = (tau_hat - 1.96 * se, tau_hat + 1.96 * se)
    return AIPWResult(effect=tau_hat, se=se, ci95=ci)


# ---------------------------
# IV: 2SLS (statsmodels)
# ---------------------------

@dataclass
class IV2SLSResult:
    effect: float
    se: float
    ci95: Tuple[float, float]
    first_stage_F: Optional[float]
    summary: str

def iv_2sls(
    df: pd.DataFrame,
    outcome_col: str,
    treat_col: str,
    instrument_col: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> IV2SLSResult:
    """
    2SLS 구현(수동):
    1) T ~ Z + X
    2) Y ~ T_hat + X
    약한 도구 진단: 1단계 Z의 F-stat(단순 근사)
    """
    covariates = covariates or []
    cols = [outcome_col, treat_col, instrument_col] + covariates + ([cluster_col] if cluster_col else [])
    d = df[cols].dropna().copy()

    y = d[outcome_col].astype(float)
    T = d[treat_col].astype(float)
    Z = d[instrument_col].astype(float)

    X = d[covariates].copy()
    X1 = _add_intercept(pd.concat([Z.rename("Z"), X], axis=1))
    fs = sm.OLS(T, X1).fit()

    T_hat = fs.predict(X1)
    X2 = _add_intercept(pd.concat([pd.Series(T_hat, index=d.index, name="That"), X], axis=1))
    ss = sm.OLS(y, X2).fit()

    # cluster-robust
    if cluster_col:
        ssr = ss.get_robustcov_results(cov_type="cluster", groups=d[cluster_col])
        se = float(ssr.bse[list(ssr.params.index).index("That")])
        effect = float(ssr.params[list(ssr.params.index).index("That")])
        summary = str(ssr.summary())
    else:
        se = float(ss.bse["That"])
        effect = float(ss.params["That"])
        summary = str(ss.summary())

    ci = (effect - 1.96 * se, effect + 1.96 * se)

    # first-stage F-stat for instrument (simple t^2 if single instrument)
    # (엄밀한 약한 도구 검정은 별도 구현/패키지 권장)  — chatgpt의 의견임
    try:
        tZ = float(fs.tvalues["Z"])
        first_stage_F = tZ ** 2
    except Exception:
        first_stage_F = None

    return IV2SLSResult(
        effect=effect, se=se, ci95=ci, first_stage_F=first_stage_F, summary=summary
    )


# ---------------------------
# DiD / Event Study
# ---------------------------

def make_event_time(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treat_time_col: str
) -> pd.Series:
    """
    event time = time - first_treat_time (개입시점 기준 리드/래그 생성용).
    - treat_time_col: 각 unit의 첫 처치 시점(없으면 NaN)
    """
    # time_col과 treat_time_col은 동일한 스케일(예: week index, date to ordinal 등)이어야 한다.
    et = df[time_col] - df[treat_time_col]
    return et

def add_event_study_dummies(
    df: pd.DataFrame,
    event_time_col: str,
    window: Tuple[int, int] = (-6, 6),
    omit: int = -1,
    prefix: str = "k"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    이벤트 스터디 더미 생성: k=-6..6 중 omit(기준) 제외.
    """
    lo, hi = window
    cols = []
    out = df.copy()
    for k in range(lo, hi + 1):
        if k == omit:
            continue
        col = f"{prefix}{k}"
        out[col] = (out[event_time_col] == k).astype(int)
        cols.append(col)
    return out, cols

@dataclass
class DiDResult:
    effect: float
    se: float
    ci95: Tuple[float, float]
    summary: str

def did_twfe(
    df: pd.DataFrame,
    outcome_col: str,
    treat_col: str,
    unit_col: str,
    time_col: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> DiDResult:
    """
    TWFE: Y ~ T + unit FE + time FE + covariates(optional)
    - staggered adoption에서는 단순 TWFE 해석 이슈가 있을 수 있음 — chatgpt의 의견임
    """
    covariates = covariates or []
    cols = [outcome_col, treat_col, unit_col, time_col] + covariates + ([cluster_col] if cluster_col else [])
    d = df[cols].dropna().copy()

    # 범주형 FE
    d[unit_col] = d[unit_col].astype("category")
    d[time_col] = d[time_col].astype("category")

    rhs = [treat_col] + covariates + [f"C({unit_col})", f"C({time_col})"]
    formula = f"{outcome_col} ~ " + " + ".join(rhs)

    model = smf.ols(formula, data=d).fit()
    if cluster_col:
        rob = model.get_robustcov_results(cov_type="cluster", groups=d[cluster_col])
        # params index가 numpy array일 수 있어 안전 추출
        idx = list(model.params.index).index(treat_col)
        effect = float(rob.params[idx])
        se = float(rob.bse[idx])
        summary = str(rob.summary())
    else:
        effect = float(model.params[treat_col])
        se = float(model.bse[treat_col])
        summary = str(model.summary())

    ci = (effect - 1.96 * se, effect + 1.96 * se)
    return DiDResult(effect=effect, se=se, ci95=ci, summary=summary)


# ---------------------------
# 보고 템플릿 생성(문장 틀)
# ---------------------------

def make_report_paragraph(
    identification: str,
    estimand: str,
    assumptions: List[str],
    diagnostics: List[str],
    effect: float,
    ci95: Tuple[float, float],
    kpi_interpretation: str,
    threats: List[str],
    decision_scope: str
) -> str:
    """
    플레이북 6.1 템플릿을 그대로 생성하는 유틸.
    """
    a = ", ".join(assumptions)
    d = ", ".join(diagnostics)
    th = ", ".join(threats)

    return (
        f"본 분석은 {identification}을 사용하여 {estimand}를 추정하였다. "
        f"핵심 식별 가정은 {a}이며, 점검한 진단은 {d}이다. "
        f"추정된 효과는 {effect:.4g} (95% CI: {ci95[0]:.4g}, {ci95[1]:.4g})로, "
        f"{kpi_interpretation}로 해석된다. "
        f"다만 {th}에 의해 해석이 달라질 수 있으며, 따라서 본 결과는 {decision_scope}로 제한한다."
    )
```

## 2) 사용 예시(더미 시나리오) — Backdoor(IPW/AIPW) + DiD + IV

아래는 “관측자료에서 기능 노출(T)이 매출(Y)에 미친 효과” 같은 전형적 상황을 가정한 호출 예시이다.

```python
import pandas as pd
from causal_toolkit import (
    fit_propensity_score_logit, make_ipw_weights, standardized_mean_difference,
    estimate_ate_ipw, estimate_aipw,
    did_twfe, iv_2sls,
    make_report_paragraph
)

# df columns 예시:
# T: 0/1, Y: float, unit_id, week_id, X1..Xp, cluster_id(optional)
covs = ["X1", "X2", "X3"]

# --- (A) Backdoor: PS -> IPW -> Balance -> Effect
ps_res = fit_propensity_score_logit(df, treat_col="T", covariates=covs, clip=(0.01, 0.99))
df["ps"] = ps_res.ps

df["w_ipw"] = make_ipw_weights(df, treat_col="T", ps=df["ps"], estimand="ATE", stabilize=True, trim=(0.0, 0.99))

smd_before = standardized_mean_difference(df, treat_col="T", covariates=covs, weight_col=None)
smd_after  = standardized_mean_difference(df, treat_col="T", covariates=covs, weight_col="w_ipw")

ipw_res = estimate_ate_ipw(df, outcome_col="Y", treat_col="T", weight_col="w_ipw", cluster_col="cluster_id")
print(ipw_res.effect, ipw_res.ci95)

# --- (B) DR/AIPW (이중강건)
aipw_res = estimate_aipw(df, outcome_col="Y", treat_col="T", covariates=covs, ps=df["ps"], cluster_col="cluster_id")
print(aipw_res.effect, aipw_res.ci95)

# --- (C) DiD (TWFE) - 패널/정책 구조가 있을 때
did_res = did_twfe(
    df, outcome_col="Y", treat_col="T", unit_col="unit_id", time_col="week_id",
    covariates=[], cluster_col="cluster_id"
)
print(did_res.effect, did_res.ci95)

# --- (D) IV (2SLS) - 외생적 변동원 Z가 있을 때
iv_res = iv_2sls(
    df, outcome_col="Y", treat_col="T", instrument_col="Z",
    covariates=covs, cluster_col="cluster_id"
)
print(iv_res.effect, iv_res.ci95, "first-stage F~", iv_res.first_stage_F)

# --- (E) 보고 문단 템플릿
paragraph = make_report_paragraph(
    identification="Backdoor + IPW",
    estimand="ATE",
    assumptions=["교환가능성(조건부)", "positivity", "일관성(consistency)", "간섭 없음(근사)"],
    diagnostics=["PS overlap 확인", "가중치 극단치 점검", "공변량 균형(SMD) 전/후"],
    effect=ipw_res.effect,
    ci95=ipw_res.ci95,
    kpi_interpretation="예: 주간 매출이 평균적으로 증가/감소",
    threats=["미측정 교란 가능성", "측정 체계 변경 가능성"],
    decision_scope="단기 운영 의사결정 참고"
)
print(paragraph)
```

## 3) 플레이북 관점에서 “필수 추가 블록” (실무 안전장치)

아래는 실무에서 자주 필요한데 빠지기 쉬운 보조 함수들이다.

### 3.1 Overlap/가중치 폭발 경고(간단)

```python
def overlap_diagnostics(ps: pd.Series, treat: pd.Series) -> dict:
    treat = treat.astype(int)
    ps_t = ps[treat == 1]
    ps_c = ps[treat == 0]
    return {
        "ps_treat_minmax": (float(ps_t.min()), float(ps_t.max())),
        "ps_ctrl_minmax":  (float(ps_c.min()), float(ps_c.max())),
        "overlap_region": (
            float(max(ps_t.min(), ps_c.min())),
            float(min(ps_t.max(), ps_c.max()))
        ),
        "frac_ps_lt_0p05": float((ps < 0.05).mean()),
        "frac_ps_gt_0p95": float((ps > 0.95).mean())
    }
```

### 3.2 Placebo 테스트(개입 시점/가짜 결과) — 틀만 제공

```python
def placebo_by_shifting_treatment(
    df: pd.DataFrame, treat_time_col: str, shift: int
) -> pd.DataFrame:
    """
    처치시점을 임의로 이동시켜(shift) 가짜 처치 생성하는 템플릿.
    - time index가 정수(week)일 때 사용 가정.
    """
    out = df.copy()
    out["treat_time_placebo"] = out[treat_time_col] + shift
    return out
```