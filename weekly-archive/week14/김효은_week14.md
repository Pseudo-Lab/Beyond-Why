## * 학습 내용 요약
### 통제집단합성법(SCT; synthetic control method)
---

- 이중차분법 한계점: 기간 T보다 실험 대상 N이 상대적으로 적은 경우 잘 작동하지 못 함
    - → 통제집단합성법 제안 (통제 집단 합성법은 실험 대상이 적어도 기간이 길다면 잘 동작할 수 있음)
- **기본 아이디어: 대조군의 결과를 결합하여 처치가 없었을 때 실험군과 비슷한 가상의 대조군(synthetic control) 만듦**
    1. **처치 전 기간**의 대조군의 결과를 특성으로 사용하여 실험군의 결과 예측하는 회귀모델 적합 **(train)**
        - 모델 학습할 때 X 데이터의 형태는 (일자 * 대조군 관측 대상) 꼴
        - ㄴ 대조군 관측 대상 대비 일자수가 많을수록 학습에 유리 (=일자수=데이터수)
        - ㄴ 대조군 관측 대상 대비 일자수가 적을수록 학습에 불리 (over-fitting 우려)
    2. **처치 후 기간**의 대조군 결과를 입력으로 하여 **실험군에 대한 가상의 대조군 만듦 (predict)**
        1. 처치 전 데이터를 이용해 적합한 모델로 처치 후 기간에 대해 예측
    3. 실험군에 대한 처치 후 결과(=관측치)와 합성 대조군 결과(=예측치)의 차이로 ATT 추정
- (참고 1.) 가상의 대조군이 잘 구성된다면 반사실 E[Y0|D=1]과 완벽히 겹치므로 **평행 추세 가정 불필요**
- (참고 2.) 통제집단합성법을 수평회귀분석이라는 별칭으로 불리기도 함
    - 일반적인 회귀분석: 수직적 접근
        - **행 = 개별 대상**
        - 열 = 개별 대상의 특징 (← 가중치의 대상)
    - 통제집단합성법: 수평적 접근
        - 행 = 일자
        - **열 = 개별 대상 (← 가중치의 대상)**

```python
# STEP 1. 학습을 위한 데이터 가공
# 인구수 대비 다운로드수로 인구수에 따른 결과 정규화 진행
df = df.assign(app_download_pct = df["app_download"] / df["population"] * 100)

co_city = list(df.query("treated == 1")["city"].unique()) # 50개 중 3개 도시만 처치군
df_pivot = df.pivot(
	  index="date", # index: 날짜
	  columns="city", # column: 도시
    values="app_download_pct" # value: 앱다운로드 비율
)

df_co = df_pivot.drop(columns=co_city) # 대조군 피봇
df_tr = df_pivot[co_city] # 실험군 피봇

# 대조군
y_co_pre = df_co[df_pivot.index < tr_start] # 처치 전 기간
y_co_post = df_co[df_pivot.index >= tr_start] # 처치 후 기간

# 실험군
y_tr_pre = df_tr[df_pivot.index < tr_start] # 처치 전 기간
y_tr_post = df_tr[df_pivot.index >= tr_start] # 처치 후 기간

# STEP 2. 모델 학습
from skelarn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
model = model.fit(X=y_co_pre, y=y_tr_pre.mean(axis=1))
# model.coef_: 대조군 도시별로 가중치 계산됨

# STEP 3. 합성대조군
# 실험군에 대한 가상의 대조군 (실험군이 처치받지 않았을 때 예측 결과)
y0_tr_hat = model.predict(y_co_post)

# STEP 4. ATT 추정
att = y_tr_post.mean(axis=1) - model.predict(y_co_post) # 일자별 ATT

# STEP 5. ATT 분석
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 3))
ax = sns.lineplot(x="date", y="att", data=df_att)
plt.axhline(0, color='red', linestyle='--')
plt.axvline(datetime.strptime("2022-05-01", "%Y-%m-%d"), color='blue', linestyle='--')
plt.xticks(rotation=90)
plt.show()

# 처치 전 기간에 대해서는 ATT = 0 (적합한 회귀모델의 정확도를 가늠할 수 있음)
# 처치 초반에는 ATT 증가 추이 -> 시간이 지날수록 하락 추이: 신기효과
```
<img width="600" height="424" alt="image" src="https://github.com/user-attachments/assets/8ff6d297-ca92-4a06-beae-647249ed1e10" />


### 표준(canonical) 통제집단합성법
---
- **표준 통제집단합성법(synthetic control method)은 회귀모델에 두 가지 제약조건 부여**
    - 제약조건 1. 가중치는 모두 양수이거나 0 &&
    - 제약조건 2. 가중치의 합은 1
- **제약조건의 목적**: 가상의 대조군이 실험군에 대한 볼록 조합(convex combination)이 되도록 하여 **외삽 피하기 위함**
    - 실험군이 대조군에 속한 모든 대상의 결과보다 대조군이 크거나 작다면
    - 표준공식으로는 E[Y0|D=1]을 구하기 위한 가상의 대조군을 만들 수 없음
    - → 즉, 실험군과 대조군이 매우 다르기 때문에 SCM을 시도하지 말라는 의미 (일종의 보호 장치)
- **제약조건의 추가 효과**: 최적화 문제에 대한 희소(sparse) 해결책
    - 가중치가 0인 대상이 많음
    - → 실험군에 대한 가상의 대조군을 만들 때 소수의 도시만 활용됨
- (노트) 볼록 조합(convex combination)의 뜻은?
    - 여러 개의 말뚝(대조군들)을 박아두고 그 둘레를 고무줄로 감쌌을 때 **고무줄 안쪽 영역**이 바로 볼록 조합으로 만들 수 있는 범위입니다. 가상의 대조군(Synthetic Control)이 이 '고무줄 영역' 안에서만 만들어지게 강제하는 것이 바로 볼록 조합의 핵심입니다. (gemini)

$$
\hat{w}^{sc}=\argmin_{w} \Vert \bar{y}_{pre, tr}-Y_{pre,co}w_{co} \Vert \text{ s.t. } \sum_{w}w_{i}= 1 \text{ and } w_{i} \ge 0 \forall{i}
$$

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted)
import cvxpy as cp

class SyntheticControlMethod(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, y_co_pre, y_tr_pre):
        y_co_pre, y_tr_pre = check_X_y(y_co_pre, y_tr_pre)

        w = cp.Variable(y_co_pre.shape[1])

        objective = cp.Minimize(cp.sum_squares(y_co_pre@w - y_tr_pre))
        constraints = [cp.sum(w) == 1, w >= 0]

        problem = cp.Problem(objective, constraints)

        self.loss_ = problem.solve(verbose=False)
        self.w_ = w.value
        self.is_fitted = True
        return self
    
    def predict(self, y_co_post):
        check_is_fitted(self)
        y_co_post = check_array(y_co_post)

        return y_co_post @ self.w_
        
model = SyntheticControlMethod()
model.fit(y_co_pre, y_tr_pre.mean(axis=1))
model.w_.round(3)
"""
array([-0.   , -0.   , -0.   , -0.   , -0.   , -0.   ,  0.076,  0.037,
        0.083,  0.01 , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ,
        0.061,  0.123,  0.008,  0.074, -0.   ,  0.   , -0.   , -0.   ,
       -0.   , -0.   , -0.   , -0.   , -0.   ,  0.   , -0.   ,  0.092,
       -0.   , -0.   ,  0.   ,  0.046,  0.089,  0.   ,  0.067,  0.061,
        0.   , -0.   , -0.   ,  0.088,  0.   ,  0.086, -0.   ])
"""
```

### 통제집단합성법과 공변량
---

- 일반적인 SCM:
    - 모델 학습: (처치 전 기간) 대조군의 결과(Y) → 실험군의 결과(Y) 학습
    - 모델 예측: (처치 후 기간) 대조군의 결과(Y) → 실험군의 결과(Y) 예측 (=합성 통제집단)
- 제안하고자 하는 방법:
    - **실험군의 결과를 예측할 때 대조군의 결과 뿐만 아니라 공변량을 활용하자!**
    - 모델 학습: (처치 전 기간) 대조군의 결과(Y), **피처(X)** → 실험군의 결과(Y) 학습
    - 모델 예측: (처치 후 기간) 대조군의 결과(Y), **피처(X)** → 실험군의 결과(Y) 예측 (=합성 통제집단)
- 제안 방법 문제점: 척도 불일치에 따른 최적화 왜곡
    - 표준 통제집단합성법에서는 가중치 제약조건(=개별 W ≥ 0, 합=1)이 존재하기 때문에
    - 결과와 피처 사이의 척도가 다르다면
    - 각 변수가 손실함수에 기여하는 가중치가 왜곡되어 특정 변수의 영향력이 지배적이거나 무시될 수 있음
- 문제점 해결: **(X, Y) 각각에 곱해지는 가중치 행렬(V) 추가 학습**
    - 오차를 최소화하는 V와 W를 찾음 
    
$$
\hat{w}^{sc}=\argmin_{w} \Vert \bar{y}_{pre, tr}-\sum{v_{k}^{*}X_{k,pre,co}w_{co}}\Vert \text{ s.t. } \sum_{w}w_{i}= 1 \text{ and } w_{i} > 0 \forall{i}
$$


### 통제집단합성법과 편향제거
---

- 문제: 처치 이전 기간의 수가 적다면 과적합 발생하기 쉬움
- **방법: cross fitting (데이터를 K개의 조각으로 나눠서 K-1개로 학습 / 1개로는 편향 구해서 예측값 보정)**
- 기존:
    - (X=y_co_pre, y=y_tr_pre) 모델 학습
    - ATT = y_co_post - model.preidct(y_tr_post)
- 변경:
    1. 학습 데이터(X=y_co_pre, y=y_tr_pre)를 K개의 조각으로 나눔
    2. K-1개의 데이터로는 선형회귀 모델 적합
    3. 1개의 데이터로는 bias 측정 (bias = y_tr_pre.loc[hold_out]- model.predict(y_co_pre.loc[hold_out])
    4. ATT 추정 = (y_tr_post - model.preidct(y_co_post)) - **bias**


### 통제집단합성법과 추론
---

- 목적: ATT 추정값에 대한 신뢰구간을 구하고 싶음
- 방법: (cross-fitting 진행했다는 것을 가정으로 함)
    - cross-fitting 기반으로 구해진 여러개의 ATT를 기반으로 표준오차 추정값(=SE)구함
    - 검정통계량 = ATT 추정량 / SE 구한 뒤 T분포 사용하여 신뢰구간 구할 수 있음

 
### 합성 이중차분법(SDID; Synthetic Difference In Difference)
---

- 이중차분법 복습

$$
\hat{\tau}^{did}=\argmin_{\mu, \alpha, \beta, \tau}{{ \sum_{n=1}^{N} \sum_{t=1}^{T}(Y_{it}-(\mu+\alpha_{i}+\beta_{t}+\tau W_{it}))^2}}
$$
        
- 통제집단합성법 복습
	- 아래의 수식에 따라 가중회귀분석을 수행했을 때 회귀계수와
	- 위에서 다루었던 통제집단합성법 로직에 따라 구해진 ATT가 동일하다는 것을 보여주어
	- 아래 수식이 맞는 이유에 대해 보여주고 있음

$$
\hat{\tau}^{sc}=\argmin_{\beta, \tau}{\sum_{n=1}^{N}{\sum_{t=1}^{T}{(Y_{it}-(\beta_{t}+\tau W_{it}))^2 \hat{\omega_{i}}^{sc}}}}
$$

- 합성이중차분법의 아이디어
    - 이중차분법: 대상고정효과 O, 시간고정효과 O, **실험대상에 대한 가중치 X**
    - 통제집단합성법: **대상고정효과 X**, 시간고정효과 O, 실험대상에 대한 가중치 O
    - → 합성이중차붑법: **대상고정효과 O**, 시간고정효과 O, **실험대상 가중치** O + **시간가중치 O (추가)**
- 합성이중차분법 방법
    - 1) 실험대상에 대한 가중치 구함
        - 적합 모델: (처치 전 기간 대조군) → (처치 전 기간 실험군) 모델 적합
        - X 데이터 형태: (일자, 대조군 개체별 Y)
        - Y 데이터 형태: (일자, 평균 Y)
        - 특이 사항: 절편이동 비허용 (= 외삽 비허용)
    - 2) 시간가중치 구함
        - 모델 적합: (처치 전 기간 대조군) → (처치 후 기간 대조군)
        - X 데이터 형태: (대조군 개체, 일자별 Y)
        - Y 데이터 형태: (대조군 개체, 평균 Y)
        - 특이사항: 절편이동 허용 (= 외삽 허용)
    - 3) 합성이중차분법 모델 적합
        - 모델 적합: (treated, post, treated*post) → Y
        - 가중치: 개체가중치 * 시간가중치

```python
df_scdid = (
    df_norm
        .assign(tr_post = lambda d: d["post"] * d["treated"])
        .merge(unit_w, on="city", how="left")
        .merge(time_w, on="date", how="left")
        .fillna({"unit_weight": df_norm["treated"].mean(), "time_weight": df_norm["treated"].mean()})
        .assign(weight = lambda d: d["unit_weight"] * d["time_weight"])
)

model = smf.wls(
    "app_download_pct ~ treated * post",
    data = df_scdid.query("weight >= 1e-10"),
    weights = df_scdid.query("weight >= 1e-10")["weight"]
)

model = model.fit()
print(model.params["treated:post"])

```

- 합성이중차분법 장점
    - E[Y(0)|D=1]에 대한 가상의 대조군을 만들어 이중차분법에 필요한 평행추세가정이 훨씬 타당해짐
    - 이중차분법을 사용하여 통제집단합성법은 실험군의 추세를 파악하는데 집중할 수 있음
    - **이중차분법, 통제집단합성법 대비 추정량의 편향과 분산이 적은 경향**
