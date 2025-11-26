# Chapter 9. Synthetic Control Method (통제집단합성법)

궁금한점 / 논의점  
1. SCM하면 왜 평행추세가정이 필요없어짐?
2. 처치 샘플이 하나여도 왜 잘 작동함?

## Ⅰ. 개요
패널데이터셋에 널리 사용되는 이중차분법이랑 다른 방법!  

통제집단합성법(Synthetic Control Method, SCM):  
실제 대조군을 convex combination(가중합) 하여 가상의 대조군(synthetic control)을 만드는 방법.
  - **처치가 없을 때의 실험군 역할**을 함 
  - 평행 추세 가정이 필요없어짐 
  - 처치 샘플이 하나여도 잘 작동함 (?)

Abadie & Gardeazabal (2003), Abadie, Diamond & Hainmueller (2010)의 연구로 널리 사용되며,
단일 처치 대상(single treated unit)에 대해 시간에 따른 처치효과(ATT)를 추정하는 데 특히 적합하다.

핵심 아이디어:  
> 1. 관측된 대조군들의 적절한 가중치 조합으로  
> 2. 처치 전(pre-treatment) 기간 동안 실험군의 경로를 가장 잘 모방하는 synthetic unit을 만든 뒤,  
> 3. 처치 후(post-treatment) 차이를 ATT로 해석한다.

예시: 온라인 마케팅 데이터셋
  - Y: 일별 앱 다운로드 수
  - D(처치): 마케팅 캠페인 집행 여부
  - X: 도시 인구, 도시별 처치 여부, 등
  - !각 지역별 인구가 달라서 시장 간 비교가 어렵다면, 결과를 시장 규모에 따라 나눠서 정규화 해도 된다.

Notation
* $D$: treatment
* $t$: time
* $T$: 기간의 수
  * $T_{\text{pre}}$: 개입 이전
  * $T_{\text{post}}$: 개입 이후
* 처치여부, 더미변수 조합:
  $$W_{it} = D_i \cdot \text{Post}_t$$

목표: ATT 추정
$$
ATT = E[Y \mid D = 1, \text{Post} = 1] - E[Y(0) \mid D = 1, \text{Post} = 1]
$$

* 우항 우변은 조건부 기대가 아니라 반사실적 결과 $Y(0)$이므로 직접 관찰할 수 없다.
* 이를 과거 데이터를 사용해 synthetic control로 근사하는 것이 SCM의 목적이다.

# Ⅱ. 행렬 표현

* $i = 1$: 실험 대상(예: 특정 캠페인 지역)
* $i = 2, \dots, J+1$: 대조군 지역
* $t = 1, \dots, T_0$: 처치 전 기간
* $t = T_0+1, \dots, T$: 처치 후 기간

### 1. Outcome 행렬

$$
Y =
\begin{bmatrix}
Y_{1,1} & Y_{1,2} & \cdots & Y_{1,T} \\
Y_{2,1} & Y_{2,2} & \cdots & Y_{2,T} \\
\vdots  & \vdots  &        & \vdots \\
Y_{J+1,1} & Y_{J+1,2} & \cdots & Y_{J+1,T}
\end{bmatrix}
$$

### 2. 처치 전 outcome 벡터

$$
Y_{1}^{\text{pre}} = (Y_{1,1}, \dots, Y_{1,T_0})'
$$

### 3. 대조군들의 처치 전 outcome 행렬

$$
Y_0^{\text{pre}} =
\begin{bmatrix}
Y_{2,1} & \cdots & Y_{2,T_0} \\
\vdots  &        & \vdots \\
Y_{J+1,1} & \cdots & Y_{J+1,T_0}
\end{bmatrix}
$$

Synthetic control의 목표는 다음과 같다.

> $$Y_0^{\text{pre}}\omega \approx Y_1^{\text{pre}}$$
> 를 만족시키는 convex weight $\omega$를 찾는 것.

# Ⅲ. 통제집단합성법과 수평 회귀분석

SCM은 아래와 같은 constrained regression으로 해석된다.

### 실험 대상 pre-period outcome을 대조군 가중합으로 근사

$$
Y_1^{\text{pre}} \approx Y_0^{\text{pre}} \omega.
$$

여기서 가중치 벡터는 $\omega = (\omega_2, \dots, \omega_{J+1})'$이다.

# Ⅳ. 표준 SCM: 두 가지 제약조건

표준 SCM은 다음 제약을 포함한다.

1. 가중치 양수성

$$
\omega_j \ge 0 \quad \forall j
$$

2. 가중치 합 1

$$
\sum_j \omega_j = 1
$$

### 최적화 목적함수

$$
\min_{\omega} ; |Y_1^{\text{pre}} - Y_0^{\text{pre}}\omega|_2^2
$$

subject to
$$
\omega_j \ge 0, \quad \sum_j \omega_j = 1.
$$

# Ⅴ. SCM의 핵심 가정

### (1) No anticipation

처치 전에는 처치효과가 존재하지 않는다.

### (2) Convex combination 가정

실험군의 counterfactual이 대조군 convex combination으로 근사 가능해야 한다.

$$
Y_{1,t}(0)
= \sum_{j=2}^{J+1} \omega_j^{*} Y_{j,t}(0) + \varepsilon_t,
\quad t > T_0
$$

### (3) 구조 안정성

pre-period에서 잘 맞는 조합이 post-period에서도 유지되어야 한다.

### (4) 공통 충격 가정

시점별 충격이 실험군과 대조군 모두에 동일하게 반영되어야 한다.

# Ⅵ. SCM과 공변량(Covariates)

공변량이 포함된 확장형 SCM에서는 다음을 추가로 balance한다.

* 실험군 공변량: $Z_1$
* 대조군 공변량 행렬: $Z_0$

최적화 문제는 다음과 같다.

$$
\min_{\omega, V}
; (Y_1^{\text{pre}} - Y_0^{\text{pre}}\omega)'
V (Y_1^{\text{pre}} - Y_0^{\text{pre}}\omega),
$$

여기서 $V$는 diagonal weighting matrix이다.

# Ⅶ. SCM과 편향 제거 (Bias Reduction)

Cross-fitting을 사용하여 overfitting과 extrapolation bias를 줄인다.

### 1. pre-period를 $K$개 블록으로 분할

$$
{1,\dots,T_0} = \bigcup_{k=1}^{K} B_k.
$$

### 2. 각 블록 $B_k$를 hold-out으로 두고

나머지 기간만 사용하여 $\omega^{(-k)}$ 추정.

### 3. hold-out 블록에서 예측오차 계산

### 4. 모든 fold에서 평균화

효과:

* 사전(overfitting) fit 감소
* extrapolation 안정성 증가
* bias 완화

# Ⅷ. 추론(Inference)

SCM은 전통적 asymptotic 기반 추론이 어렵다.
따라서 다음의 non-parametric 방식이 사용된다.

### (1) Placebo Test / Permutation Inference

가짜 treated unit을 생성하여 분포 비교.

### (2) Rank-based inference

RMSPE 순위 기반 p-value 계산.

### (3) Conformal inference / Sensitivity analysis

보다 robust한 신뢰구간 구성 가능.

# Ⅸ. 합성 이중차분법 (Synthetic DID)

Arkhangelsky et al. (2021)이 제안한 Synthetic DID(SDID)는
SCM과 DID를 결합한 hybrid estimator이다.

### (1) 시간 가중치 + 단위 가중치를 동시에 최적화

* SCM: 단위 가중치 $\omega$
* DID: 시간평균
* SDID: 시간 가중치 $\lambda$와 단위 가중치 $\omega$ 둘 다 학습

### (2) Pre-period balance 강화

SCM보다 안정적이며 DID보다 균형을 잘 맞춘다.

### (3) ATT 식별

$$
Y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \varepsilon_{it}
$$

여기서 SCM적 가중치가 $\alpha_i, \gamma_t$ 추정을 안정화한다.

# Ⅹ. 결론

Synthetic Control Method는

* 단일 실험군 환경에서 강력한 counterfactual 추정법이며
* convex combination 기반 구조 덕분에
* 관측 데이터 기반 인과추론에서 핵심적 도구이다.

또한
covariate balancing,
cross-fitting,
placebo inference,
synthetic DID
등과 결합되며 현대 causal inference에서 중요한 위치를 차지하고 있다.