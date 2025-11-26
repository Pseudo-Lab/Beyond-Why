# 7. Meta-learner 기반 CATE 추정 정리

## 0. Meta-learner 개요

**Meta-learner**는 원래 목적이 *결과 예측*인 일반 머신러닝 모델  
(예: \( \hat{\mu}(X) \approx E[Y \mid X] \))을 재조합하여,

> “누가 처치를 받았을 때 얼마나 이득을 보는가?”  
> 즉, **개별 처치 효과(CATE, Conditional Average Treatment Effect)**  
> \( \tau(x) = E[Y(1) - Y(0) \mid X = x] \)

를 추정하기 위한 프레임워크이다.

### 핵심 특징

- **Outcome modeling 접근**  
  - 잔차처리(orthogonalization)나 IPW(역확률 가중) 없이도  
    \( E[Y \mid X, T] \)를 잘 예측하는 모델을 통해 CATE를 간접 복원.
- **표현 학습(Representation learning)에 강함**  
  - 비선형 관계, 고차원 공변량(high-dimensional covariates), 복잡한 상호작용 등을  
    일반 ML 모델(XGBoost, RF, NN 등)로 캡처한 뒤,  
    이를 인과 추론에 재활용.
- **FWL/직교화 포함 여부**  
  - T-learner, X-learner, S-learner는 순수 outcome modeling 중심.
  - R-learner(이중/편향 제거 머신러닝)는 잔차 기반 직교화(orthogonalization)를 명시적으로 포함.

### 전제: 비교란성(비편향성) 가정

Meta-learner 기반 CATE 해석이 인과적으로 의미 있으려면  
**비교란성(ignorability/unconfoundedness)** 가 반드시 성립해야 한다.

> 모든 교란 요인이 \(X\)에 포함되어 있어  
> \((Y(0), Y(1)) \perp T \mid X\) 가 성립한다고 가정.

연속 처치의 경우, 적절한 정규성 조건 아래에서

\[
\frac{\partial}{\partial t} E[Y(t) \mid X]
=
\frac{\partial}{\partial t} E[Y \mid T = t, X]
\]

와 같이, 관측 데이터에서의 조건부 기댓값의 기울기를  
처치 반응 함수의 기울기로 해석할 수 있다.

---

## 7.1 이산형 처치 Meta-learner

이산형(이진) 처치: \(T \in \{0, 1\}\)  
예: 마케팅 이메일 발송 여부, 쿠폰 발급 여부 등

기본 표기:
- \(Y\) : 결과 (구매, 전환, 매칭 여부 등)
- \(T\) : 처치(0/1)
- \(X\) : 공변량(고객/사용자 특성)
- \(\mu_1(x) = E[Y \mid X=x, T=1]\)
- \(\mu_0(x) = E[Y \mid X=x, T=0]\)
- \( \tau(x) = E[Y(1) - Y(0) \mid X=x] \) (CATE)

---

### 7.1.1 T-learner (Two-model approach)

#### 정의

T-learner는 가장 단순한 Meta-learner로,  
**처치군과 대조군을 완전히 분리하여 두 개의 예측 모델을 학습**한다.

1. 처치군(T=1)에 대한 outcome 모델
   \[
   \hat{\mu}_1(x) \approx E[Y \mid X=x, T=1]
   \]
2. 대조군(T=0)에 대한 outcome 모델
   \[
   \hat{\mu}_0(x) \approx E[Y \mid X=x, T=0]
   \]
3. CATE 추정
   \[
   \hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
   \]

#### 장점

- 구조가 매우 단순하고 구현이 용이하다.
- 처치군/대조군의 데이터 생성 메커니즘이 달라도  
  각 군에 **서로 다른 모델·파라미터**를 사용해 유연하게 학습 가능하다.

#### 단점

- 두 모델을 **완전히 독립적으로 학습**하기 때문에,
  - 두 예측 오차가 CATE 추정에 그대로 누적 → **분산(variance) 커짐**.
- 처치군/대조군 표본 크기가 크게 불균형한 경우,
  - 표본이 적은 쪽 모델이
    - 과적합(overfitting)되거나,
    - 자기 정규화(self-regularization)로 지나치게 단순해져(underfitting),
    - 결과적으로 **bias가 증가**하는 문제가 발생.
- 특히 **희귀 처치(rare treatment)** 상황에서 불안정하다.

#### 사용 권장 상황

- 전체 표본 크기가 충분히 크고,
- \(T=0, T=1\) 비율이 심하게 치우치지 않은 경우,
- **Baseline CATE 모델**로 적합하다.

---

### 7.1.2 X-learner (Cross-learning approach)

T-learner의 **표본 불균형 및 분산 문제를 완화**하기 위해 고안된 Meta-learner.

핵심 아이디어:

> 잠재적 결과를 **cross-impute**하여  
> 개별 pseudo-treatment effect를 생성하고,  
> 이를 한 번 더 스무딩하여 CATE를 학습한다.

#### 절차

1. **1단계: T-learner 방식으로 outcome 모델 학습**
   - \(\hat{\mu}_1(X)\), \(\hat{\mu}_0(X)\) 추정

2. **2단계: pseudo-treatment effect 생성**

   - 처치군(T=1)에서:
     \[
     D_1 = Y - \hat{\mu}_0(X)
     \approx Y(1) - E[Y(0)\mid X]
     \]
   - 대조군(T=0)에서:
     \[
     D_0 = \hat{\mu}_1(X) - Y
     \approx E[Y(1)\mid X] - Y(0)
     \]

   여기서 \(D_1, D_0\)는 각 개체별 “추정 개별 효과”에 해당하는  
   **pseudo-treatment effect**로 볼 수 있다.

3. **3단계: pseudo-effect를 outcome으로 하는 두 개의 CATE 모델 학습**

   - 처치군 기반:
     \[
     \hat{\tau}_1(x) \approx E[D_1 \mid X=x, T=1]
     \]
   - 대조군 기반:
     \[
     \hat{\tau}_0(x) \approx E[D_0 \mid X=x, T=0]
     \]

4. **4단계: 성향점수 기반 가중 결합**

   - 성향점수(Propensity score) 추정:
     \[
     \hat{e}(X) = \hat{P}(T=1 \mid X)
     \]
   - 최종 CATE:
     \[
     \hat{\tau}(x)
     =
     w(x)\hat{\tau}_1(x) + (1-w(x))\hat{\tau}_0(x)
     \]
     - \(w(x)\)는 보통 \(\hat{e}(x)\), \(1 - \hat{e}(x)\),  
       혹은 표본 비율에 기반한 weight 등으로 설계
     - 데이터가 많은 쪽에서 추정된 \(\hat{\tau}\)에 더 큰 가중을 부여하도록 선택하는 것이 일반적

#### 직관

- 먼저 T-learner로 대략적인 반사실(counterfactual)을 채워넣고,
- 각 군에서 “개별 효과 비슷한 값(D₁, D₀)”을 만든 뒤,
- 이를 **효과 함수 τ(x)**로 스무딩 → 분산 감소, 안정성 증가.
- 특히 **처치군/대조군 표본 크기가 크게 다른 경우**,  
  데이터가 풍부한 군에서 얻은 정보를 더 신뢰하도록 설계되어 있다.

#### Domain Adaptation Learner (간단 메모)

- X-learner의 변형 버전.
- 성향점수 대신 \(1 / P(T=t)\) 형태 등,  
  IPW(역확률 가중, Inverse Probability Weighting)에 가까운 가중치 구조를 사용하여  
  집단 간 분포 차이를 추가적으로 보정하려는 확장형 방법.
- 실무/교재에서는 보통 **X-learner가 기본**, DAL은 마이너 확장으로 취급한다.

---

## 7.2 연속형 처치 Meta-learner

연속형 처치: \( T \in \mathbb{R} \)  
예: 할인율, 광고 노출 횟수, 용량(dose), 사용시간 등

이 경우 CATE는 보통 **한계 효과(marginal effect)** 형태로 정의:

\[
\tau(x) = \frac{\partial}{\partial T} E[Y \mid X=x, T]
\]

Meta-learner는 이 **dose–response 관계**를  
일반 예측 모델로부터 복원하는 틀을 제공한다.

---

### 7.2.1 S-learner (Single-model / Functional S-learner)

#### 정의

S-learner는 처치 T를 별도 모델로 나누지 않고,  
**X와 T를 하나의 입력(feature set)으로 묶어 단일 모델을 학습**한다.

\[
\hat{\mu}(x, t) \approx E[Y \mid X=x, T=t]
\]

- 이진 처치일 때:
  - \(\hat{\mu}(x, 1) - \hat{\mu}(x, 0)\)를 CATE로 사용 가능.
- 연속 처치일 때:
  - \(t\)의 변화를 따라 \(\hat{\mu}(x, t)\) 곡선을 그린 뒤,  
    그 기울기(또는 차분)를 한계효과로 해석.

#### CATE / 한계효과 추정

이론적인 형태:

\[
\hat{\tau}(x)
=
\frac{\partial}{\partial T} \hat{\mu}(x, T)
\]

실무에서는 보통 유한 차분(finite difference) 사용:

\[
\hat{\tau}(x)
\approx
\frac{\hat{\mu}(x, T+\delta) - \hat{\mu}(x, T)}{\delta}
\]

예: 할인율이 10%→15%로 증가할 때 매출 기대값 차이를 이용해  
“할인율 1%p 증가당 효과”를 근사할 수 있다.

#### 장점

- 구조가 매우 단순하고 구현이 쉬움.
- 이진/연속 처치 모두 하나의 일관된 틀로 다룰 수 있음.
- 랜덤화가 완벽하지 않은 관측 데이터에서도  
  **Baseline 모델**로 꽤 좋은 성능을 보이는 사례가 많음.
- Representation learning, 비선형성, 고차원 X에 대한 일반 ML의 장점을 그대로 활용.

#### 단점

- 정규화/규제가 강한 모델에서
  - 처치 변수 T의 영향이 상대적으로 작으면,
  - 모델이 T를 사실상 무시해버려  
    **처치 효과를 0으로 shrink하는 편향**이 발생할 수 있음.
- 특히 X 쪽 설명력이 매우 크고 T의 변동 폭이 작을 때,
  - “처치해도 효과 없음”이라는 방향으로 수렴할 위험.
- 이런 편향을 완화하기 위해  
  **잔차 기반 직교화(R-learner, DML)** 를 함께 고려하는 것이 권장된다.

---

### 7.2.2 이중/편향 제거 머신러닝  
(Double / Debiased Machine Learning, R-learner 계열)

R-learner는 **FWL 정리(Frisch–Waugh–Lovell)** 를  
머신러닝/고차원 세팅으로 확장한 대표적인 Meta-learner이다.  
Chernozhukov et al.의 이중/편향 제거 머신러닝(DML) 아이디어와 맥락을 같이하며,  
잔차 기반 직교화(orthogonalization)를 통해 편향을 줄이는 것이 핵심이다.

> “X가 설명하는 부분을 먼저 제거(잔차 생성)한 뒤,  
> 남은 변동에서 T→Y 인과효과를 X별로 추정한다.”

여기서 **nuisance model(누이전스 모형)** 이란,  
\(\tau(x)\) 자체가 아니라,  
\(E[Y \mid X]\), \(E[T \mid X]\)와 같이  
“효과 추정 과정에 필요하지만, 그 자체가 관심 대상은 아닌”  
중간 모형들을 의미한다.

#### 1) 기본 아이디어 (ATE 버전)

먼저 두 가지 nuisance 모델을 ML로 학습:

- 결과 모형:
  \[
  \hat{\mu}_y(X) \approx E[Y \mid X]
  \]
- 처치 모형(연속/이산 모두 가능):
  \[
  \hat{\mu}_t(X) \approx E[T \mid X]
  \]
  - 이진 처치라면 \(E[T\mid X] = P(T=1\mid X)\)로 성향점수와 동일.

그 다음 잔차(residual)를 생성:

- 결과 잔차:
  \[
  \tilde{Y}_i = Y_i - \hat{\mu}_y(X_i)
  \]
- 처치 잔차:
  \[
  \tilde{T}_i = T_i - \hat{\mu}_t(X_i)
  \]

이제 단순 회귀:

\[
\tilde{Y}_i = \tau \cdot \tilde{T}_i + \epsilon_i
\]

의 \(\tau\)를 추정하면, 이 값이 **ATE**에 해당한다.

이때 \(\tilde{Y}, \tilde{T}\)를 만들 때  
nuisance 모델이 조금 틀려도 \(\tau\) 추정에 1차적으로 영향이 작도록 설계한 것이  
**Neyman-orthogonal score** 구조이며,  
이 점 때문에 “debiased/double robust” 성질을 가진다.

#### 2) CATE 확장: R-learner 손실

CATE를 함수 \(\tau(X)\)로 추정하려면, 모형을 다음과 같이 쓴다.

\[
Y_i = \hat{\mu}_y(X_i)
      + \tau(X_i)\cdot (T_i - \hat{\mu}_t(X_i))
      + \epsilon_i
\]

이를 변형하면,

\[
\tilde{Y}_i
=
Y_i - \hat{\mu}_y(X_i)
=
\tau(X_i)\cdot \tilde{T}_i + \epsilon_i
\]

여기서 \(\tau(\cdot)\)를 함수로 학습하기 위해  
다음 **R-loss(Residual loss)** 를 최소화한다.

\[
\hat{L}_n(\tau)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(
Y_i - \hat{\mu}_y(X_i)
- \tau(X_i)\cdot (T_i - \hat{\mu}_t(X_i)
\right)^2
\]

즉, “결과 잔차”를 “처치 잔차”로 설명하도록  
\(\tau(X)\)를 학습하는 구조이다.

조금 더 직관적으로는,

\[
\frac{\tilde{Y}_i}{\tilde{T}_i} \approx \tau(X_i)
\]

를 타겟으로 보고,  
\(\tilde{T}_i^2\)를 가중치로 주는 회귀와 동치임을 보일 수 있다.

#### 3) 장점

- CATE \(\tau(X)\)를 **함수 형태로 직접 추정** 가능하다.
- X–Y, X–T 관계에 특정 모수적 가정을 둘 필요가 없다.
  - nuisance 모델에 일반 ML(랜덤포레스트, GBM, NN 등) 활용 가능.
- 잔차 기반 직교화 덕분에,
  - nuisance 모델 \(\hat{\mu}_y, \hat{\mu}_t\)가 조금 틀려도  
    CATE 추정이 크게 망가지지 않도록 설계(편향 제거 특성).
- 적절한 조건하에서 **\(\sqrt{n}\)-rate** (전통적 통계 추정 효율)에 근접 가능하다.

#### 4) 단점 및 실무 고려사항

- nuisance 모델이 모두 ML 기반이다 보니,
  - **과적합 위험**이 크고,
  - 하이퍼파라미터 튜닝이 매우 중요하다.
- 해결책:
  - **Cross-fitting / Out-of-fold residual** 적극 활용
    - 폴드 A에서 학습한 \(\hat{\mu}_y, \hat{\mu}_t\)로 폴드 B의 residual 계산
    - “자기 자신으로 예측한 residual”을 피함으로써  
      편향 및 과적합 문제를 완화.

---

## 4. 방법별 요약 및 사용 가이드

- **T-learner**
  - 장점: 가장 단순한 baseline, 구현 용이.
  - 단점: 표본 불균형/희귀 처치 상황에서 분산·bias 문제.
  - 권장: 데이터가 충분하고 T=0/1 비율이 크게 치우치지 않을 때.

- **X-learner**
  - 장점: T-learner의 불균형 문제를 pseudo-effect + 가중 결합으로 완화.
  - 단점: 구현 복잡도가 T-learner보다 다소 높음.
  - 권장: 처치군/대조군 **표본 크기가 크게 차이날 때** 우선 고려.

- **S-learner**
  - 장점: 이진/연속 처치를 모두 한 틀에서 다룰 수 있고, 구현이 매우 단순.
  - 단점: 정규화/규제로 인해 **처치 효과를 0 쪽으로 shrink**하는 편향.
  - 권장: 연속 처치에서 dose–response 곡선의 전체 형태를 보고 싶은 baseline 모델.

- **R-learner (DML 계열)**
  - 장점: 잔차 기반 직교화로 **편향 제거에 가장 강한 방법 중 하나**.
           고차원 X, 일반 ML과의 결합에 이론적 뒷받침이 탄탄함.
  - 단점: 구현/튜닝 부담이 크고, cross-fitting이 사실상 필수.
  - 권장:  
    - 인과효과 추정이 분석의 핵심이고,  
    - 편향 최소화와 이론적 정당성이 중요한 상황.

---

## 5. 핵심 요약

1. **Meta-learner = 예측 모델을 CATE 추정용으로 재조합하는 레시피.**
2. 모든 방법은 **비교란성 가정** 아래에서만 인과적 해석이 가능하다.
3. T-learner/X-learner/S-learner는 주로 **Outcome modeling 기반**,  
   R-learner(이중/편향 제거 ML)는 **잔차 기반 직교화(orthogonalization)**를 추가로 활용한다.
4. 실제 적용 시에는
   - 데이터 크기/불균형,
   - 연속 vs 이산 처치 여부,
   - 편향 vs 분산 trade-off,
   - 구현 난이도
   를 고려하여 적절한 learner를 선택하는 것이 중요하다.
