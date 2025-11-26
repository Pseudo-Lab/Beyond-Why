# 7. Meta-learner 기반 CATE 추정 정리

## 0. Meta-learner 개요

**Meta-learner**는 원래 목적이 *결과 예측*인 일반 머신러닝 모델  
(예: \( \hat{\mu}(X) \approx E[Y \mid X] \))을 재조합하여,

> “누가 처치를 받았을 때 얼마나 이득을 보는가?”  
> 즉, **개별 처치 효과(CATE, Conditional Average Treatment Effect)**  
> $$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$

를 추정하기 위한 프레임워크.

### 핵심 특징

- **Outcome modeling 접근**  
  - 잔차처리(orthogonalization)나 IPW(역확률 가중) 없이도  
    \( E[Y \mid X, T] \)를 잘 예측하는 모델을 통해 CATE를 간접 복원.
- **표현 학습에 강함**  
  - 비선형 관계, 고차원 공변량, 복잡한 상호작용 등을  
    일반 ML 모델(XGBoost, RF, NN 등)로 캡처한 뒤 인과 추론에 재활용.
- **FWL/직교화 포함 여부**  
  - T/X/S-learner는 순수 outcome modeling.
  - R-learner(DML)는 잔차 기반 직교화 포함.

### 전제: 비교란성(비편향성) 가정

Meta-learner 기반 CATE 해석이 인과적으로 의미 있으려면  
비교란성(ignorability/unconfoundedness)이 반드시 성립해야 함.

$$(Y(0), Y(1)) \perp T \mid X$$

연속 처치에서는 다음이 성립해야 함.  

$$\frac{\partial}{\partial t}E[Y(t) \mid X] = \frac{\partial}{\partial t}E[Y \mid T=t, X]$$

---

## 7.1 이산형 처치 Meta-learner

이진 처치: \(T \in \{0, 1\}\)

- $$\mu_1(x) = E[Y \mid X=x, T=1]$$  
- $$\mu_0(x) = E[Y \mid X=x, T=0]$$  
- $$\tau(x) = E[Y(1)-Y(0)\mid X=x]$$  

---

### 7.1.1 T-learner (Two-model)

#### 정의

1. 처치군 모델  
   $$\hat{\mu}_1(x) \approx E[Y \mid X=x, T=1]$$
2. 대조군 모델  
   $$\hat{\mu}_0(x) \approx E[Y \mid X=x, T=0]$$
3. CATE  
   $$\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

#### 장점
- 가장 단순, 구현 쉬움
- 두 집단의 구조적 차이를 각각 따로 학습 가능

#### 단점
- 두 모델을 독립 학습 → **분산 커짐**
- 표본 불균형 시 작은 집단은 **자기정규화 → underfitting**  
- 희귀 처치일수록 불안정

---

### 7.1.2 X-learner (Cross-learning)

> 누락된 반사실을 cross-impute → pseudo-treatment effect 생성 → CATE을 다시 스무딩

#### 절차

1) T-learner 실행  
   - \(\hat{\mu}_1(x)\), \(\hat{\mu}_0(x)\) 확보

2) Pseudo-treatment effect 생성
- 처치군:
  $$D_1 = Y - \hat{\mu}_0(X)$$
- 대조군:
  $$D_0 = \hat{\mu}_1(X) - Y$$

3) 두 pseudo-effect 모델 학습
- $$\hat{\tau}_1(x) \approx E[D_1 \mid X=x, T=1]$$
- $$\hat{\tau}_0(x) \approx E[D_0 \mid X=x, T=0]$$

4) 가중 결합  
   성향점수 \( \hat{e}(x)=P(T=1\mid X=x) \)을 사용하여  

$$\hat{\tau}(x) = w(x)\hat{\tau}_1(x) + (1-w(x))\hat{\tau}_0(x)$$

보통 \(w(x)=\hat{e}(x)\) 또는 집단 크기 비율 기반 weight.

#### 직관

- 표본이 많은 집단에서 나온 pseudo-effect를 더 신뢰
- **표본 불균형에서 매우 강함**

---

## 7.2 연속형 처치 Meta-learner

연속형 처치: \(T \in \mathbb{R}\)

CATE는 보통 한계효과(marginal effect):

$$\tau(x)=\frac{\partial}{\partial T} E[Y \mid X=x, T]$$

---

### 7.2.1 S-learner (Functional S-learner)

#### 정의

단일 모델로 전체 dose–response 곡선을 적합:

$$\hat{\mu}(x,t) \approx E[Y \mid X=x, T=t]$$

- 이진 처치: \(\hat{\mu}(x,1)-\hat{\mu}(x,0)\)
- 연속 처치: 기울기 계산

#### 한계효과 추정

이론형:

$$\hat{\tau}(x) = \frac{\partial}{\partial T}\hat{\mu}(x, T)$$

실무형(차분):

$$\hat{\tau}(x) \approx \frac{\hat{\mu}(x, T+\delta) - \hat{\mu}(x, T)}{\delta}$$

#### 장점
- 구조 단순 / 구현 쉬움
- 이진–연속 공통 구조
- 비선형 적합에 강함

#### 단점
- T의 영향이 작으면 모델이 T를 **무시하는 경향**
- 정규화 강하면 **처치효과가 0으로 수축(shrinkage)**  
- 따라서 편향 가능 → R-learner로 보완 가능

---

### 7.2.2 R-learner (Double/Debiased ML)

핵심:  
> FWL 정리를 머신러닝으로 확장 → 결과/처치 관계에서 X의 영향 제거(잔차화)

#### 1) 잔차 생성 단계

Outcome nuisance model:

$$\hat{\mu}_y(X) \approx E[Y \mid X]$$

Treatment nuisance model:

$$\hat{\mu}_t(X) \approx E[T \mid X]$$

잔차 생성:

$$\tilde{Y}_i = Y_i - \hat{\mu}_y(X_i)$$  

$$\tilde{T}_i = T_i - \hat{\mu}_t(X_i)$$

#### 2) ATE 추정

단순 회귀:

$$\tilde{Y}_i=\tau \cdot \tilde{T}_i + \epsilon_i$$

→ nuisance 모형 오류에 강한 **Neyman-orthogonality** 구조

#### 3) CATE 확장 (R-loss)

CATE 함수 \(\tau(X)\)는 다음 loss를 최소화하는 함수:  

$$\hat{L}_n(\tau)=\frac{1}{n}\sum_{i=1}^{n}\left(Y_i-\hat{\mu}_y(X_i)-\tau(X_i)\cdot(T_i - \hat{\mu}_t(X_i))\right)^2$$

직관적으로는,  

$$\frac{\tilde{Y}_i}{\tilde{T}_i}\approx \tau(X_i)$$

이고, 가중치 \(\tilde{T}_i^2\)로 회귀하는 것과 동치.

#### 장점

- \(\tau(x)\)를 **직접 함수로 추정**
- nuisance 모델이 조금 틀려도 효과 추정 안정적
- 고차원에서도 이론적 정당성 강함
- 적절한 조건에서 \(\sqrt{n}\)-rate 근접

#### 단점

- nuisance 모델이 ML → 과적합 위험
- **cross-fitting 필수**

---

## 4. 방법별 요약 가이드

| Learner | 장점 | 단점 | 추천 상황 |
|--------|------|------|-----------|
| **T-learner** | 단순·구현 쉬움 | 분산 큼, 불균형 취약 | 표본 균형·데이터 많음 |
| **X-learner** | 불균형에 강함 | 약간 복잡 | T=1/0 표본 격차 클 때 |
| **S-learner** | 연속·이진 일관 프레임 | shrinkage bias | 연속 처치 baseline |
| **R-learner** | 편향 제거 최고, 이론 강함 | 구현·튜닝 난도 높음 | 정확한 CATE가 핵심일 때 |

---

## 5. 핵심 요약

1. Meta-learner는 **예측 모델을 조합하여 CATE를 복원**하는 프레임워크.  
2. 비교란성 없으면 어떤 Meta-learner도 인과적 해석 불가.  
3. T/X/S-learner는 outcome modeling 중심.  
4. R-learner(DML)는 **잔차 기반 직교화로 편향 제거**.  
5. 데이터 구조(표본 불균형, 연속/이산 처치, 비선형성)에 따라 learner 선택이 달라진다.
