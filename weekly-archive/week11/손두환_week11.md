# Chapter7: Meta-learner
**Meta-Learner 기반 CATE 추정**   
6장에서는 tX항 포함한 회귀분석으로 CATE추정값을 구했다면, 여기서는 몇 가지 머신러닝 알고리즘 섞은 방법 소개함.  

메타러너(Meta-learner): 기존의 비인과적 예측(machine learning) 알고리즘을 인과추론 문제(CATE 추정)에 재활용하는 접근법
- **Outcome modeling 접근법**이며, 잔차처리(orthogonalization) 또는 weighting(IPW) 없이도 작동하지만, 고차원 X에서 좋은 예측 성능을 기반으로 CATE를 간접적으로 복원.

메타러너는 특히 Representation learning, non-linear modeling, high-dimensional covariates에 강점을 가진다.

* 이 방법들은 비교란성에 의존함
  - 모든 교란 요인이 포함되어야 조건부 기댓값의 변화율을 처치반응 함수의 기울기로 해석할 수 있음. 
  $$\frac{\partial}{\partial{t}}E[Y(t) \mid X] = \frac{\partial}{\partial{t}}E[Y \mid T=t, X]$$
      - 좌항: 잠재적결과의 기울기로, 진짜 인과적 기울기, 직접 관측 불가
      - 우항: 관찰된 조건부 평균의 기울기로, 상관관계, 직접 관측 가능
      - 우항이 좌항이 되려면, confounding이 없다는 가정이 필요

# **Ⅰ. Binary Treatment Meta-learners**
상황설정: 목표는, 어떤 고객이 마케팅 이메일을 잘 받는지 파악하기  
과거에 고객에게 이메일 보낸 기록으로, CATE모델 적합시킴  
마케팅 이메일을 무작위로 발송한 실험 데이터도 존재: 평가에 활용  

## 1. T-learner (Two-model approach)
T-learner는 가장 단순한 메타러너로, 처치군과 대조군을 완전히 독립된 두 개의 예측모형으로 학습
### 정의
* 두 개의 ML 예측모형 $\hat{\mu}_1(X), \hat{\mu}_0(X)$ 학습
  * $\hat{\mu}_1(X) = E[Y \mid X, T=1]$
  * $\hat{\mu}_0(X) = E[Y \mid X, T=0]$
* CATE 추정  
  $\hat{\tau}_{T}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$

### 장점
* 매우 간단하고 구현 용이
* 각 군의 이질적인 구조도 모형이 자유롭게 학습 가능

### 단점
* 두 모형을 독립적으로 학습 → CATE 추론시 분산(variance)이 큼
  (특히 희귀한 처치의 경우 severe overfitting)
* 처치군/대조군 표본의 심각한 불균형이 있을 때 성능 크게 악화 (regularization bias)
  - 표본이 많다면 비선형성을 포착해서 good fitting이 되지만, 그렇지 않으면 지나치게 self-regularization으로 underfitting되어 bias상승
  - 그렇다고 모델 파라미터를 줄이지 않으면 overfitting 위험
  - 이 진퇴양난은 X-learner로 해결
  - 즉, 한 실험 대상 집단 (T=0 그룹, T=1그룹) 크기 차이가 클 때 X러너 성능이 더 좋음

## 2. X-learner (Cross-learning approach)
X-learner는 T-learner의 불안정성을 개선한 방식으로,
**잠재적 결과를 cross-impute하여 pseudo-treatment-effect를 만든 뒤** 하나의 모형으로 학습함

### 절차
(1) T-learner 단계 수행  
$\hat{\mu}_1(X), \hat{\mu}_0(X)$

(2) 개별 pseudo-treatment effect 생성 (**'누락된'** 잠재적 결과를 추정)
* 대조군에 대해: $\hat{\tau}(X, T=0) = \hat{\mu}_1(X, T=0) - Y_{T=0}$
* 처치군에 대해: $\hat{\tau}(X, T=1) = Y_{T=1} - \hat{\mu}_0(X, T=1)$

(3) pseudo-treatment effect을 예측하는 새로운 두 모형 학습  
$\hat\mu(X)_{\tau0}\approx E[\tau(X) \mid T=0] $  
$\hat\mu(X)_{\tau1}\approx E[\tau(X) \mid T=1]$  
여기서 한 모델은 부정확하고 다른 모델은 정확할 수 있음 (샘플 갯수에 따른 fitting 정도가 다를것).  
이를 weighted sum 방식으로 결합하는 방법이 4번 단계.  

(4) 최종 CATE: 가중평균 형태  
$\hat{\tau}(x)=\hat\mu(X)_{\tau0}\hat{e}(X)+\hat\mu(X)_{\tau1}(1-\hat{e}(X))$  
여기서 $e(X)=P(T=1\mid X)$ 또는 다른 weight

* TODO:FIXME: X-learner의 단점? 

> 참고) 281page, X learner, domain adaptation learner 차이 (DAL설명은 거의 안하긴 함)
> | 요소 | X-learner | Domain Adaptation Learner | | |
> | -| - | - | - | - |
> | 목표 | CATE 추정에서 **샘플 불균형을 완화**, 잠재적 결과 imputation 개선                | 처치/대조군 간 **분포 차이(covariate shift)** 보정                                       |            |                          |
> | 접근 방식             | T-learner 기반 두 outcome model → pseudo-treatment-effect → 가중결합 | representation learning, reweighting, adversarial learning 등을 통해 두 domain 정렬 |            |                          |
> | Counterfactual 처리 | **Cross-imputation**으로 직접 counterfactual Y 추정                 | 직접 Y를 impute하기보다 **feature 공간을 정렬**하여 추정 안정성 확보                              |            |                          |
> | 사용 상황             | 샘플 수가 처치 vs 대조군에서 크게 다를 때 유리                                  | $P(X \mid T=1) \neq P(X \mid T=0)$ 로 **포지션 매우 다를 때** 유리 |
> | 모델 구성             | outcome model 2개 + pseudo TE model 2개                         | 하나의 인코더/representer + predictor + domain classifier 등이 조합                    |            |                          |
> | 성격                | **Meta-learner**                                              | **Representation-based CAUSAL learner**                                      |            |                          |


# **Ⅱ. Continuous Treatment Meta-learners**
상황설정: 6개 레스토랑에서 할인을 무작위로 제공, 언제 더 많은 할인을 제공하면 좋을지 분석  
T: 할인율, Y: 매출  

연속형 처치 $T \in \mathbb{R}$ 의 경우, CATE는 다음과 같은 marginal effect:
$\tau(X) = \frac{\partial}{\partial T} E[Y \mid X,T]$이다.

## 1. S-learner (Functional S-learner)
$T$를 연속형으로 취급하여 $E[Y|X, T]$ 전체를 모델링.  
input feature중의 하나로 추가하는 것!!  

### **추정**
$\hat{\tau}(X)=\frac{\partial}{\partial T}\hat{\mu}(X,T)$  
Treatment 1단계 올라갔을 때 ATE차이  

### **특성**
* S learner는 단순함 때문에 어떤 인과문제에도 처음 시도하기 좋은 선택임
* 랜덤화된 데이터가 없어도 괜찮은 성능을 보이는 경향이 있음
* 이진 및 연속형 처치 모두 활용 가능
* 단점: 
  - 반사실예측이 정확한지 확인하려면 6장의 내용들을 적용해서 새로운 모델을 만들어야 한다.
  - 처치효과를 0으로 편향시키는 경향이 있다. 
  - 처치변수가 다른 공변량보다 결과 설명에 영향력이 매우 작다면, S러너는 처치변수를 완전히 버릴수 있다 (예: tree기반 모델이면, prunning?)
    - 정규화가 클수록 이문제는 더 커짐. 
      - 모델이 Y를 잘 예측하는 데 T가 도움되지 않으면: 정규화는 T 관련 파라미터를 0으로 귀착시키는 방향으로 압력을 가한다.
    - 이중/편향제거 머신러닝 또는 R러너로 해결가능

## 2. Double robust / debiased machine learning (DML)
R-Learner라고도 함.  
FWL정리의 정제된 버전임: 결과, 처치의 잔차를 구성할 때 머신러닝 모델을 사용하는 매우 간단한 방법  

### 절차
$$\left(Y_i-\hat{\mu}_y(X_i)\right) = \tau \cdot(T_i-\hat\mu_t(X_i)) +\epsilon_i$$
 - $\hat\mu_y(X_i)$: $E[Y \mid X]$ 추정
 - $\hat\mu_t(X_i)$: $E[T \mid X]$ 추정

1. ML 회귀모델 $\mu_y$ 사용해서 X로 Y를 추정
2. ML 회귀모델 $\mu_t$ 사용해서 X로 T를 추정
3. $\tilde Y = Y - \mu_y(X)$, $\tilde T = T - \mu_t(X)$를 구한다
4. 결과의 잔차를 처치 잔차에 회귀
  - $\tilde Y = \alpha + \tau \tilde T$에서 $\tau$는 인과 매개변수 ATE  

### **장점**
* CATE 추정값을 직접 출력
  - S러너는 CATE추정하려면 모델링 새로 했어야했음
* X, Y 사이나 X, T사이 관계에 모수적 가정이 필요가 없음
* 가장 정확한 편향교정
* 고차원 X에서도 일관성 보존
* ML을 사용해도 $\sqrt{n}$-rate 얻을 수 있음

### **단점**
* 과적합의 문제가 커짐
* hyper-parameter tuning 매우 중요
* 해결책: cross prediction, out of fold residual사용 권장

### DR ML로 CATE 추정
- CATE추정값을 얻으려면 몇 가지 조정이 필요함  
: 인과 매개변수 $\tau$가 실험대상의 공변량에 따라 바뀌도록 해야함.  $$Y_i = \hat{\mu}_y(X_i) + \tau(X_i)(T_i-\hat\mu_t(X_i)) +\hat\epsilon_i$$  
- 여기서 $\hat\epsilon$ 제외한 모든 항을 왼쪽으로 옮겨서 식을 만듦  
  - 이를 causal loss function이라 함.  
  - 이 손실의 제곱을 최소화하면 원하는 CATE인 $\tau(X_i)$의 기댓값을 추정할 수 있음  $$\hat L_n(\tau(x))=\frac1n\sum^{n}_{i=1}(Y_i - \hat M_y(X_i) - \tau(X_i)(T_i-\hat M_t(X_i)))^2$$
  - R 손실이라고 부름
- 이 함수를 최소화 하는것은 $\tilde T_i^2$로 가중치를 아래 식에 주고 최소화 하는것과 동일하다
  $$\frac{\tilde Y_i}{\tilde T_i} - \tau(X_i)$$

- $\tilde T_i^2$로 가중치를 주는게 왜 좋냐?
  - residual이 모두 0을 평균으로 가우시안 분포처럼 있음
  - 기울기 $\frac{\tilde Y_i}{\tilde T_i}$에서 분모가 0에 가까우면
    - 결과가 매우 불안정해짐
    - 식별 정보가 거의 없는 샘플임 (정보량 혹은 효율성이 낮음)

  - 여기에 $\tilde T_i^2$ 가중치가 원점 근처는 덜 가중치, 사이드에 가중치 많이 줘서 불안정한 영역 좀 완화해줌
  

## S-learner vs R-learner
연속형 처치 상황에서  

S-learner
> “전체 함수 $E[Y\mid X,T]$를 부드럽게 잘 그려서
> 그중 기울기 부분을 읽어내는 방식”
문제: **T가 약하면 그림에서 T축이 사라져버림.**

R-learner
> “Y의 residual과 T의 residual 사이의 기울기를 통해
> 진짜 marginal treatment effect만 정확히 분리하는 방식”
장점: **confounding에 매우 강하다.**

* **S-learner → 단순, 빠름, baseline, 데이터 적을 때**
* **R-learner → 정교, robust, confounding 강할 때, CATE 필요한 상황에서 필수**


*개념 비교표 (S-learner vs R-learner)

| 항목 | S-learner| R-learner (DR / DML) |
| - |- | - |
| **핵심 아이디어**         | $E[Y\mid X,T]$ 전체를 하나의 모델로 학습한 뒤, $\partial_T \hat\mu(X,T)$로 CATE 도출 | Y와 T 각각의 잔차(residual)를 만든 뒤, $\tilde Y$를 $\tilde T$에 회귀하여 CATE |
| **모형 형태**           | 하나의 큰 outcome regression                                             | orthogonalized regression (FWL 기반)                             |
| **수학적 특징**          | conditional mean 곡면 전체를 smooth하게 학습                                  | orthogonal score 사용 → nuisance model(μy, μt)이 조금 틀려도 $\tau(X)$는 이론적으로 편향이 거의 없다.                     |
| **반사실 예측(c.f.) 품질** | 모델링이 단순해 오류 가능성 큼                                                    | residual-on-residual 구조로 confounding을 최대한 제거                   |
| **Bias 특성**         | T가 약한 feature이면 **CATE → 0 방향으로 shrink**                             | **Neyman orthogonality로 confounding 영향 제거**                                    |
| **Var 특성**          | 모델 단순하여 variance 낮음                                                  | nuisance를 두 번 ML로 추정하므로 variance 증가 가능                         |
| **고차원 X에서**         | overfitting 발생 쉬움                                                    | 잘 설계하면 $\sqrt{n}$-rate 가능한 강력한 방법                              |
| **연속형 처치에서**        | smooth function 형태 가정 필요                                             | nonparametric local effect 추정 가능                               |
| **해석 용이성**          | 모델 하나만 학습하면 됨 → 해석 쉬움                                                | τ(X)를 직접 모델링해야 해서 구조 복잡                                        |
| **하이퍼파라미터 민감도**     | 상대적으로 낮음                                                             | 매우 높음, cross-fitting 필수                                        |
| **데이터 요구사항**        | 상대적으로 적음                                                             | sample size가 비교적 커야 안정적                                        |
| **계산 비용**           | 낮음                                                                   | 높음                                                             |


# 기타 Learner
 - modified decision tree: [Curth, A., & van der Schaar, M. (2021)](https://arxiv.org/pdf/2101.10943)
 - Neural Network based: [Johansson, F.D., Shalit, U., & Sontag, D. (2016)](https://arxiv.org/pdf/1605.03661)
