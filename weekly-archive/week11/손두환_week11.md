# Meta-Learner 기반 CATE 추정
6장에서는 tX항 포함한 회귀분석으로 CATE추정값을 구했다면, 여기서는 몇 가지 머신러닝 알고리즘 섞은 방법 소개함.  

메타러너(Meta-learner)는 기존의 비인과적 예측(machine learning) 알고리즘을 인과추론 문제(CATE 추정)에 재활용하는 접근법.
본질적으로, 메타러너는 **Outcome modeling 접근법**이며, 잔차처리(orthogonalization) 또는 weighting(IPW) 없이도 작동하지만, 고차원 X에서 좋은 예측 성능을 기반으로 CATE를 간접적으로 복원.

메타러너는 특히 Representation learning, non-linear modeling, high-dimensional covariates에 강점을 가진다.

* 이 방법들은 비교란성에 의존함
  - 모든 교란 요인이 포함되어야 조건부 기댓값의 변화율을 처치반응 함수의 기울기로 해석할 수 있음. 
  $$\frac{\partial}{\partial{t}}E[Y(t) \mid X] = \frac{\partial}{\partial{t}}E[Y \mid T=t, X]$$

논의할거리 / 궁금점 미해결
1. 281page, X learner, domain adaptation learner 차이 (DAL설명은 거의 안하긴 함)

# **Ⅰ. Binary Treatment Meta-learners**
상황설정: 목표는, 어떤 고객이 마케팅 이메일을 잘 받는지 파악하기  
과거 고객에데 이메일 보낸 기록으로, CATE모델 적합  
마케팅 이메일을 무작위로 발송한 실험 데이터도 존재: 평가에 활용  

## 1. T-learner (Two-model approach)
T-learner는 가장 단순한 메타러너로, 처치군과 대조군을 완전히 독립된 두 개의 예측모형으로 학습
### 정의
* 두 개의 ML 예측모형 $hat{\mu}_1(X), \hat{\mu}_0(X)$ 학습
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

(2) 개별 pseudo-treatment effect 생성 (누락된 잠재적 결과를 추정)
* 대조군에 대해: $\hat{\tau}(X, T=0) = \hat{\mu}_1(X, T=0) - Y_{T=0}$
* 처치군에 대해: $\hat{\tau}(X, T=1) = Y_{T=1} - \hat{\mu}_0(X, T=1)$

(3) 각각을 예측하는 두 모형 학습  
$\hat\mu(X)_{\tau0}\approx E[\tau(X) \mid T=0] $  
$\hat\mu(X)_{\tau1}\approx E[\tau(X) \mid T=1]$  
여기서 한 모델은 부정확하고 다른 모델은 정확할 수 있음 (샘플 갯수에 따른 fitting 정도가 다를것).  
이를 weighted sum 방식으로 결합하는 방법이 4번 단계.  

(4) 최종 CATE: 가중평균 형태  
$\hat{\tau}(x)=\hat\mu(X)_{\tau0}\hat{e}(X)+\hat\mu(X)_{\tau1}(1-\hat{e}(X))$  
여기서 $e(X)=P(T=1\mid X) \text{ 또는 다른 weight}$

# **Ⅱ. Continuous Treatment Meta-learners**
상황설정: 6개 레스토랑에서 할인을 무작위로 제공, 언제 더 많은 할인을 제공하면 좋을지 분석  
T: 할인율, Y: 매출  

연속형 처치 $T \in \mathbb{R}$ 의 경우, CATE는 다음과 같은 marginal effect:
$\tau(X) = \frac{\partial}{\partial T} E[Y \mid X,T]$이 되며, Meta-learner는 이를 구현하기 위한 frameworks를 제공한다.

## 1. S-learner (Functional S-learner)
$T$를 연속형으로 취급하여 $E[Y|X, T]$ 전체를 모델링. 

### **추정**
$\hat{\tau}(X)=\frac{\partial}{\partial T}\hat{\mu}(X,T)$  
Treatment 1단계 올라갔을 때 ATE차이  

### **특성**
* S learner는 단순함 때문에 어떤 인과문제에도 처음 시도하기 좋은 선택임
* 랜덤화된 데이터가 없어도 괜찮은 성능을 보이는 경향이 있음
* 이진 및 연속형 처치 모두 활용 가능
* 단점: 처치효과를 0으로 편향시키는 경향이 있다.
* 단점2: 처치변수가 다른 공변량보다 결과 설명에 영향력이 매우 작다면, S러너는 처치변수를 완전히 버릴(?)수 있다
  - 정규화가 클수록 이문제는 더 커짐. 
  - 이중/편향제거 머신러닝 또는 R러너로 해결가능

## 2. Double robust / debiased machine learning (DML)
R-Learner라고도 함.  
FWL정리의 정제된 버전으로 볼 수 있음:  
결과, 처치의 잔차를 구성할 때 머신러닝 모델을 사용하는 매우 간단한 방법  

### 절차
$\psi_i(\tau)=\left(Y_i-\hat{\mu}_y(X_i)\right) = \tau \cdot(T_i-\hat\mu_t(X_i)) +\epsilon_i$  
 - $\hat\mu_y(X_i)$: $E[Y \mid X]$ 추정
 - $\hat\mu_t(X_i)$: $E[T \mid X]$ 추정

1. ML 회귀모델 $\mu_y$ 사용해서 X로 Y를 추정
2. ML 회귀모델 $\mu_t$ 사용해서 X로 T를 추정
3. $\tilde Y = Y - \mu_y(X)$, $\tilde T = T - \mu_t(X)$를 구한다
4. 결과의 잔차를 처치 잔차에 회귀
  - $\tilde Y = \alpha + \tau \tilde T$에서 $\tau$는 인과 매개변수 ATE  
  - ($\tau \tilde T$이 score가 Neyman-orthogonal이므로 nuisance model $Mu_y, mu_t 추정$의 small error에 robust하다)

### **장점**
* CATE 추정값을 직접 출력
  - S러너에 필요했던 모든 추가 단계가 필요없음
* X, Y 사이나 X, T사이 관계에 모수적 가정이 필요가 없음
* 가장 정확한 편향교정
* 고차원 X에서도 일관성 보존
* ML을 사용해도 $\sqrt{n}$-rate 얻을 수 있음

### **단점**
* 과적합의 문제가 커짐
* hyper-parameter tuning 매우 중요
* 해결책: cross prediction, out of fold residual사용 권장

### DML으로 CATE 추정
- CATE추정값을 얻으려면 몇 가지 조정이 필요함  
: 인과 매개변수 $\tau$가 실험대상의 공변량에 따라 바뀌도록 해야함.  
$Y_i = \hat{\mu}_y(X_i) + \tau(X_i)(T_i-\hat\mu_t(X_i)) +\hat\epsilon_i$  
- 여기서 $\hat\epsilon$ 제외한 모든 항을 왼쪽으로 옮겨서 식을 만듦  
  - 이를 causal loss function이라 함.  
  - 이 손실의 제곱을 최소화하면 원하는 CATE인 $\tau(X_i)$의 기댓값을 추정할 수 있음  $$\hat L_n(\tau(x))=\frac1n\sum^{n}_{i=1}(Y_i - \hat M_y(X_i) - \tau(X_i)(T_i-\hat M_t(X_i)))^2$$
  - R 손실이라고 부름
- 이 함수를 최소화 하는것은 $\tilde T_i^2$로 가중치를 다음 항에 주고 최소화 하는것과 동일하다
  $$\frac{\tilde Y_i}{\tilde T_i} - \tau(X_i)$$

- TODO:FIXME: 이 개념에 대한 책에 나온 직관적 해설 정리
- 수학적 증명 추가?
  
# 기타 Learner
 - modified decision tree: [Curth, A., & van der Schaar, M. (2021)](https://arxiv.org/pdf/2101.10943)
 - Neural Network based: [Johansson, F.D., Shalit, U., & Sontag, D. (2016)](https://arxiv.org/pdf/1605.03661)
