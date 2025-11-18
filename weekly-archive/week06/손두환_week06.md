# Chapter4 유용한 선형회귀
편향 제거 기법으로 인과효과 추정에 더 깊은 이해를 해보자!  
편향 제거 기법인 linear regression, ordinary least square (OLS), orthogonalization을 다룬다.

## 선형회귀의 필요성
인과추론의 핵심이자, 가장 많이 사용되는 방법임.  
panel data 방법 (difference-indifferences, DID (이중차분법)), two-way fixed effects (TWFE) 모델, 머신러닝 방법(double/biased 제거 머신러닝), 그리고 다른 식별 기법(도구변수, discontinuity design) 등 응용 방법론의 주요 구성요소임!

* ordinary least square 입문용 참고논문  
[Andrew Goodman-Bacon, 2021](http://pismin.com/10.1016/j.jeconom.2021.03.014)  
[Sloczy ski Tymon, 2020](https://direct.mit.edu/rest/article/104/3/501/97692/Interpreting-OLS-Estimands-When-Treatment-Effects)  
[Paul GoldSmith-Pinkham et al., 2022](https://arxiv.org/pdf/2106.05024)  

### 모델이 필요한 이유
$$ATE = \sum_x{\{E[Y|T=1,X=x]P(X=x) - E[Y|T=0,X=x]P(X=x)}\}$$
이런 식으로 conditional independence assumption (CIA) 만족되면 인과효과를 식별할 수 있다고 배웠음.  
근데 데이터가 연속형이라면, 그룹을 나눌 때 엄청 세밀하게 나누면 굉장히 많은 그룹을 분석해야해서 차원의 저주에 걸려버림. 즉 data sparsity문제를 겪는다.  

이를 해결하려면 잠재적 결과를 선형회귀 같은 방식으로 모델링할 수 있다고 가정하고, X로 정의된 각각의 셀을 interpolation, extrapolation하는 것이다. 즉 선형회귀분석은 차원 축소 알고리즘처럼 쓰여 결과변수를 X변수로 투영하고, 이 투영된 값으로 실험군, 대조군을 비교함.  

모델을 사용하면 좋은점:  
기존에는 통계적 유의성을 확인하려고 표준오차 계산하고 신뢰구간 구하는 검정 과정을 거쳤지만, 이제는 선형회귀 모델링으로 간편하게 모두 해결하는것임  
회귀모형은 인과추정에서 평균비교를 일반화한 형태이다.  

**랜덤화 실험에서 단순회귀**  
$$ Y = \beta_0 + \beta_1 T + e $$
를 적용하면,  

- $\hat{\beta}_0 = E[Y|T=0]$ : 대조군 평균  
- $\hat{\beta}_1 = E[Y|T=1] - E[Y|T=0]$ : 평균처치효과(ATE) 추정치 
    * 즉, 선형회귀를 통해 신뢰구간과 p-value도 함께 얻을 수 있고, 각 처치에 대한 $E[Y|T]$를 동시에 추정할 수 있다.
    * 회귀분석은 평균비교와 동일한 통계적 구조를 가지므로 표준오차, 신뢰구간, p-value를 자동으로 얻을 수 있다.  
    * 단, 인과적 의미를 부여하려면 랜덤화 또는 적절한 교란 통제(back-door 차단)가 필요하다.
    * 회귀분석은 인과추정의 도구이지만, 인과성 자체는 데이터 설계(랜덤화)에서 온다.
    * 여기에 추가로 더 좋은점 소개할 예정!
- $e$: 오차항 (error term, disturbance) — 관측된 결과 $Y$와 모델이 예측하는 평균값 간의 차이(모형에 의해 설명되지 않는 성분).
    * (1) 기대값 조건  
        - 핵심 가정(랜덤화 하에서): $E[e\mid T]=0$.  
        - 해석: 처치 $T$와 오차항 $e$가 평균적으로 독립적이며, 랜덤화로 인해 교란요인이 통제되었음을 수학적으로 나타냄.
    * (2) 의미적 해석 및 결과  
        - $e$는 개인 성향, 환경, 측정오차 등 $T$ 이외 요인들의 합성된 비설명 성분임.  
        - $E[e\mid T]=0$이면 $\operatorname{Cov}(T,e)=0$이 되어 OLS 추정량의 편향이 제거된다. 예:  
            $$
            E[\hat\beta_{\text{omit}}]=\beta + \gamma\frac{\operatorname{Cov}(T,Z)}{\operatorname{Var}(T)}.
            $$
            이때 $\operatorname{Cov}(T,e)=0$이면 $\hat\beta_1$는 편향 없는 인과추정량(unbiased causal estimator)이 된다.  
        - 실제 분석에서는 이 조건이 성립하는지(랜덤화 여부, 잔여 교란 존재 여부)를 항상 검토해야 함.

<br>

**회귀분석을 통한 보정**  
예를들어, bias 보정하려면 모든 교란요인에 따라 데이터를 나누고, 나눈 각 그룹 내에서 회귀하고, 기울기 매개변수 추출해서 결과의 평균을 구하면 되는데, 차원의 저주 때문에 기존 방법의 회귀가 불가능하다고 하자!  

이럴때는 교란 요인을 직접 보정하는 대신, OLS로 추정할 모델에 단순히 교란요인을 추가하면 된다.  
$$ Default_i = \beta_0 + \beta_1line_i + \bold{\theta}\bold{X}_i + e_i$$
X는 교란요인 벡터, $\bold\theta$는 교란요인 관련 매개변수 벡터로 편향되지 않은 $\beta_1$ 추정값을 얻는데 도움이 되는 매개변수임. 이런 매개변수를 nuisance parameter (장애모수)라고 부름.  

예:  
$$ Default_i = \beta_0 + \beta_1line_i + \theta_1wage_i + \theta_2creditScore_i + e_i$$
이렇게 모델 만들고, $\beta_1$을 추정하려고 하는 것임.  
$$ \frac{\partial}{\partial{t}}E[y \mid t, X]$$
를 해서 treatment에 대해 편미분하면 $\beta_1$ 추정치가 나옴.  
모델의 다른 모든 변수가 고정된 상태에서의 변화량에 대한 기댓값!  
즉, 처치와 결과 사이 관계를 추정하는 동안 교란요인을 고정. 
자센한건 나중에 나옴!  

## 회귀분석 이론
### 단순선형회귀
단순선형회귀는 한 개의 설명변수 $T$와 종속변수 $Y$에 대해
$$
Y_i = \beta_0 + \beta_1 T_i + \varepsilon_i,\quad E[\varepsilon_i\mid T_i]=0
$$
로 가정한다. OLS 추정량은 잔차 제곱합을 최소화하여
$$
\hat\beta_1 = \frac{Cov(Y_i,T_i)}{Var(T_i)} = \frac{\sum_i (T_i-\bar T)(Y_i-\bar Y)}{\sum_i (T_i-\bar T)^2}
$$
로 단일 설명변수에 대한 매개변수가 주어진다.  
T가 무작위로 배정되면 이 계수가 ATE다.  
즉, 회귀분석은 처치와 결과가 어떻게 함께 움직이는지(분자의 공분산으로) 파악하고 이를 처치 대상에 따라 조정함.  

확률극한에서는 $\hat\beta_1 \xrightarrow{p} \beta_1$이 성립하고, 중심극한정리에 따라 적절한 정규근사를 사용해 신뢰구간과 검정을 할 수 있다. 이때 분산추정은 이분산성(heteroskedasticity)에 민감하므로 견고한 표준오차(HC0–HC3) 또는 부트스트랩을 권장한다.

### 다중선형회귀
$$
y_i = \beta_0 + \tau T_i + \beta_1X_{1i} + ... + \beta_kX_{ki} +u_i\\
\hat{\tau}=\frac{Cov(Y_i, \tilde{T_i})}{Var(\tilde{T_i})} \\
$$
* 여기서 $\tilde{T_i}$는 $T_i$를 모든 공변량 $X_{1i}+...+X_{ki}$에 대해 회귀한 residual임  
* 다중회귀분석에서 회귀계수의 의미는 모델의 다른 변수들의 효과를 고려한 후 얻은 동일 설명변수의 bivariate 계수라는 것!  
* 인과추론 관점에서, $\tau$는 다른 모든 변수로 T를 예측하여 얻은 T의 이변량 계수

벡터화한 OLS 해:
$$
Y = X\beta + \varepsilon,\qquad E[\varepsilon\mid X]=0 \\
\hat\beta=(X'X)^{-1}X'Y
$$

## Frisch–Waugh–Lovell 정리(FWL)와 직교화
* 가장 먼저 사용할 수 있는 편향제거(debiasing) 기법이다.  
* 편향제거 단계를 직교화(orthogonalization) 또는 잔차화(residualization)이라 함.  
* FWL는 가장 먼저 사용할 수 있는 편향제거 기법으로, nonexperimental 데이터를 처치가 무작위 배정된 것 처럼 보이게 한다.  
* 다중회귀에서 특정 변수의 계수를 추정할 때 "다른 변수들에 대해 부분화(partial out)"해도 동일한 결과가 나온다는 강력한 도구다. 

세 변수군으로 나누면
$$
Y = \beta_1T + \beta_2X + \varepsilon
$$
1. $T$를 $X$에 대해 회귀하여 잔차 $\tilde T$을 얻음.
    - 이는, 편향제거 구성요소 분리가 가능함을 보여준다
2. $Y$를 $X$에 대해 회귀하여 잔차 $\tilde Y$를 얻음(즉 $Y$에서 $X$의 선형성분 제거).
3. $\tilde Y$를 $\tilde T$에 대해 회귀하여 얻은 계수는 원래의 $\beta_1$와 동일.
    - T가 Y에 미치는 인과효과 추정값  

(증명은 선형대수(사영행렬 $P_{X_2}=X_2(X_2'X_2)^{-1}X_2'$ 사용)로 간단하다. FWL은 편향제거(pre‑processing)와 추정 단계의 분리를 가능하게 하므로, 편향 제거를 위한 비선형·복잡한 방법(머신러닝 등)을 1단계에 쓰고, 마지막에는 선형추정을 적용하는 하이브리드 설계에 특히 유용하다.)  
[증명](https://datascienceschool.net/03%20machine%20learning/04.05%20%EB%B6%80%EB%B6%84%ED%9A%8C%EA%B7%80.html)

**단계1: 편향 제거**
- 1단계: 통제변수 $X$로 $Y$와 $T$를 각각 부분화하여 잔차 $\tilde Y,\tilde T$를 얻음.
- 2단계: $\tilde Y$를 $\tilde T$에 대해 회귀(또는 두 변수 간 단순회귀)하여 처리효과 추정.
이 접근은 nuisance parameter를 비선형 방법으로 추정해도 2단계에서의 $\beta$ 추정이 일관성을 가지게 하는 근거가 된다(단, 1단계 추정의 속도조건 필요 — double/debiased machine learning 참조).
- 편향제거 단계만 적용해도, 모든 교란요인이 편향제거 모델에 포함됬으면, 인과적 영향에 대해서는 unbiased estimate를 얻을 수 있다.

**단계2: 잡음 제거**
- 1단계에서 과도한 변수(노이즈)를 제거하고, 유의미한 부분만 남김으로써 분산을 줄인다.
- 이 단계에서는 결과를 처치가 아닌 공변량에 대해 회귀한다
    - E[Y|X]를 추정하고 Y - E[Y|X]를 만드는 것 (일종의 normalize?)
- (그러나 1단계에 과적합(overfitting)이 있으면 2단계 분산과 편향에 영향이 있으므로 교차검증·샘플분할(sample splitting)이 권장된다)

- 회귀 추정량의 표준오차
    - 회귀 표준오차 계산법 $$ SE(\hat\beta)=\frac{\sigma(\hat\epsilon)}{\sigma(\tilde T)\sqrt{n-DF}} $$  
        * $\hat\epsilon$: 회귀모델 잔차
        * n-DF: 모델의 자유도 (DF: 모델이 추정하는 매개변수 갯수)  
        * 결과를 잘 예측할수록 잔차가 작아져 분자가 작아져 추정값의 분산이 낮아짐
        * 처치가 결과를 많이 설명하면 매개변수 추정값의 표준오차도 낮아짐
    - FWL 절차 이후 표준오차는 2단계 절차에 맞게 조정해야 한다. 1단계에서 추정오차를 무시하면 과소추정이 될 수 있음.
    - 일반적으로 이중머신러닝 프레임워크에서는 교차적합(cross-fitting)을 통해 1단계 추정오차의 영향 제거와 일관된 표준오차를 얻는다.
    - 관측치 군집(clustered)이나 시계열 종속성 존재 시 클러스터 표준오차를 사용.

**단계3: 최종 결과 모델**
- 잔차$\tilde Y$를 잔차$\tilde T$에 회귀하는 단계
- (FWL에 의해 얻은 $\beta$는 "다른 통제변수들이 일정할 때의 국소선형효과"를 의미하며, 인과해석을 위해서는 무작위화 또는 조건부독립성 가정(CIA)이 필요하다.)
- 불연속점이나 희소영역에서의 외삽 위험을 항상 점검.

## 결과 모델로서의 회귀분석
회귀분석은 잠재적 결과를 imputation하는 방법으로도 볼 수 있다.  
처치가 0 or 1로 이진값일 때, 대조군(T=0)에서 $E[Y_0|X]$ 를 잘 근사한다면, $Y_0$를 대체하고 ATT를 추정할 수 있다.  
$$ATT = 1/N_1\sum\bold1(T_i=1)(Y_i-\hat\mu_0(X_i))$$

마찬가지로, 실험군(T=1)에서 $E[Y_1|X]$를 잘 모델링 했으면, 대조군의 평균효과를 추정할 수 있다.  

두 접근법을 병행하면, ATE 추정도 가능하다  
$$ATE = 1/N\sum(\hat\mu_1(X_i)-\hat\mu_0(X_i))$$

누락된 잠재적 결과 대체도 가능하다
$$ATE = 1/N\sum(\bold1(T_i=1)[Y_i-\hat\mu_0(X_i)]+\bold1(T_i=0)[\hat\mu_1(X_i)-Y_i])$$

T가 연속형이면, 회귀분석은 전체 처치반응함수를 대체하는 것으로 이해하면 된다.  
5장에서 다루겠지만, 잠재적결과 $E[Y_t|X]$를 정확하게 추정하거나 E[T|X]를 정확하게 추정한다는 것은 회귀분석이 doubly robust하다는 특성을 나타내는 것. (4부에 이중차분법에서도 나옴)  
(근거?)  


## 선형회귀에서의 비선형성
**양수성과 외삽**
- 양수성(positivity, common support): 모든 $x$에 대해 $0< P(T=1\mid X=x) <1$이어야 조건부효과 식별이 가능하다.
- 특정 $X$ 영역에서 한 처리군만 관찰되면 외삽이 불가피하고 추정은 모델가정에 민감해진다. 즉 오차가 매우 커질 수 있다.
- (외삽 위험을 줄이기 위해 가중법(IPW), 이중강건추정(DR), 지역적추정(지역화)을 고려할 수 있다고 함)
