# Chapter 9. Synthetic Control Method (통제집단합성법)

궁금한점 / 논의점  
1. SCM하면 왜 평행추세가정이 필요없어짐?
2. 처치 샘플이 하나여도 왜 잘 작동함?
3. inverse propensity score에서 가상의 샘플을 만드는 것과 차이?
4. convexity constraint (볼록성 제약조건)이 최적화 문제에 대한 sparsity 해결법을 어떻게 주는걸까? 수학적 배경?

## 개요
패널데이터셋에 널리 사용되는 이중차분법이랑 다른 방법!  

통제집단합성법(Synthetic Control Method, SCM):  
실제 대조군을 convex combination(가중합) 하여 가상의 대조군(synthetic control)을 만드는 방법.
  - **처치가 없을 때의 실험군 역할**을 함 
  - 평행 추세 가정이 필요없어짐 
  - 처치 샘플이 하나여도 잘 작동함 (?)

Abadie & Gardeazabal (2003), Abadie, Diamond & Hainmueller (2010)의 연구로 널리 사용되며,
단일 처치 대상(single treated unit)에 대해 시간에 따른 처치효과(ATT)를 추정하는 데 특히 적합하다.

**핵심 아이디어**:  
> 1. 관측된 대조군들의 적절한 가중치 조합으로  
> 2. 처치 전(pre-treatment) 기간 동안 실험군의 '경로'를 가장 잘 모방하는 synthetic unit을 만든 뒤,  
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

## 행렬 표현
처치배정행렬: 
$$
W =
\begin{bmatrix}
0_{pre,co} & 0_{pre,tr} \\
0_{post,co} & 1_{post,tr}
\end{bmatrix}
$$
행: 실험기간, 열: 도시(실험 대상, 비교군/대조군)  
co: control units, tr: treated units  
이 처치배정행렬로 관측된 잠재적 결과행렬을 표현하면:  
$$
Y =
\begin{bmatrix}
Y(0)_{pre,co} & Y(0)_{pre,tr} \\
Y(0)_{post,co} & Y(1)_{post,tr}
\end{bmatrix}
$$
- 이 행렬로 $ATT = Y(1)_{post, tr} - Y(0)_{post, tr}$를 구하면 되지만, $Y(0)_{post, tr}$는 측정이 안됬음  
- 그래서 $Y(0)_{pre,co}, Y(0)_{pre,tr}, Y(0)_{post,co}$를 잘 합성해서 만들고자 함.  

## 통제집단합성법과 수평 회귀분석
처치 이전 기간을 사용해서 대조군을 결합-> 실험군의 평균결과를 잘 근사할 수 있는 방법을 찾는다.
**이중차분법과 마찬가지로, 처치에 대한 비기대 가정을 만족함을 가정함. 꼬한 파급효과가 없다고도 가정함.**

  - 대조군 잘 모의하면 실험군도 잘 모의한다는 가정인듯.
  - 각 실험 대상에 가중치 $\omega_i$를 부여해서 최적화 문제로 접근 $$\hat{\omega}^{sc}= \argmin_{\omega}\left\| \bar y_{\text{pre},\, tr} - Y_{\text{pre},\, co}\,\omega_{co} \right\|^2$$
    - 처치 전 대조군에 $\omega_i$줘서 실험군의 평균과의 차이를 줄여서 최적화/fitting함
      - sc: synthetic control
    - 이를 처치 후 비교군 샘플을 입력값으로 넣어서 $Y(0)_{post,co}$를 추정하고 평균내서 사용함.
    - 이는 선형회귀의 목표와 같음
      - 대조군 결과를 특성으로 사용해서 실험군의 평균 결과를 예측하는 회귀
      - 기존 회귀분석의 design matrix: 행: N, 열: covariate
      - 지금 $\omega$구하는 식 design matrix: 행: 기간, 열: N
      - 통제집단합성법을 수평회귀분석 이라고도 함
    - 다만, 여기서 사용하는 모델은 일반적인 ML 모델일 수 없다. linear regression 식만 허용 : 가중치 획득이 목적이니까
  - 시간에 따른 평균 대조군을 얻게됨: 시간에 따른 ATT가 최종 결과임

* 시간에 따른 ATT 분석 내용
  - 처치 이후 효과가 정점에 이르기까지 일정 시간이 걸림
  - 그 이후에 효과가 사라짐: novelty effect (신기 효과)라 함
  - 처치 이전에는 ATT가 0에 가까움
    - OLS모델의 잔차 또는 표본오차
    - 0에 지나치게 가까우면 overfitting임
      - 추정 시 out-of-sample 예측오차가 커질 수 있음
    - 이 문제로 인해 잘 안씀
    - 다음 주제인, 제약조건을 부여하여 통제집단합성법으로 사용함

## 표준 SCM: 두 가지 제약조건
표준 SCM은 회귀모델에 두가지 제약을 부여함:
1. 가중치 양수성
2. 가중치 합 1

이를 고려하여 최적화 함수를 고려하면:  
$$\hat{\omega}^{sc}= \argmin_{\omega}\left\| \bar y_{\text{pre},\, tr} - Y_{\text{pre},\, co}\,\omega_{co} \right\|^2 \\
\text{s.t. for all i,} \sum\omega_i=1 \ {and} \ \omega_i>0
$$
- 제약조건의 목적: 가상의 대조군이 실험군에 대한 convex combination이 되도록 하여 외삽을 피하며 과적합 위험 감소
  - 실험군에 속한 모든 대상의 범위가 대조군의 범위 밖에 있다면, interpolation할 수 없고, 외삽으로만 추정 가능함: 이런 외삽을 막는다.
  - 가능한 가중치 범위 제한됨: 주어진 조건에서만 만족하는 특정값 찾음
  - 일반화 능력 향상, 추정값 신뢰도 증가
- convexity constraint (볼록성 제약조건)이 최적화 문제에 대한 sparsity 해결법을 준다
  - 즉 최종 가상의 대조군을 만드는 데 소수의 도시만 사용된다.
- 정규화 효과로 인해, fitting (학습)오차는 조금 더 커져도, ATT추정 시계열의 잡음은 줄어듦

## SCM과 공변량(Covariates)
드문경우이긴 한데, 공변량의 예측력이 좋아서 모델에 초가 공변량을 포함해도 좋은 경우가 있다고 함.  

$$\hat{\omega}^{sc}= \argmin_{\omega}\left\| \bar y_{\text{pre},\, tr} - \sum\nu^*_k\bold{X}_{k, pre, co}\omega_{co} \right\|^2 \\
\text{s.t. for all i,} \sum\omega_i=1 \ {and} \ \omega_i>0
$$
(TODO:FIXME: ????????? 이 부분, 코드랑 같이 다시 공부)
- 가중치가 $y_{co}$에선 곱해지지 않고, 추가 공변량인 $x_{co}$에도 곱해짐
- $y_{co}$와 $x_{co}$가 완전 다른 척도에 있거나, 한쪽이 더 예측력이 좋을 수 있어서 최적화 문제를 풀기 전에 $y_{co}$를 포함한 각 공변량에 scaling factor $\nu$를 곱함
- 이를 바탕으로 $y_{co}$를 또 다른 공변량으로 취급하고 공변량 X의 관점으로 목적함수 재구성함
  - $\nu$찾으려면 전체 통제집단합성법을 또 다른 최적화 목적함수로 묶어야함

**참고: generic horizontal regression**  
공변량 추가하는 더 쉬운 방법  
  - 중요해보이는 추가 시계열 정보를 $Y_{pre, co}$에 붙이기  
  - 수평회귀분석에서 공변량을 추가하는것과 동일
  - $[\bold{Y}_{pre, co} | \bold{X}_{pre, co}]\omega$
  - 대조군과 추가 시계열을 사용하여 $E[Y(0)_{tr}]$를 추정하므로 엄밀하게는 통제집단합성법은 아님.
    - 실험대상이랑 추가한 열 각각에 대해서도 가중치 가져버리니까

## SCM과 편향 제거 (Bias Reduction)
$T_{pre}$수가 적으면 과적합이 발생할 수 있다. 이에 SCM은 편향될수 있다. 이 편향을 줄이는 방법: cross fitting
1. 개입 전 기간을 k블록으로 나눔. 각 블록의 크기는 $min(T_{pre}/K, T_{post})$ 
2. 각 블록을 hold-out셋으로 취급, 통제집단합성법 모델 fitting해서 블록 별로 \hat{\omega}^k$를 얻는다 (즉 모델을 얻는다)
3. 각 블록별로 bias보정: hold-out데이터와 그 블록의 모델로 예측값을 얻는다.
  - 이 예측값이랑 관측값의 차이가 bias 추정값임
4. k번째 블록에서 기존 방식대로 구한 ATT에서 bias를 뺀 값이 $ATT^k$의 추정값이다
5. k별로 1~4과정을 거치고 k개 ATT추정값을 평균내서 최종 ATT값 하나를 얻는다.

## 추론(Inference)
- 편향제거 과정은 그 자체로 유용하지만, ATT추정값 주위에 신뢰구간 설정이 가능해서 더 중요하다.  
- 그러나 추론에 문제점: 대조군이 적은 경우, bootstrap같은 방법이 먹히지 않는다.  
- 대부분 시간 차원을 permutation(순열)하는 방법에 의존함
  - 편향제거 하면서 K개 fold와 ATT추정값들을 얻을 수 있었음
  - 이 k개 값의 분산과, 귀무가설(ATT=0하에서 자유도가 K-1인 asymptotic(점근) t분포를 갖는 검정 통계량 $\widehat{ATT}/\widehat{SE}$)
  - 이 방법은 기간별 추론(pre-period inference)에는 사용 못함
    - 기간별 추론을 위한 방법은 다른 논문 참고해야함

## 합성 이중차분법 (Synthetic DID, SDID)
SCM과 DID의 연관성에 대한 고찰, 두 방법을 합쳐 이중차분법 추정량으로 결합하는 방법을 다룸.  
이 방법은 [Dmitry Arkhangelsky et al., 2021](https://arxiv.org/pdf/1812.09970)의 간소화 버전임.

통제집단합성법 추정량을 다음과같이 최적화 문제를 푸는 식으로 바꿔도 ATT추정량은 동일하다.  
$$
\hat{\tau}^{sc}= \argmin_{\beta,\tau}\left\{ 
  \sum^N_{n=1} \sum^T_{t=1}(Y_{it}-\beta_t-\tau W_{it})^2 \hat\omega^{sc}_i
  \right\}
$$
  - $\tau$: ATT, 
  - $\beta_t$: 시간고정효과
  - $\hat\omega^{sc}_i$는 이미 최적화 결과에서 가져오는 것.
  - 이 목적함수는 모든 실험대상에 대해 정의되므로, 대조군과 실험군의 가중치도 고려해야함
    - 균일 가중치인 $N_{tr}/N$

여기서는 대상고정효과 $\alpha_i$가 빠져있음. 이를 추가한 것이 합성이중차분법 (SDID)임.  
  - $$ \hat{\tau}^{sc}= \argmin_{\beta,\tau}\left\{ 
    \sum^N_{n=1} \sum^T_{t=1}(Y_{it}-(\mu+\alpha_i+\beta_t+\tau W_{it}))^2 \hat\omega_i \hat\lambda_t
    \right\}
    $$
  - $\hat\lambda_t$: 시간 가중치
  - 시간고정효과, 대상고정효과, 실험대상 가중치, **시간 가중치**
  - 실험대상 가중치 $\hat\omega_t$ 의 목적은 대조군을 사용해서 실험군과 비슷하게 만들기 위함
  - 시간 가중치 $\hat\lambda_t$ 의 목적은 처치 전 기간의 가중치를 사용해서 처치 후 기간을 더 잘 근사
    - 구하는법: $\bold Y_{pre, co}$를 transpose하고 이를 처치 후 기간의 대조군 평균결과에 회귀
    $$\hat{\lambda}^{sc}_t= \argmin_{\omega}\left\| \bar y_{post, tr} - \bold{Y}'_{pre, co}\lambda_{pre} \right\|^2 \\ 
    \text{s.t. for all i,} \sum\lambda_i=1 \ {and} \ \lambda_i>0 $$
      - 근데 통제집단합성법은 외삽을 허용안함: 결과에 어떤 종류의 추세라도 존재하면 문제가 됨 (왜?)
      - 시간 가중치 구할 때 intercept shift(절편이동), $\lambda_0$을 허용
      $$\hat{\lambda}^{sc}_t= \argmin_{\omega}\left\| \bar y_{post, tr} - (\bold{Y}'_{pre, co}\lambda_{pre}+\lambda_0) \right\|^2 \\ 
    \text{s.t. for all i,} \sum\lambda_i=1 \ {and} \ \lambda_i>0 $$
      - 최적화 방식:  $(\lambda_{pre})$는 convex constraint 아래에서 최적화
        * non-negativity($\lambda_i>0$)
        * sum-to-one($\sum\lambda_i=1$)  
        - 이는 quadratic programming(QP) 문제다.  
      - $\lambda_0$는 unconstrained scalar → 해석적으로 구할 수 있다
      - 최소제곱식에서 intercept를 최적화하는 일반적 결과: $$ \hat\lambda_0 = \bar y_{\text{post},tr} - \overline{Y'_{pre,co} \lambda_{pre}} $$
      
        * synthetic combination의 평균을 실험군 평균에 맞춰주는 shift
        * residual 평균을 0으로 만드는 shift  

      - 따라서 $\lambda_0$는 단순히 synthetic curve 전체를 위·아래로 이동시키는 상수다.

합성이중차분법의 장단점:  
1. 합성이중차분법에 들어가 요소들로 인해 이중차분법이나 통제집단합성법 보다 bias낮음
2. 심지어 variance도 낮음
3. 절편이동을 허용하면, convex constraint가 깨진거라 외삽을 허용한 거라 안좋을수 있다.