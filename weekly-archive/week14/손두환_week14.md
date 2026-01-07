# Chapter 9. Synthetic Control Method (통제집단합성법)

궁금한점 / 논의점  
1. SCM하면 왜 평행추세가정이 필요없어짐?
  - 모델링 방향이 전혀 다른 horizontal regression
  - pre기간 weight학습하여 post기간 대조군으로 비교군 counterfactual 데이터를 생성
2. 처치 샘플이 하나여도 왜 잘 작동함?
  - 작동은 하지만, 추론 및 신뢰도 구하는 과정이 제한됨: 시간에 따른 bootstap 등
    - pre기간이 길다는 가정도 필요해짐
3. inverse propensity score에서 가상의 샘플을 만드는 것과 차이?
  - ips로 패널데이터에 적용하는 내용은 없었긴 함
4. convexity constraint (볼록성 제약조건)이 최적화 문제에 대한 sparsity 해결법을 어떻게 주는걸까? 수학적 배경?
5. 처치 전, 처치 후 샘플이 모두 비결측이란 가정이 panel 데이터를 씀으로 만족되는 것 같은데, pre에서 전혀 본 적 없는 새로운 sample을 반영해서 ATT계산하고싶으면 어떡할까? 
pre혹은 post기간에 결측이 생기면 어떡해야할까?

## 개요
이중차분법이랑 다른 방법!  

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
**이중차분법과 마찬가지로, 처치에 대한 ignorability 가정함. 거기에 파급효과가 없다고도 가정함.**

  - 대조군 잘 모의하면 실험군도 잘 모의한다는 가정
  - 각 실험 대상에 가중치 $\omega_i$를 부여해서 최적화 문제로 접근 $$\hat{\omega}^{sc}= \argmin_{\omega}\left\| \bar y_{\text{pre},\, tr} - Y_{\text{pre},\, co}\,\omega_{co} \right\|^2$$

    - $\bar y_{\text{pre},tr}$: $$
          \begin{pmatrix}
          \frac{1}{N_{tr}}\sum_{i\in tr} Y_{i,1}\\
          \frac{1}{N_{tr}}\sum_{i\in tr} Y_{i,2}\\
          \vdots\\
          \frac{1}{N_{tr}}\sum_{i\in tr} Y_{i,T_{\text{pre}}}
          \end{pmatrix}
          \in \mathbb{R}^{T_{\text{pre}}}
          $$
      * bar는 단면(cross-section) 평균
      * 시간축은 그대로 유지.
    - 처치 전 대조군에 $\omega_i$줘서 실험군 샘플과의 차이를 줄여서 최적화/fitting함
      - sc: synthetic control
    - 이를 처치 후 비교군 샘플을 입력값으로 넣어서 $Y(0)_{post,co}$를 추정하고 평균내서 사용함.
    - 이는 선형회귀의 목표와 같음
      - 대조군 결과를 특성으로 사용해서 실험군의 평균 결과를 예측하는 회귀
      - 기존 회귀분석의 design matrix: 행: N, 열: covariate
      - 지금 $\omega$구하는 식 design matrix: 행: 기간, 열: N
      - 통제집단합성법을 수평회귀분석 이라고도 함
    - 다만, 여기서 사용하는 모델은 일반적인 ML 모델일 수 없다. linear regression 식만 허용 : 해석 가능한 가중치 획득이 목적이니까 unrestricted ML(예: random forest, NN)은 원래 SC의 목적과 충돌.

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
- argmin 안에 좌항: $T_{pre}\times 1$, $\omega_{co}$: $N_{tr} \times 1$
- norm이 있으므로 시간별 차이를 모두 더한 스칼라를 최소화하도록 최적화 됨
- 제약조건의 목적: 가상의 대조군이 실험군에 대한 convex combination이 되도록 하여 외삽을 피하며 과적합 위험 감소
  - 즉 convex hull 바깥을 향한 이동이 되던걸 막음
    - 점의 개수는 $T_{pre}$ 가 아니라 $N_{co}$ 이며, $\omega_{sc}$ 는 시간별로 따로 나오지 않고 하나의 가중치 벡터
  - 실험군에 속한 모든 대상의 범위가 대조군의 범위 밖에 있다면, interpolation할 수 없고, 외삽으로만 추정 가능함: 이런 외삽을 막는다.
  - 가능한 가중치 범위 제한됨: 주어진 조건에서만 만족하는 특정값 찾음
  - 일반화 능력 향상, 추정값 신뢰도 증가
    - bias–variance tradeoff에서 variance를 줄이는 방향
- convexity constraint (볼록성 제약조건)이 최적화 문제에 대한 sparsity 해결법을 준다
  - 즉 최종 가상의 대조군을 만드는 데 소수의 도시만 사용된다.
  - 정확히는, convexity constraint 하에서 최적해가 종종 sparse해진다
  
- 정규화 효과로 인해, fitting (학습)오차는 조금 더 커져도, ATT추정 시계열의 잡음은 줄어듦

## SCM과 공변량(Covariates)
드문경우이긴 한데, 공변량의 예측력이 좋아서 모델에 초가 공변량을 포함해도 좋은 경우가 있다고 함.  

$$\hat{\omega}^{sc}= \argmin_{\omega}\left\| \bar y_{\text{pre},\, tr} - \sum\nu^*_k\bold{X}_{k, pre, co}\omega_{co} \right\|^2 \\
\text{s.t. for all i,} \sum\omega_i=1 \ {and} \ \omega_i>0
$$
- 가중치가 $y_{co}$와 더불어 공변량인 $x_{co}$에도 곱해짐 (y랑 X랑 한 배열로 묶음: [y_pre_co, x_pre_co])
  - $y_{co}$와 $x_{co}$가 완전 다른 척도에 있거나, 한쪽이 더 예측력이 좋을 수 있어서 최적화 문제를 풀기 전에 $y_{co}$를 포함한 각 공변량에 scaling factor $\nu$를 곱함
    - $\nu$: $\mathbb{R}^{1\times 2}$벡터로, y에 대한 가중치, X에 대한 가중치 하나씩임
  - $\nu$찾으려면 전체 통제집단합성법을 또 다른 최적화 목적함수로 묶어야함: 즉 최적화 따로 추가되는 것.

- 이 $\nu$찾은것으로 위 수식에 맞게 목적함수 재구성함

**참고: generic horizontal regression**  
: 공변량 추가하는 더 쉬운 방법  
  - 중요해보이는 추가 시계열 정보를 $Y_{pre, co}$에 붙이기  
  - 수평회귀분석에서 공변량을 추가하는것과 동일
  - $[\bold{Y}_{pre, co} | \bold{X}_{pre, co}]\omega$
  - 대조군과 추가 시계열을 사용하여 $E[Y(0)_{tr}]$를 추정하므로 엄밀하게는 통제집단합성법은 아님.
    - 실험대상이랑 추가한 열 각각에 대해서도 가중치 가져버리니까

## SCM과 편향 제거 (Bias Reduction)
$T_{pre}$수가 적으면 과적합이 발생할 수 있다. 이에 SCM은 편향될수 있다. 이 편향을 줄이는 방법: cross fitting
1. 개입 전 기간을 k블록으로 나눔. 각 블록의 크기는 $min(T_{pre}/K, T_{post})$ 
2. 각 블록을 hold-out셋으로 취급, 통제집단합성법 모델 fitting해서 블록 별로 $\hat{\omega}^k$를 얻는다 (즉 모델을 얻는다)
3. 각 블록별로 bias보정: hold-out데이터와 그 블록의 모델로 예측값을 얻는다.
  - 이 예측값이랑 관측값의 차이가 bias 추정값임
4. k번째 블록에서 기존 방식대로 구한 ATT에서 bias를 뺀 값이 $ATT^k$의 추정값이다
5. k별로 1~4과정을 거치고 k개 ATT추정값을 평균내서 최종 ATT값 하나를 얻는다.

## 추론(Inference)
- 편향제거 과정은 그 자체로 유용하지만, ATT추정값 주위에 신뢰구간 설정이 가능해서 더 좋은 방법이었음.  
- 그러나 추론에 문제점: 대조군이 적은 경우, 실험대상축으로 (시간축이 아닌)bootstrap하는 방법은 부적절하다. 
  - bootstrap 가정은 재표집이 표본분포를 근사한다는 것
  - 보통 대조군 샘플은 적은게 흔하므로 부적절
- 대부분 시간 차원을 permutation(순열)하는 방법에 의존함
  - Chernozhukov, Wüthrich, Zhu (2021)의 방법 소개함
  - 편향제거 하면서 K개 fold와 ATT추정값들을 얻을 수 있었음
  - 이 k개 값으로 귀무가설(ATT=0하에서 자유도가 K-1인 asymptotic(점근) t분포를 갖는 검정 통계량 $\widehat{ATT}/\widehat{SE}$)을 구함
  - 이 방법은 기간별 추론(per-period inference)에는 사용 못함
    - mean(axis=0)하고 k개로 분산 구했으니...
    - 기간별 추론을 위한 방법은 다른 논문 참고해야함

## 합성 이중차분법 (Synthetic DID, SDID)
SCM과 DID의 연관성에 대한 고찰, 두 방법을 합쳐 이중차분법 추정량으로 결합하는 방법을 다룸. [Dmitry Arkhangelsky et al., 2021](https://arxiv.org/pdf/1812.09970)의 간소화 버전임.

**SCM은 단위 방향 balancing, DID는 시간 방향 balancing인데, SDID는 둘 다 한다**
- SCM
  - 단위별로 가중치 $\hat \omega_i$
    - 단위 간 수준 차이를 보정(match)하는 역할은 하지만, 이 알고리즘이 단위고정효과를 포함했다고는 얘기 안한다.
  - 시간 고정효과만 남음 왜?
    - 시간고정효과는 이질성(heterogeneity)을 개체 간이 아니라 시간 간에 둔다 (특정 시간에 모든 개체에 동일한 영향, 시간에 따라서는 다른 영향)
    - 표준 SCM은 시간에 따른 공통 변화 구조를 전혀 제한하지 않기 때문에(모델링에 사용 안했기 때문에), 그 자체가 시간고정효과를 완전히 자유롭게 허용한 모형
- DID
  - 단위·시간 고정효과 반영
  - 가중치 없음
- SDID
  - 단위·시간 고정효과 반영
  - 단위 가중치 + 시간 가중치
  - 두 세계를 결합

통제집단합성법 추정량을 다음과같이 최적화 문제를 푸는 식으로 바꿔도 ATT추정량은 동일하다.  
$$
\hat{\tau}^{sc}= \argmin_{\beta,\tau}\left\{ 
  \sum^N_{n=1} \sum^T_{t=1}(Y_{it}-\beta_t-\tau W_{it})^2 \hat\omega^{sc}_i
  \right\}
$$
  - $\mu$: 모든 단위·모든 시간에 공통인 baseline. 절편.
  - $\tau$: ATT, 
  - $\beta_t$: 시간고정효과
  - $\hat\omega^{sc}_i$는 별도의 최적화로 구함
  - 이 목적함수는 모든 실험대상에 대해 정의되므로, 대조군과 실험군의 가중치도 고려해야함
    - 균일 가중치인 $N_{tr}/N$

여기서는 대상고정효과 $\alpha_i$가 빠져있음. 이를 추가한 것이 합성이중차분법 (SDID)임.  
  $$ \hat{\tau}^{sc}= \argmin_{\mu, \alpha, \beta,\tau}\left\{ 
    \sum^N_{n=1} \sum^T_{t=1}(Y_{it}-(\mu+\alpha_i+\beta_t+\tau W_{it}))^2 \hat\omega_i \hat\lambda_t
    \right\}
    $$
  - $\hat\lambda_t$: 시간 가중치
  - 실험대상 가중치 $\hat\omega_t$ 목적: 대조군으로 실험군과 비슷하게 만들기
  - 시간 가중치 $\hat\lambda_t$ 목적: 대조군(control)에서 pre-period의 가중합이 post-period 평균을 근사하도록 선택
    - 구하는법: $\bold Y_{pre, co}$를 transpose하고 이를 처치 후 기간의 대조군 평균결과에 회귀
    $$\hat{\lambda}^{sc}_t= \argmin_{\lambda}\left\| \bar y_{post, tr}' - \bold{Y}'_{pre, co}\lambda_{pre} \right\|^2 \\ 
    \text{s.t. for all i,} \sum\lambda_i=1 \ {and} \ \lambda_i>0 $$
      - 근데 통제집단합성법은 외삽을 허용안함
        - 어떤 종류의 추세가 존재해서, 처치 후 기간 전체가 처치 전 기간보다 높거나 낮을 수 있어서 외삽일 필요해질 수 있다.
      - 위 문제를 해결하기 위해 시간 가중치 구할 때 intercept shift(절편이동), $\lambda_0$을 허용
      $$\hat{\lambda}^{sc}_t= \argmin_{\lambda}\left\| \bar y_{post, tr}' - (\bold{Y}'_{pre, co}\lambda_{pre}+\lambda_0) \right\|^2 \\ 
    \text{s.t. for all i,} \sum\lambda_i=1 \ {and} \ \lambda_i>0 $$
      - 최적화 방식:  $(\lambda_{pre})$는 convex constraint 아래에서 최적화
        * non-negativity($\lambda_i>0$)
        * sum-to-one($\sum\lambda_i=1$)  
      - $\lambda_0$
        * 계산방법: intercept칼럼 하나를 X에 horizontal concat해서 회귀돌리면 $(\lambda_{pre})$랑 함께 계산됨
        * synthetic combination의 평균을 실험군 평균에 맞춰주는 shift
        * residual 평균을 0으로 만드는 shift  

### 합성이중차분법의 장단점
1. 많은 설정에서, 합성이중차분법에 추가한 요소들로 인해 이중차분법이나 통제집단합성법 보다 bias낮음
2. 많은 설정에서, 심지어 variance도 낮음
3. 평행추세가정이 완전 잘맞아서 DID가 unbiased면 SDID는 별로
  - 처리군·대조군 규모가 크고 균형 잡힌 패널인 경우도.

#### SDID가 SCM보다 나쁜 경우
**단일 처리 유닛 + 매우 잘 맞는 donor pool**  
표준 SCM이 잘 작동하는 전형적 상황:
* 처리 유닛 1개
* 대조군이 많음
* pre-path가 거의 완벽히 재현됨

이 경우 SCM의 bias ≈ 0, variance도 비교적 작음

SDID는 여기에
* 단위 고정효과
* 시간 가중치
* 추가 회귀
  를 얹음 → **불필요한 자유도 사용**

결과적으로, SCM보다 변동성 증가 가능

**post-period가 매우 짧은 경우**  
SDID의 시간 가중치는 control에서 pre → post 구조를 학습함.  
하지만 post-period가 1~2개 시점뿐이면 시간 가중치 문제 자체가 정보 부족  
이로인해 시간 가중치 추정이 불안정하며, 오히려 noise만 증폭

#### 구조적으로 SDID가 취약한 경우
**단위·시간 모두에서 convex hull이 약한 경우**

SDID는 단위 방향 convex hull, 시간 방향 convex hull 모두 필요함.

만약, 처리군이 단위 방향에서도 out-of-support, post 기간이 시간 방향에서도 out-of-support 이면  
SDID는 두 축에서 동시에 어렵고, 오히려 단순 DID가 덜 나쁠 수 있음

**데이터가 매우 noisy한 경우**  
SDID는 pre-period fitting을 적극적으로 활용하는데, pre 데이터가 noisy하면 가중치가 noise에 맞춰지며, post 추정도 불안정해짐.  
반면 DID는 평균 기반 → 노이즈에 상대적으로 둔감

**추론(inference) 관점에서의 단점**  
SDID는 가중치추정, 회귀추정이 중첩된 절차를 가지므로, 표준 오차 계산이 복잡함.  
그리고 작은 표본에서 asymptotic 근거가 약함  
이 경우 단순 DID의 클러스터-robust SE가 해석·보고 측면에서 더 낫기도 함

최종 정리 (표)
| 상황                | 더 나은 방법    |
| ----------------- | ---------- |
| 평행추세 매우 잘 성립      | DID        |
| 단일 처리 + donor 풍부  | SCM        |
| 패널 큼, 구조 단순       | DID        |
| pre-path mismatch | SDID       |
| 단위·시간 이질성 모두 큼    | SDID       |
| post 기간 매우 짧음     | DID 또는 SCM |
> **합성이중차분법은 DID와 SCM의 “안전한 상위호환”이 아니라, 두 방법이 각각 실패할 수 있는 상황을 보완하는 방법이며, 기존 가정이 이미 잘 맞는 경우에는 오히려 불필요한 복잡성 때문에 성능이 나빠질 수 있다.**
