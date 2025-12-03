# Chapter8: Difference-in-Differences (이중차분법), Pannel data(패널데이터)
- 패널데이터: **동일 대상**을 **시간에 따라 반복해서 관측** 하는 데이터 구조  
- **동일 대상**에 처치 이루어지기 전, 후에 무슨 일이 일어나는지 알 수 있음
- 랜덤화 불가능할 때 인과효과 식별하는 좋은 대안
- 관측데이터가 있고, 관측되지 않은 교란요인 가능성이 높다면 패널데이터가 처치효과 식별에 좋음

* 용어 정리  

| 용어 | 시간변화(시간축 존재 여부) | 동일 개체 반복 관측 여부 | 특징 / 추가 설명 |
| - | - | - | - |
| **Cross-sectional (횡단면)** | X | X | 한 시점에서 서로 다른 개체를 관측함. 시간적 구조 없음. |
| **Time series (종단면)** | O | 1개 개체 | 단일 개체(국가·기업·상품 등)를 시간에 따라 관측. |
| **Repeated cross-sectional** | O | X (대체로 다른 개체) | 각 시점은 cross-section이지만 동일 개체가 아님. 예: 매년 다른 사람 표본으로 설문조사. |
| **Longitudinal** | O | O (same subjects) | 동일 개체 다수를 반복 관측. 개인 수준 관찰을 지속적으로 추적. 주로 '개인'수준이라 Panel과 다름. 예: 같은 개인 1,000명을 10년간 매년 조사|
| **Panel data** | O | O (multiple subjects) | 여러 개체(Longitudinal과 달리, 개인이 아니라 무엇이든 될 수 있다)를 시간에 따라 반복 관측. Longitudinal이 포함되는 더 넓은 개념. |
| **Pooled cross-sectional** | △(거의 의미없음) | X | 서로 다른 시점의 cross-sectional 데이터를 '그냥 합쳐놓은' 자료. 개체 추적 없음. 데이터 수량 늘리기용|
| **Pseudo-panel** | △ | △(cohort만 same) | 개인은 매년 다르지만, 동일한 집단(cohort) 을 기준으로 평균을 취하면 마치 그 집단을 시간에 따라 반복 관측한 '패널처럼' 사용할 수 있다|


* Longitudinal과 Panel은 데이터 구조상 거의 동일하지만,
연구 목적·도메인·통계모형의 철학이 다르다
* 사용하는 통계기법도 일부 겹치지만, Longitudinal에서는 **개인 변화 추적·시간 내 상관·결측·dropout** 처리가 핵심이고,  
Panel에서는 **불변효과(FE) 제거·이질성·정책효과 추정(DID 등)** 이 핵심이다.

| 항목    | Longitudinal           | Panel                 |
| ----- | ---------------------- | --------------------- |
| 관측 단위 | 보통 개인                  | 개인/기업/국가/지역 등 무엇이든    |
| 목적    | 개인의 변화(trajectory) 분석  | 인과추론·정책효과·불변효과 제거     |
| 통계기법  | LME, growth curve, GEE | FE/RE, DID, SCM, SDID |
| 시간 구조 | 불규칙 관측도 가능             | 일반적으로 정규 시간축          |
| 결측 처리 | dropout 모델링 중요         | 상대적으로 단순              |
| 포함 관계 | 패널의 특수한 형태             | 더 큰 개념                |

궁금/논의할점  
1. 이중차분법의 가정인 $E[\Delta y_0] = E[\Delta y\mid D = 0]$ 
  - $\Delta$: 시간에 따른 변화
    - cross-sectional 같은 데이터는 무의미. time-series도 무의미. repeated cross-sectional/패널 데이터만 해당하는 가정
  - 좌항 = 보이지 않는 counterfactual (신만 아는 값)
  - 우항 = 데이터에서 대조군을 통해 관측 가능한 값
  - 왜 같다고 놓나? TODO:FIXME: 
  
## 1. 패널데이터와 DID의 기본 구조
### 기호 정의
* 관측값: $Y_{it}$
* 시간: $t$
* 개체(개인·기업·지역) index $i=1,\dots,N$
* 시점 index $t=1,\dots,T$
  - $T_{\mathrm{pre}}$: pre-intervention, 개입 이전 기간
  - $T_{\mathrm{post}}$ 또는 $T_{\mathrm{pre}} + 1, \dots, T$: post-intervention, 개입 이후 기간
* 처치변수: $D_{it}\in\{0,1\}$
  - 패널데이터에서는 처치를 개입(intervention)이라고도 함
  - D=1: 비교군. 처치 전후 따라서 Y(0)일수도 Y(1)수도 있다
  - D=0: 대조군. 처치 전후 모두 Y(0)
* 처치변수 및 개입 후 에 대한 조합 표기법
  - $W_{it} = D_{it}\,\mathbf{1}(t > T_{\mathrm{pre}})$ 또는 $W = D\times Post$
* 공변량: $X_{it}$

관심 효과: **처치된 집단에 대한 ATT**
$$
ATT = E\big[\, Y_{it}(1) - Y_{it}(0)\ \big|\ D=1, t>T_{pre}\big]
$$  

-  개입 후 기간의 실험군에서만 Y(1)을 관측할 수 있고, 그 외 조건 3개(개입 후 대조군, 개입 전 대조군, 개입 전 실험군)에서는 Y(0)만 관측할 수 있다.  
- 이 3개에서 누락된 잠재적 결과인 $E[Y(0)|D=1, t>T_{Pre}]$를 추정할 수 있다.
  - 시간 간 관계: 개입 전 실험군 결과 활용 (쉬움)
  - 실험 대상 간 관계: 개입 후 대조군 결과 활용 (어려움)

## 2. 이중차분법 (Difference-in-Differences, DID)
이중차분법: 관측된 '실험군' 기준값에 '대조군' 결과 추세를 보정하여, 누락된 잠재적 결과인 $E[Y(0) \mid D=1, Post=1]$를 추정하는 것.  

  - 평행추세 가정: 두 그룹의 사전/사후 변화량이 같아야 한다
    - $E[\Delta y_0\mid D = 1] = E[\Delta y_0 \mid D = 0]$

  - 일치성(consistency): 대조집단에서는 처치받지 않은 잠재적 결과와 실제 관측값이 같다
    - $E[\Delta y_0 \mid D=0] = E[\Delta y \mid D=0]$  
    - 참고: $E[\Delta y_1 \mid D=1] = E[\Delta y \mid D=1]$ 도 일치성

  - 즉
    - $E[\Delta y_0 \mid D=1] = E[\Delta y_0 \mid D=0] = E[\Delta y \mid D=0]$
    - 따라서 실험군의 반사실적 변화량을 대조군의 관측된 변화량으로 대체할 수 있다

### 1차 차분(formal)
DID 기반 잠재적 결과 추정식

$$
E[Y(0)\mid D=1,\ \text{Post}=1] = E[Y\mid D=1,\ \text{Post}=0] + \big( E[Y\mid D=0, \text{Post}=1] - E[Y\mid D=0,\ \text{Post}=0] \big)
$$
* $E[Y\mid D=1,\text{Post}=0]$: 실험군 기준값 (사전 평균)
* $E[Y\mid D=0,\text{Post}=1]-E[Y\mid D=0,\text{Post}=0]$: 대조군의 결과 추세

**ATT(처치효과) DID 추정량**  
관측된 실험군 사후결과는 $E[Y(0)\mid D=1,\ \text{Post}=1]$  
따라서 ATT는
$$
\begin{aligned}
ATT &=
\big( E[Y\mid D=1,\ \text{Post}=1] - E[Y\mid D=1,\ \text{Post}=0] \big) \\
 &- \big( E[Y\mid D=0,\ \text{Post}=1] - E[Y\mid D=0,\ \text{Post}=0] \big)
\end{aligned}
$$

아래 표 (2x2 DID라 부름)로 본다면, 
$$
\begin{array}{c|cc}
 & \text{개입 전 (pre)} & \text{개입 후 (post)} \\ \hline
\text{실험집단 }(D=1) & Y(0)_{(1)} & Y(1)_{(2)} \\
\text{대조집단 }(D=0) & Y(0)_{(3)} & Y(0)_{(4)} 
\end{array}
$$
즉 ATT = (2) − (1) − (4) + (3).  
차이의 차이를 구한다고 DID추정량 이라하며, 이 형태를 canonical(표준) 형태라고 함.  

### 결과 변환
실험대상 i의 시간경과에 따른 결과 차이를 $\Delta y_i = E[y_i \mid t> T_{pre}]-E[y_i \mid t \leq T_{pre}]$ 라 하자. 
그러면 $ATT = E[\Delta y_1 - \Delta y_0]$인데, 
이중차분법은 $\Delta y_0$를 대조군 평균으로 대체하여 ATT 식별 가능.  
$$ ATT = E[\Delta y \mid D= 1] - E[\Delta y \mid D= 0]$$
이는 canonical form에서 구한 DID추정값과 동일함 (2-4-(1-3)).  
이는 이중차분법의 가정인 $E[\Delta y_0] = E[\Delta y\mid D = 0]$를 보여주는 접근법임.  

### OLS모형
포화회귀모델로 DID추정량을 얻을수도 있음.  
$$Y_{it} = \beta_0 + \beta_1D_i +\beta_2\text{Post}_t+\beta_3D_i\text{Post}_t +e_{it}$$
- D랑 Post는 0 또는 1값
- $\hat\beta_3$: DID 추정값
  - 실험군과 대조군 간의 차이와 시간 추세를 모두 고려한 DID추정량
- $\beta_0$: 대조군 기준값
- $\beta_0 + \beta_1$: 개입 전 실험군 도시 기준값
- $\beta_1$: 단순히 실험군 도시와 대조군 도시 간의 기준값 차이
- $\beta_0 + \beta_2$: 개입 후 대조군 도시 다운로드 수준
- $\beta_2$: 시간이 지나며 발생하는 개입 후 변화량
- $\beta_0 + \beta_1 +\beta_2 +\beta_3$: 개입 후 실험군 도시의 다운로드 수준

### Two-Way Fixed Effects (TW FE, 이원고정효과) 모형
시간-대상 고정효과 모델이라고도 함.  
일반적 패널 DID에서는 다음과 같은 이원 고정효과(TWFE) 회귀를 쓴다.

$$
Y_{it} =\tau W_{it} + \alpha_i + \gamma_t + \varepsilon_{it}
$$
$$
Y_{it} = Y_{it}(0) + W_{it}\big( Y_{it}(1) - Y_{it}(0) \big)
$$

$$
Y_{it}(0) = \alpha_i + \lambda_t + u_{it}
$$
- $\tau$: 처치효과, DID추정값과 같다
- $\alpha$: 개별 대상 효과, "관측되지 않은, 하지만 시간 불변인 교란요인"
  - 시점별 차분 또는 고정효과 회귀로 $\alpha_i$를 제거할 수 있다
- $\gamma$: 시간 고정효과
- $\varepsilon_{it}$ : 개체·시간 고정효과로도 설명되지 않는 잔차(error term 또는 idiosyncratic shock)
- $W_{it} = D_i\text{Post}_t$
- 이 모델을 추정하면 W와 관련된 매개변수 추정값이 앞의 DID추정값과 일치하고 ATT구할 수 있음
  - 더미변수를 사용하거나, 데이터 평균을 제거하여 고정효과를 추정할 수 있다 (4장 참고)
    - 추가 설명:  
    **Ⅰ. 더미변수 회귀 (Dummy-variable regression)**
    모형:
    $$
    Y_{it} = \tau W_{it}
    + \sum_{i=1}^{N}\alpha_i \mathbf{1}(i)
    + \sum_{t=1}^{T}\gamma_t \mathbf{1}(t)
    + \varepsilon_{it}
    $$

    * 각 개체마다 더미 하나씩 → $\alpha_i$ 추정
    * 각 시점마다 더미 하나씩 → $\gamma_t$ 추정
    * 최소 한 개는 reference로 제거해야 다중공선성 방지

    **Ⅱ. Within Transformation (이원 고정효과의 표준적 제거법)**  
    Demeaning(평균제거)을 두 단계로 수행한다.

    1. 개체효과 제거: 개체-평균 제거 (entity demeaning)
    $$
    \tilde{Y}_{it} = Y_{it} - \bar{Y}_{i\cdot} \\
    \tilde{W}_{it} = W_{it} - \bar{W}_{i\cdot}
    $$

    여기서
    $$
    \bar{Y}_{i\cdot} = \frac{1}{T}\sum_t Y_{it}
    $$

    → 이 변환을 하면 $\alpha_i$가 제거된다:

    $$
    \tilde{Y}_{it}
    = \tau\, \tilde{W}_{it}
    + (\gamma_t - \bar{\gamma})
    + (\varepsilon_{it} - \bar{\varepsilon}_{i})
    $$

    2. 시간효과 제거: 시간-평균 제거 (time demeaning)

    $$
    \ddot{Y}_{it} = \tilde{Y}_{it} - \tilde{Y}_{\cdot t} \\
    \ddot{W}_{it} = \tilde{W}_{it} - \tilde{W}_{\cdot t}
    $$

    여기서
    $$
    \tilde{Y}_{\cdot t} = \frac{1}{N}\sum_i \tilde{Y}_{it}
    $$
    → $\gamma_t$도 제거된다:

    $$
    \ddot{Y}_{it}
    = \tau\, \ddot{W}_{it}
    + (\varepsilon_{it} - \bar{\varepsilon}_{i} - \bar{\varepsilon}_{\cdot t} + \bar{\varepsilon})
    $$

    → 이제 **$\tau$만 남는다.**

| 특징    | Canonical DID (2×2) | TWFE DID              |
| ----- | ------------------- | --------------------- |
| 시간    | pre 1개 + post 1개    | 여러 pre/post 가능        |
| 그룹    | 2개(실험, 대조) | 여러 treated·control 가능 |
| 추정 방식 | 단순 차이의 차이 | 회귀 기반 FE 제거 |
| 가중치   | 명확한 1,−1 구조 | 복잡한 가중치, 음수 가능 |
| 동일한가? | TWFE와 일치  | 2×2 상황에서만 identical   |


### Block Design (블록 디자인)
* 블록디자인
  - 기존의 2×2 DID 표를 시간·집단 차원에서 여러 개의 작은 2×2 블록들로 분해해, 각 블록의 DID 추정치를 가중평균하는 구조로 확장한 것
  - 비슷한 특성을 가진 단위들을 블록으로 묶고, 블록 내에서만 실험군/대조군 비교하는 방법
    - 시간window에 따라, control subset에 따라, 둘 다에 따라 다양하게 나눔
  - DID나 실험에서 원하는, 처치받은 그룹과 안 받은 그룹이 비슷한 특성을 가져야 하는 조건을 만족시키기 위해
  
  - 예시
    - A도시 / B도시 / C도시 각각 동일한 기간을 관측했는데, 어떤 도시(A)는 개입이 있었고 다른 도시(B,C)는 없었다면 도시가 블록이 됨.
    - 블록(도시) 안에서는 기간이 동일하므로 평행추세 가정을 훨씬 설득력 있게 만든다.

* 절차
  - 공변량 $X$ 의 특정 범주(예: 지역, risk level)에 따라 블록을 나누고,
  - 각 블록 내부에서 DID를 수행한 뒤,
  - 블록별 추정치를 가중평균

* 평행추세 가정을 "블록 내에서만" 요구하는 conditional DID 접근, 혹은 **stratified DID** 라고 해석할 수 있다.
* 장점: 대규모 이질성이 있는 경우, 블록 내에서 더 믿을 만한 평행추세를 가정할 수 있다

### Inference, confidence interval
- 패널데이터에서의 추론이 굉장히 까다로우므로, 신뢰구간이 부정확할 수 있음.  
  - 실험 대상이 같으면 시간별 관측값이 서로 독립이 아님
  - OLS는 이걸 무시하므로 표준오차를 과소추정 → 너무 작은 p-value → 과신하게 됨.
  - (최근 계속 연구되는 분야라고 함!)  
- NT개 데이터 포인트가 있지만, 동일 실험대상이 여러 번 나타나므로 IID를 고려하면 N개 샘플이 있다고 봐야함.
- 실제로 처치는 시간대가 아닌, 실험 대상에 배정된 것.
- 보정방법: 실험대상 별로 clustering해서 clustered standard error(군집표준오차)를 구함
  - 군집표준오차: 군집 내 서로 상관되어 있을 때, 이를 보정하면서 표준오차가 증가하여 기존 표준오차보다 신뢰구간이 넓어짐
  - 각 군집당 더 많은 시간대가 있으면 분산을 줄일 수 있음
  - 키워드: Cluster-robust variance estimator
    $$
    \widehat{Var}(\hat{\beta})=
    (X'X)^{-1}
    \left(
    \sum_{g=1}^{G} X_g' \hat{u}_g \hat{u}_g' X_g
    \right)
    (X'X)^{-1}
    $$
    * (g): 클러스터 index (예: 개체)
    * (X_g): 클러스터 g의 디자인 행렬
    * (\hat{u}_g): 클러스터 g의 잔차 벡터
    * (G): 클러스터 개수

- 다른방법: 블록 부트스트랩 (전체 추정과정을 부트스트랩)
  - 개채별로 블록으로 지정하고 복원추출 하여 신뢰구간을 계산!
  - 단점: 실험군의 수가 적으면, 표본에 실험군이 포함되지 않을 수 있음
    - 이를 해결하는 좋은 방법은 아직 없는듯?
- 참고논문
  - [Abadie, A., et al., 2022](https://arxiv.org/pdf/1710.02926)
  - [Chernozhukov V., et al., 2021](https://arxiv.org/pdf/1712.09089)

## DID의 식별 가정
### 1. 평행 추세 (Parallel Trends)
가정: 처치가 없으면 평균적으로 실험군, 대조군의 결과 추세가 동일하다.
$$
E[Y_{it}(0)-Y_{i,t-1}(0)\mid D=1] = E[Y_{it}(0)-Y_{i,t-1}(0)\mid D=0]
$$
($E[Y_{it}(0) \mid D=1]$은 구할 수 없으므로 검증은 불가능)  

### 2. 비기대 가정(no antipation) / SUTVA
no interference 또는 Stable unit of treatment value assumption (SUTVA, 상호 간섭 없음)  
  - 하나의 실험 대상에 대한 효과는 다른 실험 대상의 영향을 받지 않는다.
  - 추가: consistency 
    - 실제 받은 처치값에 대응하는 잠재적 결과가
실제 관측된 결과다. $$
          Y_{it} =
          \begin{cases}
          Y_{it}(1), & \text{if } W_{it} = 1 \\
          Y_{it}(0), & \text{if } W_{it} = 0
          \end{cases}
          $$
    - 마스크 사용을 처치 했는데, 턱마스크 하는 개체 같은 예외가 생기지 않는다고 가정
  - 만족하기 까다로운 조건임
  - 시공간에 걸쳐서 SUTVA위배 발생 가능
    - 블랙프라이데이 전인데 판매량 증가
    - 사람들이 이 지역 저 지역 넘나들어서 대조군 지역도 처치 효과를 받아버림
      - 공간적 파급효과 참고논문 [Butts. K, 2021](https://arxiv.org/pdf/2105.03737)

### 3. Strict Exogeneity (강외생성)
주로 고정효과모델에서의 잔차에 대한 가정이고, 더 강력한 가정이며 평행추세를 내포함.  
1. 시간에 따라 변하는 교란요인 없음
  * 시간에 걸쳐 반복되는 관측값이 있으면, 관측되지 않은 교란요인이 존재해도 **시간에 따라 일정하다면** 인과효과 식별이 가능함!
  * 개체별 고정효과 $\alpha_i$로 표현 가능한 부분은 DID/FE로 제거 가능
    - 더미변수를 추가
    - 평균을 제거 (실험 대상 별 결과 및 처치변수의 평균을 계산하여 원래 변수에서 빼기)  
    결과에서 시간고정 공변량 제거: $\ddot Y_{it} = Y_{it} - \bar Y_{i}$  
    처치에서 시간고정 공변량 제거: $\ddot W_{it} = W_{it} - \bar W_{i}$
    - Y나 W에 시간에 따라 일관되게 영향을 주던 효과들이 빼기로 사라져버림 $(A+a -(A'+a) = A - A')$
    - 마찬가지로, 시간 고정효과도 이런 원리로 가능하다.
  * 하지만 **시간에 따라 변하는 교란**은 그대로 남는다.

2. 피드백 없음
  * 미래의 정책 시행 이전에는 과거 결과를 참고로 한 행동이 없어야 한다.
  * 처치가 이후의 공변량이나 다른 구조에 직접 영향을 미쳐, 다시 $Y$ 에 피드백하는 구조는 단순 DID 모형에선 보통 고려하지 않는다 (보다 정교한 구조모형 필요).
    -  sequential ignorability 등
    - [Xu. Y., 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3979613)
  
3. carry over(이월효과) 없음
  * 과거 처치가 현재 결과에 영향 주지 않음
  * 가장 단순한 DID는 "$T$가 1이 되는 시점에만 효과가 발생하고, 이후 동태효과는 동일하다"는 가정을 두는 셈이다.
    * 처치의 lag버전을 모델에 포함시켜서 완화할수는 있음 
    $$
    Y_{it} =\tau W_{it} + \theta W_{it-1} + \alpha_i + \gamma_t + \varepsilon_{it}
    $$

  * 추가로, (optional인거같지만), lagged dependent variable(시차종속변수)가 없다 가정
    - 과거 결과가 현재 결과의 직접적인 원인이 되지 않는다
    - 그러나, 화살표 있어도 큰 문제는 아니라고 함. (왜?)
      - 시간 고정효과(λₜ) + 개체 고정효과(αᵢ)가 이미 Y의 자기상관 구조 대부분을 흡수하기 때문
      - LDV(아래 식에 $\rho Y_{i,t-1}$를 말함)포함 시 FE 패널에서 Nickell bias 발생 → 더 위험
       $Y_{it} = \rho Y_{i,t-1} + \tau W_{it} + \alpha_i + \lambda_t + u_{it}$
      - 평행추세는 AR 구조와 직접적으로 충돌하지 않음
      - 실무·논문에서 LDV를 쓰지 않는 것이 표준
