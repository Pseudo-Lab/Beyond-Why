# Chapter3 Graphical Causal Model
그래프 인과모델: 인과추론 문제를 구조화 하고 식별 가정을 명쾌하고 시각적으로 표현함.    
Structural Causal Model (SCM) 이라고도 함.  
directed acyclic graph (DAG) 라고도 함.  

* 그래프 인과모델에서 연관성이 어떻게 흐르는지 (information flow), 그래프를 query하는 방법 배움  
* 그래프 모델로 identification을 재해석  
* 식별을 방해하는 두가지 편향읜 원인 배우기, 인과 그래프 구조로 할 수 있는일 배우기  

## faithfulness(충실성) 가정  
**faithfulness**는 인과 그래프를 데이터로부터 학습할 때(특히 PC/FCI 같은 constraint-based 방법) 자주 쓰이는 핵심 가정이다.  

### 핵심 아이디어  
그래프가 말하는 **d-separation(그래프적 독립)** 과 실제 데이터 분포에서 관측되는 **통계적 독립/조건부 독립**이 **일치**한다는 가정이다.  

- 그래프에서 d-separation이면  
  $$
  X \perp Y \mid Z
  $$
  가 분포에서도 성립해야 한다.  
- 반대로(보통 알고리즘에서 더 중요): 분포에서 독립이 관측되면, 그 독립은 그래프 구조(d-separation)로 “설명 가능한” 독립이어야 한다.  
  즉 **우연한 수치적 상쇄(cancellation)** 때문에 독립이 생기면 안 된다는 뜻이다.

### 왜 필요한가? (상쇄 문제)  
선형 모형에서는 계수 값이 “우연히 딱 맞아 떨어지면” 그래프상으로는 경로가 열려 있어 **연관성이 있어야 하는데도**, 관측 상관/의존이 0이 되어버릴 수 있다.  

- 예: 두 개 이상의 경로가 존재할 때, 한 경로의 효과와 다른 경로의 효과가 크기/부호가 맞아 **서로 상쇄**되면  
  $$
  \text{Corr}(X,Y)=0
  $$
  처럼 보일 수 있다(혹은 어떤 조건부 상관이 0).  
- 그러면 학습 알고리즘은 “독립이네 → 엣지가 없거나 경로가 차단됐네”라고 해석해서, 실제로는 연결된 구조를 **연결이 없다고 오판**할 수 있다.


## 인과관계 시각화  
노드: 확률변수, 엣지: 원인 -> 결과  
측정되지 않은 모든 변수를 U node라 함 (주로 생략하는듯)  
T: treatment  
Y: Outcome  
X: 관측 가능한 변수들 집합  

**인과그래프에서 연관성이 어떻게 흐르는가?**   
즉, 어떤 독립성 및 조건부 독립성 가정이 수반되는지 이해해야함.  
경로는 다음과 같은 필요충분조건에 따라 차단될 수 있다. (d-separation 조건)  
d-separation: 그래프에서 두 변수 간의 독립성을 표현하는 또 다른 용어  
1. 조건으로 주어진 non-collider 구조 (chain, fork)
2. 조건으로 주어지지 않고 descendant가 없는 collider 구조

구체적인 3가지 경우는 chain, fork, collider가 있음!  
### chain (사슬) 구조  
$$T \to M \to Y$$
M: intermediate node, 중간노드  
인과관계는 화살표 방향으로만 흐르지만 연관관계는 양방향으로 흐름.  

두 변수가 서로 연관되면, 두 변수는 독립이 아니다 라고 함.  
$$T \not\!\perp Y$$
<br/>

여기서 매개자가 추가된다면?  
**매개자 고정** : 특정 변수의 값을 유지하면서 다른 변수들 간의 관계를 관측하는 것. 이 과정으로 해당 고정 변수의 영향을 통제하고, 다른 변수들 사이의 관계를 더 명확하게 이해할 수 있다. '조건부로 설정'한다고도 함. **이 경우 dependence가 block됨**
$${T \perp Y} \mid M $$
그래프 그림으로는 검은색 노드가 됨.  

예시: 문제 해결력이 동일한 사람들을 살펴볼 때 (M이 동일한 샘플들에서) 어떤 사람이 인과추론에 능숙하다고 해서 (T값이 높다고) 승진가능성이 어떻다는 추가 정보를 얻을 수 없다 (Y값의 패턴을 알 수는 없다): $E[Y|M, T] = E[Y|M]$  
반대로도, M을 알 때 Y를 알아도 T에 추가 정보를 제공하지 않는다.  

즉, “중간 원인을 알고 있으면, 시작점이 결과에 주는 추가정보는 사라진다.”

### fork (분기) 구조  
common cause가 있는 구조. 
$$
\begin{array}{ccccc}
 & & X & & \\
 & \swarrow & & \searrow & \\
 T & & & & Y
\end{array}
$$
인과추론에서는 T, Y 사이에 공통원인이 있을 때, 이 공통원인을 confounder (교란요인) 이라고 함.  
공통 원인을 공유하는 두 변수는 종속이지만, 공통원인(X)이 주어지면 독립.  
$$T \not\!\perp Y$$
$$T \perp Y \mid X$$

### collider (충돌부) 구조  
두 노드가 하나의 자식을 공유하지만, 그들 끼리는 직접적인 관계가 없는 경우.  
$$
\begin{array}{ccccc}
 T & & & & Y \\
 & \searrow & & \swarrow & \\
 & & X & & \\
\end{array}
$$
즉, 두 변수가 공통의 효과를 공유함. 이 공통 효과는 두 개의 화살표가 충돌하므로 충돌부 라고 부름. 
두 부모 노드는 서로 독립이지만, 공통 효과를 조건부로 두면 서로 종속이 된다.  
Y랑 T는 평소에는 독립인데 갑자기 X가 결정되면 Y로 T를 알 수 있게 된다. 이를 **explaining away**현상이라고도 한다 
$$T \perp Y$$
$$T \not\!\perp Y \mid X$$

충돌부 (X)가 아니라 충돌부의 결과 (X의 다음 노드)를 알 때도 Y와 T는 종속이 된다.  
$$
\begin{array}{ccccc}
 T & & & & Y \\
 & \searrow & & \swarrow & \\
 & & X & & \\
 & & \downarrow & & \\
 & & Z & &
\end{array}
$$

## identification 재해석  
$$
\begin{array}{ccccc}
 & & X & & \\
 & \swarrow & & \searrow & \\
 T & \rightarrow & & & Y
\end{array}
$$

$T \to Y$ : causal path  
$X \to Y$ : non-causal path, **backdoor path**, 공통원인 때문에 교란받는 경로  
이로 인해 T to Y 인과관계만으로 설명할 수 없게됨  
이런식으로 그래프를 활용하면, 인과관계를 방해하는 편향을 없애려면 뭘 해야하는지 이해할 수 있음.  
위의 그래프에서는 X를 고정하여 back door path를 없애고 causal path만 남겨야 하는것임.  

### 조건부 독립성 가정과 보정 공식  
여기서 하는 가정: 실험군, 결과 사이 모든 비인과 연관성은 측정가능하고, 조건으로 둘 수 있는 공통원인 때문이다. $$(Y_0, Y_1) \perp T \mid X$$  
이를 **conditional independence assumption (CIA)** 라 하고, 공변량X 수준이 동일한 대상을 비교하면 잠재적 결과는 평균적으로 같음을 말함.  
(독립성 가정은 ignorability(무시가능성), exogeneity(외생성), exchangability(교환가능성) 등 다양한 이름으로 불린다.)

**adjustment formula (보정 공식)(또는 conditionality principle, 조건부 원칙)**  
$$
\begin{aligned}
ATE &=E_x[E[Y_1|T=1] - E[Y_0|T=0]] \\
&= \sum_x{\{(E[Y|T=1,X=x] - E[Y|T=0,X=x])P(X=x)}\} \\
&= \sum_x{\{E[Y|T=1,X=x]P(X=x) - E[Y|T=0,X=x]P(X=x)}\}
\end{aligned}
$$
: X를 조건부로 두거나 통제하면 평균 처치효과는 실험군과 대조군 간 그룹 내 차이의 가중평균으로 식별할 수 있다. 이렇게 교란 요인을 보정해서 back door를 차단하는 과정을 **backdoor adjustment**라 한다.  

### positivity(양수성) 가정  
위 보정 공식에서 한가지 가정이 더 있는데, X의 모든 그룹에 실험군과 대조군의 실험 대상이 반드시 존재해야하는 것임. 이를 양수성 가정이라 함. $0 < P(T|X) < 1$. (common support(공통 지지)나 overlap(중첩)이라고도 함). 이를 위배해도 식별은 가능하나, extrapolation을 해야할수도 있다. 빈 자료 somehow 채워서 한다는 말인듯.   


(cf: frontdoor 보정)
$$
\begin{array}{ccccc}
  &   & U &   & \\
  & \swarrow & & \searrow & \\
  T & \rightarrow & M & \rightarrow & Y
\end{array}
$$
: 처치가 매개자에 미치는 영향, 해당 매개자가 결과에 미치는 영향을 식별. 근데 이런 사례가 거의 없다고 함.  

## confounding bias (교란 편향)  
편향의 첫 번째 주요 원인은 교란.  
교란은 대개 비인과적으로 연관성이 흐르는 열린 뒷문 경로가 있을 때 발생하는데, 이는 일반적으로 처치와 결과가 공통 원인을 공유하기 때문임.  
모든 공통원인을 항상 보정할수는없다. 원인을 알 수 없거나, 측정불가능할 경우 (ex: 관리자 자질. 이런 추상적인걸 test하는 방법이 있을 수 있긴할듯.)  

### surrogate confounder (대리 교란 요인)
이렇게 측정할 수 없는 인자를 대리하는 다른 측정된 변수들을 사용하는 시나리오.  
$$
\begin{array}{cccccc}
 & X_1 & & X_2 & & \\
 & \searrow & \nearrow & & & \\
 & & U & & & \\
 & \swarrow & & \searrow & & \\
 T & \rightarrow & & & Y
\end{array}
$$
X1, X2를 대신 통제하여 U를 통제하는 것임.  

### 랜덤화 재해석
$$
\begin{array}{ccccc}
  &   & U &   & \\
  &  & & \searrow & \\
  랜덤성 & \rightarrow & T & \rightarrow & Y
\end{array}
$$
처치를 무작위로 배정하면 관측할 수 없는 교란 요인이 있는 그래프에서 처치의 유일한 원인이 randomness인 그래프로 바뀐다.  
(원래 U to T 경로가 있었는데, 랜덤성 적용 후 사라지는 것임)  

비랜덤한 상황에서는 
U가 T와 Y 모두에 영향을 주어, $bias = 𝐸[𝑌∣𝑇=1]−𝐸[𝑌∣𝑇=0]−(𝐸[𝑌(1)]−𝐸[𝑌(0)])$와 같은 교란편향(confounding bias) 가 존재한다.  
랜덤화를 통해 𝑇⊥𝑈가 되면, 교란요인에 의한 편향이 사라져서 $ 𝐸[𝑌∣𝑇=1]−𝐸[𝑌∣𝑇=0]=𝐸[𝑌(1)]−𝐸[𝑌(0)]$ 이 된다. 즉 관측된 차이가 곧 인과효과(causal effect) 이다.

예시  
비랜덤 상황:  
공부시간(T)과 성적(Y) 사이의 관계를 보려는데,  
지능(U)이 높을수록 공부도 많이 하고 성적도 잘 받음 → 교란.  
랜덤화:  
학생들을 무작위로 “공부시간 그룹”에 배정.  
이제 공부시간(T)은 더 이상 지능(U)과 상관없음 → 편향 제거. 

→ E[Y∣T=1]−E[Y∣T=0]이 곧 공부시간의 인과효과.  
공부시간이 있는 학생을 조사하는게 아니고, 공부하기전에 학생들 모아서 공부시간 주는거임 (observational study는 selection bias생기므로, randomized experiment를 인위적으로 하는 것임)  

(cf)  
sensitivity analysis(민감도 분석)  
모든 공통원인을 측정할 수 없을 때, 포기하는 대신 "측정되니 않은 교란 요인이 분석 결과를 크게 바꾸려면 얼마나 강력해야 하는가" 로 질문을 바꾸는 것.  
참고: [Carlos Cinelli and Chad Hazlett, 2020](https://academic.oup.com/jrsssb/article/82/1/39/7056023)

partial identification (부분 식별)  
관심 있는 인과 추정량을 정확히 식별할 수 없을 때도 관측 가능한 데이터를 사용하여 그 주변에 '경계'를 설정할 수 있음. 현재 활발히 연구되는 분야라고 함.  

## selection bias (선택 편향)  
교란편향: 처치, 결과의 공통원인을 통제하지 않았을 때.  
선택편향: collider 구조에서, 공통 효과와 매개자에 대한 조건부와 관련됨.  
  - 예시: T를 랜덤으로 뿌려서 진행함. 그래서 모든 샘플들에 설문조사를 실행해서, (일부만 대답할것임) T=1인 그룹이 순고객추천지수(NPS)가 더 높았음. 이 결과를 그대로 활용해도 될까? 아님!!  
  - 공통효과인 설문응답에 조건이 걸리면, 신규기능과 고객만족도에 새로운 경로가 열려버림
  - 즉, 직접적인 인과경로 외에 새로운 경로(bias)가 생김
  $$
\begin{array}{cccccccc}
 & & \text{랜덤성} & & \\
 & & \downarrow & & \\
 & & \text{신규기능} & \rightarrow &\text{고객만족도}\\
 & & \downarrow & \swarrow &  \downarrow \\
 & & \text{설문응답} & & \text{NPS}
\end{array}
$$


즉, 실제로는 독립이던 변수들이 “표본에 들어온다는 조건”을 걸면 (conditioning on selection) 서로 종속(dependent) 이 되어버린다.

결과의 효과에 조건부를 두기만해도 selection bias가 생길 수 있다. 그 효과와 처치가 공유되지 않더라도. 이를 virtual collider라고 함. 선택편향은 복잡한 주제임! 더 공부하려면 [Carlos Cinelli et al., 2020](https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf)

### 선택편향 보정
추가 가정으로 보정을 시도할 수 있다!  
선택편향은 실험으로 해결되지 않아 (정말?) 위험하며, 직관적이지 않고 발견하기 어려울 수 있다고 함...

예: 결과가 선택을 야기하지 않는다고 가정  
$$Y \not\to R$$
또는  
$$R \perp Y∣T$$
collider한쪽 화살표를 없애고, 우회하는 경로를 설정하는 것임.  
즉 선택과 결과 모두를 유발하는 다른 관측 가능한 변수를 넣는다. (도메인 지식 필요)  

선택편향을 보정하려면 선택을 유발하는 모든 요인을 보정해야함.  
결과나 처치가 직접 선택을 유도하거나 선택과 숨겨진 공통 원인을 공유하지 않는다고 가정해야함.  

그러나, 모든 편향을 완전히 제거할 수는 없을 수 있다...  

### 매개자 조건부 설정
$$
\begin{array}{ccccc}
 & & M & & \\
 & \nearrow & & \searrow & \\
T & & \longrightarrow & & Y
\end{array}
$$
M을 통제하면서 T별로 Y를 비교하면, 직접적인 인과 정도를 확인할 수 있다.  
하지만, 매개자(M)의 자식이 조건부로 주어지면 편향이 생기고, 인과 경로를 부분적으로 차단한다.  
$$
\begin{array}{ccccc}
 & & Y & & \\
 & \nearrow & \uparrow & \\
T & \longrightarrow & M & \longrightarrow& X
\end{array}
$$
