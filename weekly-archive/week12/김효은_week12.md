## * 학습 내용 요약

---
### * 패널데이터(Panel Data)
- **패널데이터: 여러 개체가 시간에 따라 반복해서 관측되는 데이터 구조**
- ㄴ **동일 실험 대상**을 **여러 시간대에 걸쳐 관측**하면 처치 전/후에 어떤 일이 일어나는지 알 수 있음
- ㄴ 따라서 패널 데이터는 **랜덤화가 불가할 때 인과효과를 식별할 수 있는 좋은 대안**이 될 수 있음
- ㄴ 관측 데이터를 가지고 있고, 관측되지 않은 교란요인이 존재할 가능성이 높다면 처치효과 식별하는 좋은 해결책

| **데이터 구조 유형** | **시간 반복** | **개체 동일성** | **설명** | **목적** |
| --- | --- | --- | --- | --- |
| 횡단면 데이터 (cross-sectional) | **단일 시점** | 여러 개체 (2개 이상) | 여러 개체를 동일 시점에 관측한 결과 | 한 시점의 현황 파악 목적 |
| 종단 데이터 (longitudinal) | **여러 시점** | 시점마다 **동일 개체** (1개 혹은 2개 이상) |  | 동일한 개체의 시간에 따른 변화 파악 |
| 패널 데이터 (panel) | **여러 시점** | 시점마다 **동일 개체** (2개 이상) | 여러 개체를 여러시점에서 관측한 데이터 (종단 데이터의 일종) | 동일한 개체의 시간에 따른 변화 파악 |
| 반복 횡단면 데이터 (repeated cross-sectional) | **여러 시점** | 시점마다 **다른 개체** | 여러 시점에 관측을 진행하나 매번 다른 개체를 관측 | 시점에 따른 변화 확인 목적 |

---
### * 패널데이터 사례
- 관측 대상: 도시
- 기간 변수(T): 일자
- 결과 변수(Y): 앱다운로드 수
- 처치 변수(D): 마케팅 시행 대상 여부
- **알고싶은 인과추정량: ATT** = E[Y(1) - Y(0) | D = 1, T > T_{post}]
    - 패널데이터에서는 ATU를 추정하기 위한 데이터가 제한적이기 때문에 ATU 보다는 ATT 추정에 집중

---
### * 표준 이중차분법(DID; Difference-In-Difference)
### * (1) 이중차분법 (← 평균의 차이의 차이)
$$ DID = (E[Y|D=1,Post=1] - E[Y|D=1, Post=0]) \\ -(E[Y|D=0, Post=1] - E[Y|D=0, Post=0]) $$

|  | **Post = 0 (마케팅 미집행 기간)** | **Post = 1 (마케팅 집행 기간)** |  |
| --- | --- | --- | --- |
| **D = 1 (실험군)** | E[Y\|D=1, Post=0] | E[Y\|D=1, Post=1] | **시간 간 상관관계 + 인과효과** |
| **D = 0 (대조군)** | E[Y\|D=0, Post=0] | E[Y\|D=0, Post=1] | **시간 간 상관관계** |
|  | 실험 대상 간 상관관계 |  |  |

```python
df_did = (
    df_marketing
        .groupby(["treated", "post"])
        .agg({"downloads": "mean"})
        .reset_index()
)

#    treated  post  downloads
# 0        0     0  50.335034
# 1        0     1  50.556878
# 2        1     0  50.944444
# 3        1     1  51.858025

```

### * (2) 이중차분법 결과 변환 (← 차이의 평균의 차이: 선형성)

$$
\Delta{y_{i}}=E[y_{i} \mid t_{i}>T_{pre}]-E[y_{i} \mid t{i} \le {T_{pre}}]
$$

$$
ATT = E[\Delta{y}\mid{D=1}]-E[\Delta{y}\mid{D=0}]
$$


```python
df_pre = df_marketing.query("post == 0").groupby(["treated", "city"]).agg({"downloads": "mean"})
df_post = df_marketing.query("post == 1").groupby(["treated", "city"]).agg({"downloads": "mean"})

df_pre_post = pd.merge(
    left=df_pre,
    right=df_post,
    how="inner",
    on=["treated", "city"],
    suffixes=["_pre", "_post"]
)

df_diff = (
    df_pre_post
        .assign(diff = df_pre_post["downloads_post"] - df_pre_post["downloads_pre"])
        .groupby(["treated"])
        .agg({"diff": "mean"})
        .head(10)
)

print(df_diff)
# treated          
# 0        0.221844 <- E[\Delta{Y}|D=0]
# 1        0.913580 <- E[\Delta{Y}|D=1]
print(df_diff.diff().loc[1, "diff"]) # DID = E[\Delta{Y}|D=1] - E[\Delta{Y}|D=0]
# 0.6917359536407155
```

### * (3) 이중차분법과 OLS

$$
Y_{it}=\beta_{0}+\beta_{1}D_{i}+\beta_{2}Post_{i}+\beta_{3}D_{i}Post_{i}+\epsilon_{it}
$$

| **D** | **Post** | **Y** |
| --- | --- | --- |
| 0 | 0 | B0 |
| 0 | 1 | B0 + B2 |
| 1 | 0 | B0 + B1 |
| 1 | 1 | B0 + B1 + B2 + B3 |
- **회귀계수 해석**
    - B0: 대조군 기준값
    - B1: 실험군과 대조군의 차이
    - B2: 시간에 따른 효과
    - B3: ATT (실험군과 대조군의 차이와 시간 추세를 모두 고려한 DID 추정량)
- **코드 예시**

```python
import statsmodels.formula.api as smf

smf.ols(
    formula="downloads ~ treated * post",
    data = df_marketing
).fit().params

"""
Intercept       50.335034
treated          0.609410
post             0.221844
treated:post     0.691736
"""
```

### * (4) 이원고정효과모델 (TWFE; Two Way Fixed Effects)

$$
	Y_{it}=\tau W_{it}+ \alpha_{i} + \gamma_{t} + e_{it} \\ (W_{it} = D_{i}Post_{t})
$$

- **모델 아이디어**
    - 2개의 효과((1) 개별고정효과 (2) 시간고정효과)를 고정하고 처치의 효과를 추정하는 모델
    - **결과변수 = 개별고정효과 + 시간고정효과 + 처치효과 + 오차**
- **수식 해석**
    - 𝜶: 개별고정효과 (← 각 **개체(i)**가 가진 시간에 따라 변하지 않는 특성)
    - 𝛄: 시간고정효과 (← 모든 개체에 공통적으로 영향을 주는 **시점(t)**별 요인)
    - 𝝉: 처치효과 추정량 (← 처치 효과)
    - W: 처치변수 (=D*P)
    - Y: 결과변수
- **코드 예시**

```python
import statsmodels.formula.api as smf

model = smf.ols(
    formula="downloads ~ treated:post + C(city) + C(date)",
    data=df_marketing
).fit()

# treated:post: 처치효과 추정량
# C(city): 개별고정효과
# C(date): 시간고정효과

print(model.params["treated:post"])
# 0.6917359536407255
```

---
### DID 추정량의 신뢰구간
### * 문제 제기
- N*T개의 데이터 포인트가 있지만 동일한 실험 대상이 여러번 나타나므로 IID 아님
- 실제 표본 크기는 N*T가 아니라 N(관측대상수)에 불과하다고 볼 수도 있음
- 그러나 **회귀분석에서는 표준오차를 계산할 때 표본크기로 N*T를 고려하기 때문에 실제보다 과소평가됨**

```python
model = smf.ols(
    formula="downloads ~ treated:post + C(city) + C(date)",
    data=df_marketing
).fit()

print(model.params["treated:post"]) # 0.691735
print(model.conf_int().loc["treated:post"]) # 0.478014 0.905457 (신뢰구간)
```

### * 대안 1. 군집표준편차
- `cov_type="cluster"`: 클러스터 표준오차 사용
- `cov_kwds={"groups": df_marketing["city"]}`: 어느 단위를 기준으로 클러스터링할지 지정
    - 해당 사례에서는 city 단위로 오차항 상관있을 수 있다고 봄
- (참고) 일반표준오차 vs. 군집표준오차
    - 일반 표준오차: 모든 관측치가 독립이라고 가정 / 관측치 단위로 계산
    - 군집 표준오차: 같은 그룹 내 상관 발생 가능 / 다른 그룹간 독립 가정 / 그룹 단위로 잔차 조정

```python
model = smf.ols(
    formula="downloads ~ treated:post + C(city) + C(date)",
    data=df_marketing
).fit(cov_type="cluster", cov_kwds={"groups": df_marketing["city"]})

print(model.params["treated:post"]) # 0.691735
print(model.conf_int().loc["treated:post"]) # 0.296101 1.087370
```

```python
df_did = (
    df_marketing
        .groupby(["city", "treated", "post"])
        .agg({"downloads": "mean"})
        .reset_index()
)

model = smf.ols(
    formula="downloads ~ treated:post + C(city) + C(post)",
    data=df_did
).fit(cov_type="cluster", cov_kwds={"groups": df_did["city"]})

print(model.params["treated:post"]) # 0.691735
print(model.conf_int().loc["treated:post"]) # 0.138188 1.245284 (기간에 대한 데이터 줄어드니 구간 넓어짐)
```

### * 대안 2. 블록부트스트랩
- **아이디어: 관측 대상 기준으로 복원추출**하여 회귀계수를 여러번 구함
- 신뢰구간 정의: 회귀계수 값을 정렬하여 하위 2.5% ~ 상위 2.5% 구간을 95% 신뢰구간으로 정의
- 주의할 점: 이 때 실험군의 수가 적으면 표본에 실험군이 포함되지 않는 경우 발생할 수 있음

```python
from joblib import Parallel, delayed

def block_sample(df, unit_col):
    units = df[unit_col].unique()
    sample = np.random.choice(units, size=len(units), replace=True) # unit 기준 복원추출

    return (
        df
            .set_index(unit_col)
            .loc[sample]
            .reset_index(level=[unit_col])
    )
 
def block_bootstrap(df, est_fn, unit_col, rounds=200, seed=123, pcts=[2.5, 97.5]):
    np.random.seed(seed)

    stats = Parallel(n_jobs=4)(
        delayed(est_fn)(block_sample(df, unit_col=unit_col))
        for _ in range(rounds)
    )

    return np.percentile(stats, pcts)
    
def est_fn(df):
    model = smf.ols("downloads ~ treated:post + C(city) + C(date)", data=df)
    model = model.fit()
    return model.params["treated:post"]
    
block_bootstrap(df_marketing, est_fn, "city") # [0.23162214, 1.14002646]
```

---
### * 식별 가정
- 인과추론은 **통계적 도구와 가정 사이의 끊임없는 상호작용**
- 이중차분법(DID)을 사용할 때 종종 인식하지 못한 채로 어떤 가정을 했는지 파고들자!

### * (1) 평행 추세 (parallel trend) ⭐️
- 평행 추세 가정: 처치가 없으면 **평균적으로 실험군과 대조군의 결과(=Y(0)) 추세가 동일할 것**
- 노트 1. 평행 추세 가정은 관측할 수 없는 항이 포함되므로 검증할 수 없는 가정임
- 노트 2. Y축을 변환(예: 로그변환)하는 경우 평행 추세 가정 충족 여부가 달라질 수 있음

$$ (\Delta y_{0}, \Delta y_{1}) \perp T $$
$$ E[Y(0)_{it=1}-Y(0)_{it=0}|D=1]=E[Y(0)_{it=1}-Y(0)_{it=0}|D=0] $$


### * (2) 비기대 가정(no assumption)과 SUTVA ⭐️
- SUTVA 가정: (1) 비간섭성 (2) 처치 버전 단일성
- SUTVA 위반 사례
    - (1) 시간적 파급효과: 현재 시점의 결과가 과거 혹은 미래 결과에 영향을 미침 (예) 교육 프로그램의 장기 효과
    - (2) 공간적 파급효과: 한 지역의 처치가 다른 지역의 결과에 영향을 미침 (예) 특정 지역에서의 최저임금 제도 개편
    - (3) 실험 대상 간 파급효과: 어떤 대상에 대한 처치가 다른 대상의 결과에 영향을 미침 (예) 백신 접종, 보조금 지급

### * (3) 강외생성(strict exogeneity)
- **(1) 시간에 따라 변하는 교란요인 없음 (time-fixed confounder)**
    - 패널데이터에서는 시간에 따라 교란요인 일정하다면, 교란요인이 있더라도 인과효과 식별 가능함
    - 실험대상 고정효과는 시간에 걸쳐 일정한 변수(교란변수 포함)를 모두 없애버림
- **(2) 피드백 없음 (no feedback)**
    - 현재의 처치 여부는 과거, 현재, 미래의 잠재적 결과와 독립적임
- (3) 이월 효과 없음 (no carry-over effect)
    - 이월 효과 없음 = 과거 처치가 현재 결과에 영향을 주지 않음 = 현재의 처치만 현재의 결과에 영향을 줌
    - 다만 이월 효과 있다면 → 처치의 시차항(lag term) 포함하여 해결 가능
- (4) 과거 결과가 현재 결과의 직접적인 원인이 되지 않음
    - 다만, 이러한 자기회귀 구조가 존재해도 대조군/실험군 두 집단 모두에 동일하게 존재하므로
    - 평행 추세를 보존하는한 식별에 큰 문제가 되지 않음
