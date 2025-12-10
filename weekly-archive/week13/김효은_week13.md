## * 학습 내용 요약

---
### 시간에 따른 효과 변동
- **문제 정의**
    - 처치효과가 즉각적이지 않은 경우 존재함
    - 처치 적용 기간 전/후를 기준으로 ATT 추정하면 과소평가 할 수 있음
- **대응 방안**
    - 시간에 따른 ATT 추정
    - 모든 시간대를 반복하며 마치 해당 시간대만이 처치 이후 기간인 것처럼 DID 적용
        - 처치 전 모든 데이터(post=0) 유지
        - 처치 후 데이터는 지정된 date 딱 하나만 남김
        - date 이전의 데이터만 사용
        - 마지막에 post를 해당 date만 1, 나머지 0으로 재정의
- **결과 해석**
    - 처치 적용 이전 시점에 대해서는 ATT 0으로 추정 → 옳음
    - 처치 적용 초반에는 ATT 0.12 부근에서 변동 → 초반에는 처치효과 크지 않음
    - 처치 적용 5일 차 이후부터는 0.90 부근에서서 변동 → 초반 대비 처치효과 커짐
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/d99a061c-44f1-4d98-bdaf-700f83606da1" />

```python
def did_date(df, date):
    df_date = (
        df
            .query("date == @date | post == 0")
            .query("date <= @date")
            .assign(post = lambda d: (d["date"] == date).astype(int))
    )

    model = smf.ols(formula="downloads ~ treated:post + C(city) + C(date)", data=df_date)
    model = model.fit(cov_type="cluster", cov_kwds={"groups": df_date["city"]})

    att = model.params["treated:post"]
    ci = model.conf_int().loc["treated:post"]

    return pd.DataFrame({"att": att, "ci_low": ci[0], "ci_up": ci[1]}, index=[date])
    
post_date_list = sorted(df_marketing["date"].unique())[1:]
df_att = pd.concat([did_date(df_marketing, date) for date in post_date_list]).reset_index(names="date")
```

---
### 이중차분법과 공변량
- **문제 정의**
    - 전체 데이터에 대해서는 평행 추세 가정을 만족하지 않지만
    - **공변량을 조건부로 두었을 때 평행 추세 가정 충족**하는 경우 존재
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/4797db21-9f3f-4735-9c7c-6bf3c7310615" />
<img width="996" height="360" alt="image" src="https://github.com/user-attachments/assets/aedbb67e-058a-45ad-9299-5102f47a0b5f" />

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 3))
sns.lineplot(x="date", y="downloads", hue="region", style="treated", data=df_marketing_all)
plt.xticks(rotation=90)
plt.axvline(x="2021-05-15", color="black")
```

- **대응 방안**
    - 공변량을 조건부로 하여 ATT 추정
    - (방법 1: 최선책) 공변량을 별도로 DID 회귀모델 적용
    - (방법 2: 최선책) 전체 이중차분법 모델을 공변량 더미변수와 상호작용 (Y ~ T * D * C(공변량))
    - (방법 3: 차선책) Y ~ T * (D + C(공변량)) 형태로 모델 적합
        - 공변량이 많거나 연속형 공변량에 대해서 적용가능한 후순위 방법
        - 여기서 구해진 ATT는 표본크기가 아닌 처치 분산에 따라 가중치 부여하여 구해짐
        - 각 지역 실험군의 추세는 별도로 추정하지만, 실험군과 처치 후 기간에 대해서는 단일 절편 이동 적합
            - → 처치(treatment)에 따른 변화는 모든 지역에 대해 하나의 숫자로 추정한다

```python
# 공변량 고려 전
model = smf.ols(formula = "downloads ~ treated * post", data=df_marketing_all)
model = model.fit()
print(round(model.params["treated:post"], 3)) # 2.068(ATT)
```

```python
# 공변량 고려 후: (1) 공변량 별도로 DID 회귀모델 적용
att = 0.0
region_list = df_marketing_all["region"].unique()

for region in region_list:
    w = len(df_marketing_all.query("region == @region")) / len(df_marketing_all)
    model = smf.ols(formula = "downloads ~ treated * post", data=df_marketing_all.query("region == @region"))
    model = model.fit()
    print(region, round(model.params["treated:post"], 4))
    att += w * model.params["treated:post"]
print("ATT", round(att, 4))
    
# W 3.0462
# N 1.3331
# S 0.6917
# E 1.6768
# ATT 1.6940
```

```python
# 공변량 고려 후: (2) 전체 이중차분법 모델을 공변량 더미변수와 상호작용
model = smf.ols(formula = "downloads ~ treated * post * C(region)", data=df_marketing_all)
model = model.fit()
print(model.summary().tables[1])

# E: 1.6768
# N: 1.3331 = 1.6768 - 0.3437
# S: 0.6917 = 1.6768 - 0.9851
# W: 3.0462 = 1.6768 + 1.3694
"""
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                      17.2758      0.111    155.988      0.000      17.059      17.493
C(region)[T.N]                 26.6759      0.156    171.479      0.000      26.371      26.981
C(region)[T.S]                 33.0592      0.151    219.040      0.000      32.763      33.355
C(region)[T.W]                 10.6681      0.151     70.683      0.000      10.372      10.964
treated                         0.3099      0.204      1.518      0.129      -0.090       0.710
treated:C(region)[T.N]         -1.7759      0.313     -5.679      0.000      -2.389      -1.163
treated:C(region)[T.S]          0.2995      0.318      0.941      0.347      -0.324       0.923
treated:C(region)[T.W]          0.4208      0.318      1.322      0.186      -0.203       1.045
post                            4.9819      0.148     33.737      0.000       4.692       5.271
post:C(region)[T.N]            -3.2730      0.207    -15.780      0.000      -3.680      -2.866
post:C(region)[T.S]            -4.7601      0.201    -23.654      0.000      -5.155      -4.366
post:C(region)[T.W]            -1.7829      0.201     -8.860      0.000      -2.177      -1.388
treated:post                    1.6768      0.272      6.158      0.000       1.143       2.211
treated:post:C(region)[T.N]    -0.3437      0.417     -0.824      0.410      -1.161       0.474
treated:post:C(region)[T.S]    -0.9851      0.424     -2.321      0.020      -1.817      -0.153
treated:post:C(region)[T.W]     1.3694      0.424      3.227      0.001       0.538       2.201
===============================================================================================
"""
```

```python
# 공변량 고려 후: (3) Y ~ T * (D + C(공변량)) 형태로 모델 적합
model = smf.ols(formula = "downloads ~ post * (treated + C(region))", data=df_marketing_all)
model = model.fit()
print(model.summary().tables[1])
"""
=======================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------
Intercept              17.3522      0.101    172.218      0.000      17.155      17.550
C(region)[T.N]         26.2770      0.137    191.739      0.000      26.008      26.546
C(region)[T.S]         33.0815      0.135    245.772      0.000      32.818      33.345
C(region)[T.W]         10.7118      0.135     79.581      0.000      10.448      10.976
post                    4.9807      0.134     37.074      0.000       4.717       5.244
post:C(region)[T.N]    -3.3458      0.183    -18.310      0.000      -3.704      -2.988
post:C(region)[T.S]    -4.9334      0.179    -27.489      0.000      -5.285      -4.582
post:C(region)[T.W]    -1.5408      0.179     -8.585      0.000      -1.893      -1.189
treated                 0.0503      0.117      0.429      0.668      -0.179       0.280
post:treated            1.6811      0.156     10.758      0.000       1.375       1.987
=======================================================================================
"""
```

---
### 이중 강건 이중차분법(DR-DID; Doubly-Robust DID)
- **성향점수 모델**
    - 처치 전 공변량을 활용하여 실험 대상이 실험군에 속할 확률 추정
    - 입력: 공변량 → 예측: 처치받을 확률 모델 적합
- 델타 **결과 모델**
    - 처치 전후 기간의 평균 결과값(Y) 차이 계산
    - 입력: 공변량 → 예측: 평균 결과값(Delta Y) 모델 적합
- 최종 결과
    - (1) 성향점수 모델과 (2) 델타 결과 모델 결합
    - **→ 두 개의 차이를 통해 실험군에 대한 처치효과 추정**

- ATT 추정량 = dy1_treament - dy0_treatment
$$
\hat{\tau}_{DRDID}=\hat{\Delta y_1}^{DR}-\hat{\Delta y_0}^{DR}
$$

- dy1_treament: 실험군에 대한 처치효과 + 시간 상관관계
    
$$
\hat{\Delta y_1}^{DR} = \frac{1}{N_{tr}}\sum_{i \in tr}(\Delta y- \hat{m}(X))
$$
    
- dy0_treatment: 실험군에 대한 시간 상관관계
    
$$
w_{co}=\hat{e}(X)\frac{1}{1-\hat{e}(X)}
$$

$$
\hat{\Delta y_0}^{DR}=\sum_{i \in co}{w_{co}(\Delta y - \hat{m}(X))} / \sum_{i \in co}{w_{co}}
$$
            

```python
# 1. PS 모델
# -- 특정 일자의 데이터 추출 (한 시점의 데이터 필요)
# -- 지역별(X)로 처치 여부(D) 예측 모델 적합
min_date = df_marketing_all["date"].min()

df_unit = (
    df_marketing_all
        .query(f"date == @min_date")
        .drop(columns=["date"])
)

ps_model = smf.logit("treated ~ C(region)", data=df_unit).fit()

# 2. DELTA Y 모델
# -- 목적: 처치가 없었다면 각 지역에서 관측되었을 DELTA Y 예측치
# -- 방법:
# -- ㄴ 도시별로 처치 전 기간 평균 다운로드수와 처치 후 기간 평균 다운로드수 차이 구함
# -- ㄴ 지역별(X)로 DELTA Y 예측 모델 적합

post_0_mean = df_marketing_all.query("post == 0").groupby("city")["downloads"].mean()
post_1_mean = df_marketing_all.query("post == 1").groupby("city")["downloads"].mean()
delta = post_1_mean - post_0_mean

df_delta_y = df_unit.set_index("city").join(delta.rename("delta_y"))

outcome_model = smf.ols("delta_y ~ C(region)", data=df_delta_y).fit()

# 3. 데이터 준비
# -- 도시별로 PS 예측 / DELTA Y 예측
df_dr = (
    df_delta_y
        .assign(ps = lambda d: ps_model.predict(d))
        .assign(y_hat = lambda d: outcome_model.predict(d))
)

# 4. DR 적합
df_control = df_dr.query("treated == 0") # 대조군
df_treatment = df_dr.query("treated == 1") # 실험군

dy1_treatment = (df_treatment["delta_y"] - df_treatment["y_hat"]).mean()

# 실험군의 공변량 분포와 맞추기 위한 작업
w_control = df_control["ps"] / (1 - df_control["ps"])
dy0_treatment = np.average(df_control["delta_y"] - df_control["y_hat"], weights=w_control)

# dy1_treatment: 실험군의 처치효과 + 시간효과 (결과모델 의존)
# dy0_treatment: 실험군의 시간효과 (IPW 의존)
print("ATT", dy1_treatment - dy0_treatment) # 1.6773
```

---
### 처치의 시차 도입 (실험군 내에서도 서로 다른 시간에 처치 발생)
- **용어 정의 및 배경지식**
    - 코호트 정의
        - 처치 받는 시점에 따라 그룹을 구분 (= 코호트)
        - t 시점에 처치 받는 그룹 G를 코호트 G_{t}라고 함
        - (참고) 처치를 받지 않는 코호트는 G_{inf} 라고 하면 됨
    - 이원고정 효과 모델
        
        $$
        Y_{it} = \beta_0 + \beta_1 \cdot \mathbf{D}_{it} + \alpha_i + \gamma_t + \epsilon_{it}
        $$
        
        - Y_{it}: 단위 i의 시점 t에서의 결과
        - D_{it}: 처치 변수(Treatment Dummy) / **단위 i가 시점 t에 처치를 받았다면 1, 아니면 0**
        - beta_{1}: 우리가 추정하고자 하는 평균 처치 효과(ATT) ← **시간에 걸쳐 동일하는 가정**
        - alpha_{i}: 개체 고정효과 (Unit Fixed Effect)
        - gamma_{t}: 시간 고정효과 (Time Fixed Effect)
- **문제 제기**
    - 만약 실험 대상이 동일 시점에 처치를 받는 것이 아니라 **서로 다른 시점에 처치를 받는다면?**
- **문제 발생: 이원고정효과 모델 적용 시 ATT 편향 발생**
    - 처치받은 그룹의 효과를 추정할 때 대조군으로 다음 두 가지를 이용
    - (1) 아직 처치를 받지 않은 그룹: 적절한 대조군
    - (2) 이미 처치를 받은 그룹 (이른 처치 코호트): 문제가 되는 대조군
        - 처치 효과가 지속하거나 증가 혹은 감소하고 있을 수 있는데 (= ,시간에 따라 변동)
        - 이 그룹을 대조군으로 사용하게 되면 처치 효과 편향 발생할 수 있음
    
    ```python
    twfe_model = smf.ols(
    	formula="downloads ~ treated:post + C(date) + C(city)",
    	data=df_mkt_data_cohorts_w
    ).fit()
    
    true_tau = df_mkk_data_cohorts_w.query("post==1 & treated ==1")["tau"].mean()
    
    print("True Effect: ", true_tau) # 2.2625
    print("Estimated ATT: ", twfe_model.params["treated:post"]) # 1.7599
    ```
    

- **문제 해결**: 순수한 대조군 설정
    - (1) 아직 처치를 받지 않은 그룹으로만 대조군 설정
    - ㄴ (1-1) Never-Treated Group: 연구 기간 내내 처치를 받지 않은 그룹
    - ㄴ (1-2) Not-Yet-Treated Group: 아직 처치를 받지 않았지만 미래에 처치를 받을 예정인 그룹
