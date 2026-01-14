---
Created: 2024-05-22 21:43
tags:
  - AI/머신러닝
---

# 개요
여러 집단의 평균 차이를 통계적으로 유의미한지 검정

# 내용

## 분산분석

분산분석(ANOVA)은 여러 집단의 평균 차이를 통계적으로 유의미한지 검정

- 일원 분산 분석 (One-way ANOVA): 하나의 요인에 따라 평균의 차이 검정
- 이원 분산 분석 (Two-way ANOVA): 두 개의 요인에 따라 평균의 차이 검정

## 일원 분산 분석

- 3개 이상의 집단 간의 평균의 차이가 통계적으로 유의한지 검정
- ==하나인 요인이고, 집단의 수가 3개 이상일 때 사용==

### 기본가정

- 독립성: 각 집단의 관측치는 독립적이다.
- 정규성: 각 집단은 정규분포를 따른다. (샤피로 검정)
- 등분산성: 모든 집단은 동일한 분산을 가진다. (레빈 검정)

### 귀무가설과 대립가설

- 귀무가설: 모든 집단의 평균은 같다.
- 대립가설: 적어도 한 집단은 평균이 다르다.

### 일원 분산 분석

```python
# 사이파이
f_oneway(sample1, sample2, ...)
```

F_onewayResult(statistic=7.2969837587007, pvalue=0.0006053225519892207)

```python
# 스테츠모델즈 (아노바 테이블)
model = ols('종속변수 ~ 요인', data = df).fit()
print(anova_lm(model))
```

![[Pasted image 20240522214448.png]]

- df: 자유도
- sum_sq: 제곱합 (그룹 평균 간의 차이를 나타내는 제곱합)
- mean_sq: 평균 제곱 (sum_sq/자유도)
- F: 검정통계량
- PR(>F): p-value

### 프로세스

![[Pasted image 20240522214614.png]]

투키(tukey)

```
 group1 group2 meandiff p-adj lower upper reject 
 A      B     0.41 0.0397  0.0146  0.8054   True
 A      C     0.09 0.9273 -0.3054  0.4854  False
 A      D    -0.27 0.2722 -0.6654  0.1254  False
 B      C    -0.32 0.1483 -0.7154  0.0754  False
 B      D    -0.68 0.0003 -1.0754 -0.2846   True
 C      D    -0.36 0.0852 -0.7554  0.0354  False

```

본페로니(bonferroni)

```
 group1 group2 stat pval pval_corr reject
 A      B -2.7199  0.014    0.0843  False
 A      C  -0.515 0.6128       1.0  False
 A      D  1.7538 0.0965    0.5788  False
 B      C  2.2975 0.0338    0.2028  False
 B      D  6.0686    0.0    0.0001   True
 C      D  2.5219 0.0213    0.1279  False

```

### 실습

#### 기초 
~~~ python
import pandas as pd
df = pd.DataFrame({
    'A': [3.5, 4.3, 3.8, 3.6, 4.1, 3.2, 3.9, 4.4, 3.5, 3.3],
    'B': [3.9, 4.4, 4.1, 4.2, 4.5, 3.8, 4.2, 3.9, 4.4, 4.3],
    'C': [3.2, 3.7, 3.6, 3.9, 4.3, 4.1, 3.8, 3.5, 4.4, 4.0],
    'D': [3.8, 3.4, 3.1, 3.5, 3.6, 3.9, 3.2, 3.7, 3.3, 3.4]
})
print(df.head(2))
~~~

~~~
     A    B    C    D
0  3.5  3.9  3.2  3.8
1  4.3  4.4  3.7  3.4
~~~

~~~ python
# 일원 분산 분석
from scipy import stats
stats.f_oneway(df['A'], df['B'], df['C'], df['D'])
~~~

~~~
F_onewayResult(statistic=7.2969837587007, pvalue=0.0006053225519892207)
~~~

~~~ python
# 정규성, 등분산, 일원 분산 분석

# Shapiro-Wilk(샤피로-윌크) 정규성 검정
print(stats.shapiro(df['A']))
print(stats.shapiro(df['B']))
print(stats.shapiro(df['C']))
print(stats.shapiro(df['D']))

# Levene(레빈) 등분산 검정
print(stats.levene(df['A'], df['B'], df['C'], df['D']))

# 일원 분산 분석
print(stats.f_oneway(df['A'], df['B'], df['C'], df['D']))
~~~

~~~
ShapiroResult(statistic=0.949882447719574, pvalue=0.667110025882721)
ShapiroResult(statistic=0.934644877910614, pvalue=0.49509894847869873)
ShapiroResult(statistic=0.9871343374252319, pvalue=0.9919547438621521)
ShapiroResult(statistic=0.9752339720726013, pvalue=0.9346861243247986)
LeveneResult(statistic=1.5433829973707245, pvalue=0.22000894224209636)
F_onewayResult(statistic=7.2969837587007, pvalue=0.0006053225519892207)
~~~

#### 크루스칼-왈리스 검정 (비모수 검정)
- 한 그룹이라도 정규분포에 만족하지 못했을때
~~~ python
# 크루스칼-왈리스 검정 (비모수 검정)
import pandas as pd
from scipy import stats

# 데이터
df = pd.DataFrame({
    'A': [10.5, 11.3, 10.8, 10.6, 11.1, 10.2, 10.9, 11.4, 10.5, 10.3],
    'B': [10.9, 11.4, 11.1, 11.2, 11.5, 10.8, 11.2, 10.9, 11.4, 11.3],
    'C': [10.2, 10.7, 10.6, 10.9, 11.3, 11.1, 10.8, 10.5, 11.4, 11.0],
    'D': [13.8, 10.4, 10.1, 10.5, 10.6, 10.9, 10.2, 10.7, 10.3, 10.4]
})

# 정규성 검정
print(stats.shapiro(df['A']))
print(stats.shapiro(df['B']))
print(stats.shapiro(df['C']))
print(stats.shapiro(df['D']))

# Kruskal-Wallis 검정
stats.kruskal(df['A'], df['B'], df['C'], df['D'])
~~~

~~~
ShapiroResult(statistic=0.949882447719574, pvalue=0.667110025882721)
ShapiroResult(statistic=0.934644877910614, pvalue=0.49509894847869873)
ShapiroResult(statistic=0.9871343374252319, pvalue=0.9919547438621521)
ShapiroResult(statistic=0.5759974718093872, pvalue=2.8656615540967323e-05)
KruskalResult(statistic=11.183607021517561, pvalue=0.010773365310213669)
~~~

#### 심화
~~~ python
# 데이터 재구조화 (긴 형태)
df_melt = df.melt()
df_melt.head()
~~~

#### 분산분석 테이블
~~~ python
# ANOVA 테이블
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('value ~ variable', data=df_melt).fit()
anova_lm(model)

# model = ols('value ~ variable', data=df_melt).fit()
# anova_lm(model)
~~~
#### 사후검정
- 목적: 어떤 그룹들 간에 통계적으로 유의미한 차이가 있는지 구체적으로 파악하는 것
~~~ python
# 사후검정
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# Tukey HSD (투키)
tukey_result = pairwise_tukeyhsd(df_melt['value'], df_melt['variable'], alpha=0.05)
print(tukey_result.summary())

# Bonferroni(본페로니)
mc = MultiComparison(df_melt['value'], df_melt['variable'])
bon_result = mc.allpairtest(stats.ttest_ind, method='bonf')
print(bon_result[0])
~~~

~~~
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
     A      B     0.41 0.0397  0.0146  0.8054   True <--- 유의미한 차이가 있다.
     A      C     0.09 0.9273 -0.3054  0.4854  False
     A      D    -0.27 0.2722 -0.6654  0.1254  False
     B      C    -0.32 0.1483 -0.7154  0.0754  False
     B      D    -0.68 0.0003 -1.0754 -0.2846   True <--- 유의미한 차이가 있다.
     C      D    -0.36 0.0852 -0.7554  0.0354  False
----------------------------------------------------
Test Multiple Comparison ttest_ind 
FWER=0.05 method=bonf
alphacSidak=0.01, alphacBonf=0.008
=============================================
group1 group2   stat   pval  pval_corr reject
---------------------------------------------
     A      B -2.7199  0.014    0.0843  False
     A      C  -0.515 0.6128       1.0  False
     A      D  1.7538 0.0965    0.5788  False
     B      C  2.2975 0.0338    0.2028  False
     B      D  6.0686    0.0    0.0001   True <--- 유의미한 차이가 있다.
     C      D  2.5219 0.0213    0.1279  False
---------------------------------------------
~~~
----
## 이원 분산 분석

- 3개 이상의 집단 간의 평균의 차이가 통계적으로 유의한지 검정
- 요인의 수가 2개, 집단의 수가 3개 이상일 때 사용

### 기본가정

- 독립성: 각 집단의 관측치는 독립적이다.
- 정규성: 각 집단은 정규분포를 따른다. (샤피로 검정)
- 등분산성: 모든 집단은 동일한 분산을 가진다. (레빈 검정)

### 귀무가설과 대립가설

주 효과와 상호작용 효과

- 주 효과(요인1)
    - 귀무가설: 모든 그룹의 첫 번째 요인의 평균은 동일하다.
    - 대립가설: 적어도 두 그룹은 첫 번째 요인의 평균은 다르다.
- 주 효과(요인2)
    - 귀무가설: 모든 그룹의 두 번째 요인의 평균은 동일하다.
    - 대립가설: 적어도 두 그룹은 두 번째 요인의 평균은 다르다.
- 상호작용효과
    - 귀무가설: 두 요인의 그룹 간에 상호작용은 없다.
    - 대립가설: 두 요인의 그룹 간에 상호작용이 있다.

### 이원 분산 분석

```python
# 스테츠모델즈 (아노바 테이블)
model = ols('종속변수 ~ C(요인1) * C(요인2)', data=df).fit()
print(anova_lm(model))
```
![[Pasted image 20240522214553.png]]

![[Pasted image 20240522214601.png]]
# 출처


# 관련 노트


# 외부 링크

