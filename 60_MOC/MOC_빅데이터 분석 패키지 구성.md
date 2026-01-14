---
Created: 2024-05-07 22:01
tags:
  - AI/머신러닝
---

# 개요
데이터 분석 학습 유형에 따른 머신러닝 모델 유형을 사이킷런 구성 단위로 한눈이 파악할 수 있도록 하는 것이 이 노트의 목적이다.

# 내용

## Data Loading

#### CSV 파일 불러오기
- 함수 : read_csv()

## Data Processing

### 판다스 기초

#### 데이터컬럼 추출
- 모듈명 : pandas

##### Object 추출
- 함수 : select_dtypes(include='object')
~~~ python
object_col = df.select_dtypes(include='object').columns
~~~

#### 시계열 데이터
##### Datetime
- 모듈명 : pandas
- 함수 : to_datetime()
~~~ python
# 변환
df['DateTime1'] = pd.to_datetime(df['DateTime1'], format="%y-%m-%d %H:%M:%S")
~~~

~~~
0    24-02-17 11:45:30
1    24-02-18 12:55:45
2    24-02-19 13:30:15
Name: DateTime1, dtype: object
0   2024-02-17 11:45:30
1   2024-02-18 12:55:45
2   2024-02-19 13:30:15
Name: DateTime1, dtype: datetime64[ns]
~~~

~~~ python
# 년, 월, 일, 시간, 분, 초 추출
df['year'] = df['DateTime4'].dt.year
df['month'] = df['DateTime4'].dt.month
df['day'] = df['DateTime4'].dt.day
df['hour'] = df['DateTime4'].dt.hour
df['minute'] = df['DateTime4'].dt.minute
df['second'] = df['DateTime4'].dt.second

# 일자 
df['date'] = df['DateTime4'].dt.date # 2023-07-02

# 요일 dayofweek 0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
df['DateTime4'].dt.dayofweek

# 기간 to_period()
print(df['DateTime4'].dt.to_period('Y')) # 2024
print(df['DateTime4'].dt.to_period('Q')) # 2024Q1
print(df['DateTime4'].dt.to_period('M')) # 2024-02
print(df['DateTime4'].dt.to_period('D')) # 2024-02-17
print(df['DateTime4'].dt.to_period('H')) # 2024-02-17 11:00
~~~
##### Timedelta
- 모듈명 : pandas
- 함수 : Timedelta

~~~ python
# 100일째 되는 날
day = pd.Timedelta(days=99)
df['100일'] = df['DateTime4'] + day

# 100시간 이후
hour = pd.Timedelta(hours=100)
df['100시간'] = df['DateTime4'] + hour

# 시간 +/- (3주, 3일, 3시간, 3분, 3초 이후)
diff = pd.Timedelta(weeks=3, days=3, hours=3, minutes=3, seconds=3)
df['+diff'] = df['DateTime4'] + diff

# 기간을 초로 변환 total_seconds()
print(diff.dt.total_seconds()) #초
print(diff.dt.total_seconds()/60) #분
print(diff.dt.total_seconds()/60/60) #시간
print(diff.dt.total_seconds()/60/60/24) #일

# 일(days), 초(seconds)
print(diff.dt.days)
print(diff.dt.seconds)
~~~

#### 지수형 표기 변환
- 모듈명 : pandas
- 함수 : set_option('display.float_format', '{.10f}'.format)
~~~ python
# 소수점이 e지수 형태로 표현될 경우 
pd.set_option('display.float_format', '{:.10f}'.format)
train['total'].describe()
~~~

~~~~
count       700.0000000000
mean     485078.0175000000
std      364390.7265411940
min       19041.7500000000
25%      200119.5000000000
50%      381874.5000000000
75%      706127.6250000000
max     1563975.0000000002
Name: total, dtype: float64
~~~~

~~~ python
# 아래와 같이 하나의 숫자에 대해서는 쉽게 확인 가능
format(2.462846382e-10,’.10f‘)
~~~ 

#### 병합 (Merge)
- 모듈명 : pandas
- 함수 : merge()

~~~~ python
### basic1 데이터와 basic3 데이터를 'f4'값을 기준으로 병합

import pandas as pd
b1 = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
b3 = pd.read_csv("../input/bigdatacertificationkr/basic3.csv")

df = pd.merge(left = b1 , right = b3, how = "left", on = "f4")
~~~~

#### 분할 (Slicing)

##### qcut
~~~ python
# 동일한 개수로 나이 순으로 3그룹으로 나누기
df['range'] = pd.qcut(df['age'], q=3, labels=['group1','group2','group3']) 
df['range'].head()
~~~

~~~
0    group1
1    group1
2    group1
3    group3
4    group1
~~~

##### cut
~~~ python
grades = ['4등급', '3등급', '2등급', '1등급']
cut_bins = [0, 69, 79, 89, 100]

df['영어등급'] = pd.cut(df['영어점수'], bins = cut_bins, labels = grades)
~~~

####  요약

##### pivot
~~~ python
df2.pivot(index='Location', columns='Period', values='First Tooltip2')
~~~

~~~
Period	2015	2016	2017	2018	2019
Location					
Afghanistan	70.44	67.57	64.94	62.54	60.27
Albania	9.57	9.42	9.42	9.53	9.68
Algeria	25.18	24.79	24.32	23.81	23.26
Andorra	3.53	3.37	3.22	3.09	2.97
Angola	88.20	84.21	80.62	77.67	74.69
...	...	...	...	...	...
Syrian Arab Republic	23.18	23.27	22.97	22.11	21.53
Tajikistan	37.75	36.82	35.81	34.80	33.78
Thailand	10.80	10.32	9.86	9.42	9.01
The former Yugoslav Republic of Macedonia	12.97	11.97	9.94	7.83	6.12
Timor-Leste	50.76	49.01	47.27	45.62	44.22
172 rows × 5 columns
~~~

##### pivot table
~~~ python
df2.pivot_table(index='Dim1',columns='Period',values='First Tooltip2',aggfunc='mean')
~~~

~~~
Period	2015	2016	2017	2018	2019
Dim1					
Both sexes	31.012093	29.956337	29.030465	28.083837	27.191744
~~~

~~~ python
kr.pivot_table(index='Year', columns='Medal', aggfunc='size').fillna(0)
~~~

~~~
Medal	Bronze	Gold	Silver
Year			
1992	1.0	5.0	1.0
1994	1.0	8.0	1.0
1998	2.0	6.0	4.0
2002	0.0	5.0	2.0
2006	2.0	14.0	3.0
2010	2.0	6.0	10.0
2014	2.0	7.0	5.0
~~~
### [[결측값 처리|결측치 처리]]

#### 결측치 확인 및 제거
- 결측치가 포함되어 있는지를 확인 : isnull()
- 각 컬럼에 대한 결측치 추가 확인 : sum(), info()
- 결측치가 있는 행 전체를 제거 : dropna()

~~~ python
df.dropna(subset=['r2'])  # r2 컬럼의 결측치를 제거
df.dropna(how='all') # 모든 값이 NA인 경우만 제거
df.dropna(how='any') # NA가 하나라도 가진다면 제거
~~~
- 결측치를 지정값으로 대체 : fillna()
- 중복값 제거 : drop_duplicates()
~~~ python
df2 = df.drop_duplicates(subset=['age'])
~~~

### [[이상값 처리|이상치 처리]]

#### IQR (사분위범위) 방법
- 함수 : quantile()




## Feature Engineering

### 범주형 변수 처리
#### 라벨 인코딩 (Label Encoding)
- 설명 : 카테고리형 데이터(Categorical Data)를 수치형 데이터(Numerical Data)로 변환해주는 전처리 작업
- 모듈명 : sklearn.preprocessing
- 함수 : LabelEncoder()

#### 원-핫 인코딩 (One-Hot encoding) 
- 설명 : 범주형 변수를 0 또는 1 값을 가진 하나 이상의 새로운 특성으로 바꾼 것이다. 
- 모듈명 : sklearn.preprocessing
- 함수 : OneHotEncoder()

#### get_dummies()
- 설명 : pandas를 통해 아예 데이터프레임 형식으로 반환받는 것이다. 각 column 이름에 변수 특성을 명시해줘서 그냥 array 타입보다는 훨씬 보기 편하다.
- 모듈명 : pandas
- 함수 : get_dummies()
### [[정규화 (Normalization)|정규화]]
##### 최소-최대 정규화(Min-Max Normalization)
- 설명 : 모든 특성이 0과 1사이에 위치하도록 데이터를 변환한다.
- 모듈명 : sklearn.preprocessing
- 함수 : MinMaxScaler()

##### Z-Score 정규화(Z-Score Standardization)
- 설명 : 기존 특성을 평균이 0, 분산이 1인 정규분포로 변환하여 특성의 스케일을 맞춘다.
- 모듈명 : scipy.stats
- 함수 : zscore()
### [[표준화 (Standardization)|표준화]]

#### StandardScaler
- 설명 : 서로 다른 분포를 비교하기 위해 표준에 맞게 통일시키는 방법. 평균을 0, 표준 편차를 1로 변환
- 모듈명 : sklearn.preprocessing
- 함수 : StandardScaler()

#### RobustScaler
- 설명 : 중앙값과 사분의 값을 활용하는 방법으로 이상치 영향을 최소화
- 모듈명 : sklearn.preprocessing
- 함수 : RobustScaler()

## Feature Selection / Extraction

참고자료 : https://dodonam.tistory.com/m/387

##### 주성분 분석(PCA)
- 설명 : 서로 상관성이 높은 여러 변수들의 선형 조합으로 새로운 변수들로 요약, 축소하는 기법이다. 
- 모듈명 : sklearn.decomposition
- 함수 : PCA


## Modeling

### [[지도학습]]

#### 분류분석
##### [[로지스틱 회귀분석|로지스틱 회귀]]
- 설명 : 독립변수의 선형결합을 활용하여 사건의 발생 가능성을 예측(확률)하여 이항 분류를 진행하는 분석 방법
- 모듈명 : sklearn.linear_model
- 함수 : LogisticRegression()
##### [[의사결정 트리 분석|의사결정나무]]
- 설명 : 데이터를 학습하여 데이터 내에 존재하는 규칙을 찾아내고, 이 규칙을 나무구조로 모형화해 분류와 예측을 수행하는 방법이다.
- 모듈명 : sklearn.tree
- 함수 : DecisionTreeClassifier()
##### [[KNN (K-Neareat Neighbor)|KNN]]
- 설명 : 데이터로부터 거리가 가까운 K개의 다른 데이터의 정답(목표값)을 참조하여 분류하는 방법이다.
- 모듈명 : sklearn.neighbors
- 함수 : KNeighborsClassifier(n_neighbors = n)

##### [[앙상블 분석]]
- 설명 : 여러 개의 예측모형을 만든 후 결과를 종합해 하나의 최종 결과를 도출하는 방법이다.
- 모듈명 : sklearn.ensemble
- 함수
	- 보팅 : VotingClassifier()
	- 배깅 : 
	- 부스팅 : GradientBoostingClassifier()
##### [[서포트 벡터 머신(SVM)|SVM]]
- 설명 : 데이터가 어느 카테고리에 속할지 판단하는 이진 선형 분류 모델을 만드는 기법으로 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 그 중 가장 큰 폭을 가지는 경계를 찾는 알고리즘이다.
- 모듈명 : sklearn.svm
- 함수 : SVC()

##### [[랜덤 포레스트]](분류)
- 설명 : 의사결정 트리 분석 기반의 알고리즘으로 다수의 의사결정 트리들을 배깅하여 분류 또는 회귀를 수행하는 앙상블 기법 중 하나이다. 
- 모듈명 : sklearn.ensemble
- 함수 : RandomForestClassifier()

#### 회귀분석

##### [[단순 선형 회귀분석]]
- 설명 : 독립변수와 종속변수 간의 선형적인 관계를 도출하여 회귀(예측)하는 분석 기법이다.
- 모듈명 : sklearn.linear_model
- 함수 : LinearRegression()
##### [[다중 선형 회귀분석]]
- 설명 : 단순 선형 회귀분석이 독립변수를 하나 가지고 있는 선형 회귀분석이라면, 다중 선형 회귀분석은 독립변수가 두 개 이상이고 종속변수가 y 하나인 선형 회귀분석이다.
- 모듈명 : sklearn.linear_model
- 함수 : LinearRegression()
##### 규제모델 - [[Lasso]]
- 설명 : 회귀분석의 규제모델 중 하나로 기존 선형회귀식에 규제항(L1 Regularization)이 적용된 모델
- 모듈명 : sklearn.linear_model
- 함수 : Lasso()

##### 규제모델 - [[Ridge]]
- 설명 : 회귀분석의 규제모델 중 하나로 기존 선형회귀식에 규제항(L2 Regularization)이 적용된 모델
- 모듈명 : sklearn.linear_model
- 함수 : Ridge()

##### [[의사결정 트리 분석|의사결정나무]](회귀)
- 설명 : 분류 기능과는 달리 각 항목에서의 범주를 예측하는 것이 아니라 어떠한 값 자체를 예측하는 것
- 모듈명 : sklearn.tree
- 함수 : DecisionTreeRegressor()
##### [[랜덤 포레스트]](회귀)
- 설명 : 의사결정 트리 분석 기반의 알고리즘으로 다수의 의사결정 트리들을 배깅하여 분류 또는 회귀를 수행하는 앙상블 기법 중 하나이다. 
- 모듈명 : sklearn.ensemble
- 함수 : RandomForestRegressor()
##### [[KNN (K-Neareat Neighbor)|KNN]](회귀)
- 설명 : 데이터로부터 거리가 가까운 K개의 다른 데이터의 정답(목표값)을 참조하여 회귀하는 방법이다.
- 모듈명 : sklearn.neighbors
- 함수 : KNeighborsRegressor()

##### xgboost
- 설명 : 앙상블의 부스팅 기법의 한 종류. 이전 모델의 오류를 순차적으로 보완해나가는 방식으로 모델을 형성하는데, 이전 모델에서의 실제값과 예측값의 오차(loss)를 훈련데이터 투입하고 gradient를 이용하여 오류를 보완하는 방식을 사용합니다.
- 모듈명 : xgboost
- 함수 : XGBRegressor()

### [[비지도학습]]

#### [[군집분석]]

##### K-means
- 설명 : 비지도 학습 기반의 대표적인 비계층적 군집 알고리즘으로 서로 유사한 데이터는 동일 그룹으로, 유사하지 않은 데이터는 다른 그룹으로 분류하는 군집분석
- 모듈명 : sklearn.cluster
- 함수 : KMeans()
#### [[연관분석]]
##### apriori
- 설명 : 하나의 알고리즘으로 둘 이상의 항목들로 구성된 연관성 규칙을 도출하는 분석이다. 지지도를 사용해 빈발 아이템 집합을 판별하고 계산의 복잡도를 감소시키는 알고리즘
- 모듈명 : mlxtend.frequent_patterns 
- 함수 : apriori()
- 규칙도출 : association_rules()

##### 피어슨 상관계수
- 설명 :  절대값이 1에 가까울수록 강한 선형관계를 가짐
- 모듈명 : scipy.stats
- 함수 : pearsonr()

### 가설검정, 범주형 데이터 분석, 회귀 분석, 분산 분석
#### 가설검정

##### 단일표본 t 검정
- 설명 : 모집단 1개
	- 예) 과자의 무게는 200g과 다른지 검정 
- 모듈명 : scipy.stats
- 함수
	- 정규성 검정 : shapiro(data)
		- 단일 표본 검정 (정규성O) : ttest_1samp(data, 기대값)
		- Wilcoxon 부호 순위 검정 (정규성X) : wilcoxon(data - 기대값)

##### 대응표본 t 검정
- 설명 : 모집단 2개 (같은 집단)
	- 예) 신약 효과 (전후) 검정
- 모듈명 : scipy.stats
- 함수
	- 정규성 검정 : shapiro(data1 - data2)
		- 대응(쌍체) 표본 검정 (정규성O) : ttest_rel(data1, data2, alternative='less')
			- alternative = less, greater, two-sided
		- Wilcoxon의 부호 순위 검정 (정규성X) : wilcoxon(data1, data2) 
			- 또는 wilcoxon(data1 - data2)

~~~ python
# 예시문제 : 혈압 치료 전/후에 따른 해당치료가 효과가 있는지?
df['diff'] = df['bp_after'] - df['bp_before']
print(df['diff'].mean()) # 표본평균 

from scipy.stats import ttest_rel
st, pv = ttest_rel(df['bp_after'], df['bp_befor'], alternative='less')
print(st) # 검정통계량
print(pv) # p-value

# p-value가 0.0005로 유의수준 0.05보다 높게 나왔을때
print('귀무가설을 기각하고, 대립가설을 채택한다.')
~~~

##### 독립표본 t 검정
- 설명 : 모집단 2개
	- 예) 1반과 2반의 성적 차이 검정
- 모듈명 : scipy.stats
- 함수
	- 정규성 검정 : shapiro(data1), shapiro(data2)
		- 등분산 검정 : levene(data1, data2)
			- 독립 표본 검정 (정규성O, 등분산O) : ttest_ind(data1, data2, equal_var=True)
			- 독립 표본 검정 (정규성O, 등분산X) : ttest_ind(data1, data2, equal_var=False)
		- Mann-Whitney U 검정 (정규성X, 등분산X) : manwhitneyu(data1, data2)


#### [[분산 분석(ANOVA)]]
- 설명 : 어려 집단의 평균 차이를 통계적으로 유의미한지 검정

##### 일원 분산 분석 (One-way ANOVA)
- 설명 : 하나의 요인에 따라 평균의 차이 검정
- 모듈명 : scipy.stats
- 함수
	- 정규성 검정 : shapiro(data1)
		- 등분산 검정(정규성O) : levene(data1, data2, ...)
			- 일원 분산 분석(정규성O, 등분산O) : f_oneway(data1, data2, ...)
				- 사후검정
					- Tukey HSD : pairwise_tukeyhsd()
					- Bonferroni : MultiComparison()
		- 크루스칼(정규성X) : kruskal(data1, data2, ...)
~~~ python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('value ~ variable', data=df_melt).fit()
anova_lm(model)
~~~

~~~
		    df	    sum_sq	 mean_sq	F	        PR(>F)
variable	3.0	    2.35875	 0.78625	7.296984	0.000605
Residual	36.0	3.87900	 0.10775	NaN	        NaN

 - df: 자유도
 - sum_sq: 잔차제곱합 (그룹 평균 간의 차이를 나타내는 제곱합)
 - mean_sq: 평균 제곱 (sum_sq/자유도)
 - F: 검정통계량
 - PR(>F): p-value
~~~
~~~ python
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
     A      B     0.41 0.0397  0.0146  0.8054   True
     A      C     0.09 0.9273 -0.3054  0.4854  False
     A      D    -0.27 0.2722 -0.6654  0.1254  False
     B      C    -0.32 0.1483 -0.7154  0.0754  False
     B      D    -0.68 0.0003 -1.0754 -0.2846   True
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
     B      D  6.0686    0.0    0.0001   True
     C      D  2.5219 0.0213    0.1279  False
---------------------------------------------
~~~
##### 이원 분산 분석 (Two-way ANOVA)
- 설명 : 두 개의 요인에 따라 평균의 차이 검정
- 모듈명
	- statsmodels.formula.api
	- statsmodels.stats.anova
- 함수
	- ols
	- anova_lm

>[!note] 범주형 변수 처리 : C()
>작업형3에서 사용하는 ols(회귀), logit(로지스틱회귀)는 범주형 변수가 있다면 자동으로 이를 (원핫인코딩)처리 해줍니다. 하지만 여전히 숫자인 범주형 변수 그러니깐 소형, 중형으로 작성된 것이 아니라 1, 2로 작성된 변수만 별도 범주형 변수로 인식시켜줘야 해요. 
>
>이때 C()를 사용합니다.
>
>분산분석에서는 독립변수가 범주형 변수이니 숫자만 C()로 묶어도 되고, 습관적으로 독립변수 모두를 각각 C()로 묶어도 됩니다.

#### 범주형 분석 (카이제곱 검정)

##### 적합도 검정
- 설명 : 관찰도수와 기대도수의 차이
- 모듈명 : scipy.stats
- 함수 : chisquare(observed, expected)
	- observed : 관찰된 빈도 리스트
	- expected : 기대 빈도 리스트

~~~
[문제] 지난 3년간 빅데이터 분석기사 점수 분포가 60점 미만: 50%, 60-70점 35%, 80점이상 15%로였다. 이번 회차부터 단답형을 제외하고, 작업형3을 추가하여 300명을 대상으로 적용한 결과 60점 미만: 150명, 60-70점: 120명, 80점이상: 30명이었다. 유의수준 0.05일 때, 새로운 시험문제 유형과 기존 시험문제 유형은 점수에 차이가 없는지 검정하시오.
귀무가설(H0): 새로운 시험문제는 기존 시험문제 점수와 동일하다.
대립가설(H1): 새로운 시험문제는 기존 시험문제 점수와 다르다.
~~~

~~~ python
#관찰
ob = [150, 120, 30]
#기대
ex = [0.5*300, 0.35*300, 0.15*300]

from scipy import stats
stats.chisquare(ob, ex)
~~~

~~~
Power_divergenceResult(statistic=7.142857142857142, pvalue=0.028115659748972056)
~~~

##### 독립성 검정
- 두 변수가 서로 독립적인지(연관성이 있는지) 확인
- 모듈명 : scipy.stats
- 함수 : chi2_contigency(table, correction=True)
	- table : 교차표
	- correction : 연속성 보정 (기본값 True)

~~~
[문제] 빅데이터 분석기사 실기 언어 선택에 따라 합격 여부를 조사한 결과이다. 언어와 합격 여부는 독립적인가? 가설검정을 실시하시오. (유의수준 0.05)
귀무가설(H0): 언어와 합격 여부는 독립이다.
대립가설(H1): 언어와 합격 여부는 독립이지 않다.
~~~

~~~ python
# 교차표 데이터일 경우
import pandas as pd
df = pd.DataFrame({
    '합격':[80, 90],
    '불합격':[20, 10]
    },index=['R', 'P']
)

from scipy import stats
stats.chi2_contingency(df)


# 로우 데이터일 경우
import pandas as pd
data = {
    '언어': ['R']*100 + ['Python']*100,
    '합격여부': ['합격']*80 + ['불합격']*20 + ['합격']*90 + ['불합격']*10
}
df = pd.DataFrame(data)
df = pd.crosstab(df['언어'], df['합격여부'])
stats.chi2_contingency(df)
~~~
~~~
Chi2ContingencyResult(statistic=3.1764705882352944, pvalue=0.07470593331213068, dof=1, expected_freq=array([[85., 15.],
       [85., 15.]]))
~~~
#### 회귀 분석

##### 상관관계 (피어슨)
- 설명 : 두 변수 간의 선형관계의 강도와 방향 (-1 <= r <= 1)
- 함수
	- 피어슨 : corr()
	- 스피어맨 : corr(method='spearman')
	- 켄달타우 : corr(method='kendall')
- 함수 (t검정)
	- 피어슨 : scipy.stats.corr(x, y)
	- 스피어맨 : scipy.stats.spearmanr(x, y)
	- 켄달타우 : scipy.stats.kendaltau(x, y)
##### [[단순 선형 회귀분석]]
- 설명 : OLS: 최소제곱법(Ordinary Least Squares)
- 모듈명 : statsmodels.formula.api
- 함수 : ols

~~~ python
from statsmodels.formula.api import ols

# 모델 학습
model = ols('키 ~ 몸무게', data=df).fit()
print(model.summary())

# 결정계수
model.rsquared

# 기울기 (회귀계수)
model.params['몸무게']

# 절편 (회귀계수)
model.params['Intercept']

# pvalue
model.pvalues['몸무게']

# 잔차 제곱합
df['잔차'] = df['키'] - model.predict(df['몸무게'])
sum(df['잔차'] ** 2)

# MSE
from sklearn.metrics import mean_squared_error
pred = model.predict(df)
mean_squared_error(df['키'], pred)

# 몸무게가 50일 때 예측키에 대한 신뢰구간(mean), 예측구간(obs)
newdata = pd.DataFrame({'몸무게':[50]})
pred = model.get_prediction(newdata)
pred.summary_frame(alpha=0.05) # 신뢰구간 0.05
~~~

##### [[다중 선형 회귀분석|다중선형회귀]]1
- 모듈명 : statsmodels.formula.api
- 함수 : ols 

~~~ python
# 모델 학습 summary 출력 (종속변수: 매출액, 독립변수: 광고비, 플랫폼)
from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 플랫폼', data=df).fit()
print(model.summary())

# 예측
new_data = pd.DataFrame({
	'광고비':[30]
	'플랫폼':[1]					 
})
pred = model.get_prediction(new_data)
print(pred.summary_frame(alpha=0.05)  #신뢰구간 95% 
# mean : 종속변수 예측값
# mean_ci_lower ~ mean_ci_upper : 신뢰구간
# obs_ci_lower ~ obs_ci_upper : 예측구간
~~~

~~~
       mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  \
0  21.56163  0.175263      21.213737      21.909524     18.082985   

   obs_ci_upper  
0     25.040276  
~~~

##### [[다중 선형 회귀분석|다중선형회귀]]2 (범주형 변수)
- 모듈명 : statsmodels.formula.api
- 함수 : ols

##### [[로지스틱 회귀분석|로지스틱 회귀]]
- 모듈명 : statsmodel.formula.api
- 함수 : logit
- 참고자료 : [[오즈 (Odds)]]

~~~ python
# 모델 학습 및 summary 출력 (종속변수: Survived, 독립변수: Gender(object), SibSp, Parch, Fare)
from statsmodels.formula.api import logit
model = logit('Survived ~ C(Gender) + SibSp + Parch + Fare', data=df).fit()
print(model.summary())
~~~
~~~
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               Survived   No. Observations:                  891
Model:                          Logit   Df Residuals:                      886
Method:                           MLE   Df Model:                            4
Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.2761
Time:                        08:20:24   Log-Likelihood:                -429.52
converged:                       True   LL-Null:                       -593.33
Covariance Type:            nonrobust   LLR p-value:                 1.192e-69
=====================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept             0.9466      0.169      5.590      0.000       0.615       1.279
C(Gender)[T.male]    -2.6422      0.186    -14.197      0.000      -3.007      -2.277
SibSp                -0.3539      0.098     -3.604      0.000      -0.546      -0.161
Parch                -0.2007      0.112     -1.792      0.073      -0.420       0.019
Fare                  0.0147      0.003      5.553      0.000       0.010       0.020
=====================================================================================
~~~

~~~ python
# 로지스틱 회귀분석의 Sibsp의 계수(회귀계수)
result1 = round(model.params['SibSp'],3)
print(result1)
# 값 : -0.354

# SibSp 변수가 한단위 증가할때의 오즈비 값
import numpy as np
odds_ratio = np.exp(model.params['SibSp'])  # 오즈비
result2 = round(odds_ratio,3)
# 값 : 0.702

# 학습한 모델의 우도값
result3 = round(model.llf, 3)
# 값 : -429.52
~~~

> [!note] 우도 : 관측치가 가장 많이 발견될 것으로 보이는 경위의 확률값을 '우도'라고 하며, 우도가 낮을 수록 이상값에 가깝다고 할 수 있다.


###### GLM
- 설명 : 일반화 선형 모델(Generalized Linear Models, GLM)의 한 종류. GLM은 선형 회귀 모델을 확장한 것으로, 종속변수의 분포가 정규분포 외에 다른 분포(예: 이항분포, 포아송분포 등)를 따르는 경우에도 적용할 수 있음
- 모듈명 : statsmodels.formula.api
- 함수명 : glm()
- 참고 : 잔차 이탈도 (Residual Deviance)를 구하는 문제에서 적용

~~~ python
from statsmodels.formula.api import glm
import statsmodels.api as sm

# 1) glm 모델 적합 (로지스틱 회귀를 위해 이항 분포 사용)
formula = "gender ~ age + length + diameter + height + weight"
model = glm(formula, data=train, family=sm.families.Binomial()).fit()

# 2) 잔차이탈도 계산
print(model.summary())
print(round(model.deviance,2))
~~~


## Data Splitting / Trainning
##### 데이터 분리
- 설명 : 교차검증을 위한 학습용/테스트용 데이터 분리
- 모듈명 : sklearn.model_selction
- 함수 : train_test_split()
##### 데이터 학습
- 설명 : 데이터 학습
- 함수 : fit()
## [[분석모델 성능 평가|Evaluation]]

#### [[분류분석 평가지표]]

##### [[혼동행렬(Confusion Matrix)]]
- 설명 : 이진 분류의 예측 오류가 얼마인지 더불어 어떠한 유형의 예측 오류가 발생하고 있는지 함께 보여준다.
- 모듈명 : sklearn.metrics
- 함수 : confusion_matrix()
##### 정확도(Accuracy)
- 설명 : 실제 데이터와 예측 데이터가 얼마나 정확한지를 판단하는 자료이다.
- 모듈명 : sklearn.metrics
- 함수 : accuracy_score()
###### 오류율
- 설명 : 오류율 = (잘못 분류된 샘플 수) / (전체 샘플 수)
- 수식 : 오류율 = 1 - accuracy_score
##### 정밀도(Precision)
- 설명 : Positive로 에측한 것중에서 실제 값이 Positive인 비율을 말한다.
- 모듈명 : sklearn.metrics
- 함수 : precision_score()
##### 재현율(Recall),민감도(Sensitivity)
- 설명 : 실제 Positive인 값 중 Positive로 분류한 비율을 말한다. (실제 예측와 예측값이 일치)
- 모듈명 : sklearn.metrics
- 함수 : recall_score()
##### f1-score 
- 설명 : 정밀도와 재현율의 조화평균으로, 정밀도와 재현율 중 한쪽만 클 때보다 두 값이 골고루 클때 큰 값이 된다.
  (predict로 예측했을때는 f1-score를, predict_proba로 예측했을때는 ROC-AUC를 사용한다.) 
	- F1 Score는 Precision과 Recall의 조화평균으로 주로 분류 클래스 간의 데이터가 불균형이 심각할때 사용한다.
	- 높을수록 좋은 모델이다.
- 모듈명 : sklearn.metrics
- 함수 : f1_score()

##### ROC-AUC
- 설명 : ROC 곡선 아래 면적을 AUC(Area Under Curve)라고 하며, AUC가 0.5일 때 분류 능력이 없다고 평가할 수 있고, 면적이 넓을수록, 즉 1에 가까울수록 분류를 잘하는 모형이라 할 수 있다. 
  (predict_proba로 예측했을때는 ROC-AUC를 사용한다.)
- 모듈명 : sklearn.metric
- 함수 : roc_auc_score

~~~ python
# 다중분류 데이터
y_true = pd.DataFrame([2, 2, 3, 3, 2, 1, 3, 3, 2, 1]) # 실제값
y_pred = pd.DataFrame([2, 2, 1, 3, 2, 1, 1, 2, 2, 1]) # 예측값

from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='macro')  # 다중 분류시 컬럼 추가 필요 average= micro, macro, weighted
~~~

#### [[회귀분석 평가지표]]

##### 평균절대오차(MAE) (Mean Absolute Error)
- 설명 : 모형의 예측값과 실제값의 차이를 평균한 값으로 정의하며, 음수오차와 양수오차가 서로 상쇄되는 것을 막기 위해 절댓값을 사용한다. 오차의 크기를 그대로 반영하여 오차 평균 크기를 확인
- 모듈명 : sklearn.metrics
- 함수 : mean_absolute_error()

##### 평균제곱오차(MSE) (Mean Squared Error)
- 설명 : 모형의 예측값과 실제값 차이를 제곱하여 평균한 값으로 정의한다.
- 모듈명 : sklearn.metrics
- 함수 : mean_squared_error(squared=True)

##### 평균제곱근오차(RMSE) (Root Mean Squared Error)
- 설명 : 평균제곱오차(MSE)에 루트를 씌운 값이다. 회귀모형의 평가지표로 실무에서도 자주 사용된다. MSE 크기를 줄이기 위한 목적으로 사용
- 모듈명 : sklearn.metrics
- 함수 : mean_squared_error(squared=False)

##### 결정계수 R2 (R-Squared)
- 설명 : 주어진 데이터에 회귀선이 얼마나 잘 맞는지, 적합 정도를 평가하는 척도이자 독립변수들이 종속변수를 얼마나 잘 설명하는지 보여주는 지표다.
- 모듈명 : sklearn.metrics
- 함수 : r2_score()

##### RMSLE (Root Mean Squared Log Error)
- 설명 : 예측값이 실제값보다 작을때 더 큰 패널티를 부여
- 모듈명 : sklearn.metrics
- 함수 : mean_squared_log_error(squared=False)
~~~ python
def rmsle(y_test, y_pred): #RMSLE
    return np.sqrt(np.mean(np.power(np.log1p(y_test) - np.log1p(y_pred), 2)))
~~~

##### MAPE (Mean Absolute Percentage Error)
- 설명 : MAE를 퍼센트로 표시
- 모듈명 : sklearn.metrics
- 함수 : mean_absolute_percentage_error()
~~~ python
def mape(y_test, y_pred): #MAPE
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
~~~

## Inferencing

### 예측 수행
- 함수 : predict()
## Submission

### 제출
- 함수 : to_csv()

# 출처


# 관련 노트
##### 빅데이터 분석기사 실기 노트
- [[메타코드M_빅분기실기_1유형_데이터다루기 [1강].pdf]]
- [[메타코드M_빅분기실기_1유형_문제풀이 [2강].pdf]]
- [[메타코드M_빅분기실기_2유형_문제풀이_분류분석(1) [3강].pdf]]
- [[메타코드M_빅분기실기_2유형_분류분석(2) [4강].pdf]]
- [[메타코드M_빅분기실기_2유형_분류분석(3)_[4강].pdf]]
- [[메타코드M_빅분기실기_2유형_회귀분석(1) [4강].pdf]]
- [[메타코드M_빅분기실기_2유형_회귀분석(2) [4강].pdf]]
- [[메타코드M_빅분기실기_3유형_1_모평균 검정(모집단 1개)_[5강].pdf]]
- [[메타코드M_빅분기실기_3유형_2_모평균 검정(모집단 2개)_[5강].pdf]]
- [[메타코드M_빅분기실기_3유형_3_모평균 검정(모집단 3개 이상)_[5강].pdf]]
- [[메타코드M_빅분기실기_3유형_4_카이제곱검정_[5강].pdf]]
- [[메타코드M_빅분기실기_3유형_5_다중회귀분석 및 상관분석_[5강].pdf]]
- [[메타코드M_빅분기실기_3유형_6_로지스틱 회귀분석_[5강].pdf]]

##### 빅데이터 분석기사 실기 마인드맵 (2,3유형)
- [[작업형2_마인드맵_-_퇴근후딴짓_v1.1.pdf]]
- [[작업형3_마인드맵_-_퇴근후딴짓_v1.0.pdf]]

# 외부 링크
![[머신러닝 학습 유형 (지도학습, 비지도학습).png]]
##### 주요 모듈
![[빅데이터 분석_파이썬_사이킷런_주요 모듈.png]]
##### 프레임워크
![[빅데이터 분석_파이썬_사이킷런_프레임워크.png]]


![[머신러닝 프로세스.png]]
