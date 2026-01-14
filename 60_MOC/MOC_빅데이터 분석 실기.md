---
Created: 2023-10-22 14:53
tags:
  - AI/머신러닝
share_link: https://share.note.sx/g57zzda1#i92n9vi6KGnpTU8NGy30y5aHNfxmd1DtIjresJt5vjg
share_updated: 2024-08-24T23:15:25+09:00
---
# 개요
빅데이터 분석기사 기준으로 실기위주의 이론을 담았다. Python 기반의 실제 분석 구문과 절차에 대한 설명이 담긴 노트이다. 

***
# 내용
## 제1유형 : 파이썬, 판다스 활용
제1유형 작업은 데이터 탐색, 데이터 변환, 이상치, 결측치 처리 등 데이터 전처리에 필요한 개념을 이해하고 이를 코드로 구현한다.
### 데이터 탐색

#### 개요
- 요약정보, 기초통계, 시각화 등을 통해 자료를 관찰하고 이해하는 과정이다.
- 데이터 분포 및 값을 검토함으로써 패턴을 발견하고 잠재적 문제에 대해 인식하여 해결안을 도출할 수 있다.
#### 탐색적 데이터 분석(EDA : Exploratory Data Analysis)

##### 분석과정 및 절차
1) 개별 데이터 관찰
	- 데이터 값을 눈으로 확인하면서 특이사항 및 추세 등을 확인
2) 데이터의 문제성 확인
	- 결측치와 이상치 유무를 확인
	- 결측치 대치 방법 : 단순대치법, 다중대치법 등
	- 이상치 대치법 : 제거, 대체, 유지 등
3) 데이터의 개별 속성값 분포 확인
	- 통계지표를 확인하여 데이터를 이해
	- 데이터의 중심 : 평균(Mean), 중앙값(Median), 최빈값(Mode)
	- 데이터의 분산 : 범위(Range), 분산(Variance), 표준편차(Standard Deviation)
	- 사분위 범위(IQR) 방법 등을 사용한다.
4) 데이터의 속성간 관계 파악
	- 상관관계 분석을 통해 데이터 속성 간의 관계를 파악한다.
##### 데이터 탐색

###### 데이터 호출
~~~ python
# 주요 패키지 호출
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt          # 맷플롯립 패키지 임포트

# 깃허브에 있는 csv 파일을 읽어와서 데이터프레임 df로 넣는다.
df = pd.read_csv("http://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
~~~
###### 변수타입 변환
~~~python
# 변수 확인
df.info()         # 데이터프레임 구조

# 변수 타입 변환
df["Survived"] = df["Survived"].astype(str)
df["Pclass"] = df["Pclass"].astype(str)
~~~

###### 데이터 탐색
- head() 함수 : 지정한 수의 레코드를 보여준다.
- describe() 함수 : 데이터셋의 수치형 변수에 대해서 기초 통계량을 보여준다.
	- include='all' 옵션을 사용하면 모든 변수에 대한 통계량을 보여준다.
	- unique : 변수에 저장된 범주(값)의 수, top : 가장 많이 출현하는 값, freq : 가장 많이 출현하는 값의 개수
~~~ python
# head 함수를 활용한 탐색
df.head()

# describe 함수를 활용한 탐색
df.describe(include='all')
~~~

###### 기초 데이터 분석
- groupby() 함수를 활용한 빈도수 확인
~~~ python
# Pclass(등급) 변수 분석 - 빈도수 구하기
grouped = df.groupby("Pclass")
grouped.size()

## Sex 변수 분석 - 남녀간의 생존율 차이가 있는지?

# 성별 생존자 수
data_0 = df[df["Sex"]=="female"]["Survived"]          # 여자 생존 데이터
grouped = pd.DataFrame(data_0).groupby("Survived")
print(grouped.size())

data_1 = df[df["Sex"]=="male"]["Survived"]            # 남자 생존 데이터
grouped = pd.DataFrame(data_1).groupby("Survived")
print(grouped.size())
~~~

~~~
Survived
0     81
1    233
dtype: int64

Survived
0    468
1    109
dtype: int64
~~~

### 데이터 전처리
#### 개요
###### 필요성 
- 수집한 데이터를 탐색해보면 빠지거나 틀린 값, 단위가 다를 수 있다.
- 목적에 맞게 데이터를 재가공하여 분석하기 좋게 데이터 클리닝 작업도 진행한다.
###### 유형
- 결측치 처리
	- 빠진 값을 처리하는 것을 말한다.
	- 처리 방법으로는 항목을 모두 버리거나, 적절한 값으로 대체, 또는 NaN으로 표시하여 분석단계로 결측치를 넘기는 방법도 있다.
- 틀린(invalid) 값 처리
	- 잘못된 값들을 처리하는 것을 말한다.
	- 처리 방법으로는 항목을 모두 버리거나, 적절한 값으로 대체, 분석단계로 틀린 값을 넘기는 방법이 있다
- 이상치 처리
	- 범위가 일반적인 범위를 벗어나 특별한 값을 갖는 것을 말한다.
- 데이터 변환
	- 데이터를 분석하기 좋은 형태로 바꾸는 작업을 말하며, 분포를 고려하여 정규화하거나 단위를 조정하거나 로그 스케일로 변환하는 것을 모두 데이터 변환이라고 한다.
	- 변환 종류는 아래와 같다.
		- 범주형으로 변환 
		  : 수치 데이터를 범주형으로 변환하는 경우가 많다. (ex) 10대, 20대...)
		- 일반 정규화 
		  : 여러 데이터들을 비교 분석할때 동일한 스케일로 변경하여 사용한다. (ex) 두 시험과목의 만점 기준에 따른 점수 비교)
		- Z-score 정규화 
		  : 표준 편차를 고려해서 데이터를 변환 (ex) 평균을 0점으로, 표준 편차를 1점으로 환산)
		- 로그 변환
		  : 로그를 취한 값으로 사용하는 것을 말함. 로그를 취하면 분포가 정규분포에 가까워지는 경우가 많다.
		- 역수 변환
		  : 역수를 사용하면 오히려 선형적인 특성을 가지게 되어 의미를 해석하기 쉬워지는 경우에 역수로 변환하여 사용
- 데이터 축소
	- 데이터 크기를 줄이는 것을 말함
	- 예를 들어 허리치수와 몸무게 패턴이 일정하다면 둘 중 하나의 값만 사용한다.
- 샘플링
	- 전체 데이터 중 필요한 데이터만 취하는 것을 샘플링이라고 함
- 훈련데이터와 테스트 데이터
	- 데이터 분석은 모델을 만드는 과정과 모델을 검증하는 과정 크게 두 단계 절차가 필요하다.
	- 훈련(Trainning)과 테스트(Test)를 위하여 데이터를 랜덤으로 준비하는 것이 필요하다.

#### 데이터 

[[표준화 (Standardization)]]
- 정의
	- 서로 다른 분포를 비교하기 위해 표준에 맞게 통일시키는 방법
	- 평균을 0, 표준 편차를 1로 변환
- 분석방법
	- 표준 정규화 (표준화)
		- 표준 정규분포를 갖는 데이터를 생성하고, 이를 데이터 프레임으로 변환한다.
			~~~ python
			import numpy as np
			import pandas as pd
			import matplotlib.pyplot as plt  
			
			# 한국인, 일본인 각 성인 1000명 육류소비량 데이터 생성
			meat_consumption_korean = 5*np.random.randn(1000) + 53.9
			meat_consumption_japan = 4*np.random.randn(1000) + 32.7
			
			# 데이터 프레임 생성
			meat_consumption = pd.DataFrame({"한국인":meat_consumption_korean,"일본인":meat_consumption_japan})
			
			# 상위 6개 데이터 확인
			display(meat_consumption.head(6))
			~~~
	
		- 시각화를 통해서 히스토그램에 의한 파악을 수행해본다.
			~~~ python
			# 한국인의 육류소비량 히스토그램
			plt.hist(meat_consumption_korean)
			plt.xlabel('Korea')
			plt.show()
			
			# 일본인의 육류소비량 히스토그램
			plt.hist(meat_consumption_japan)
			plt.xlabel('Japan')
			plt.show()
			~~~ 
		- 사이킷런 StandardScaler 를 활용한 표준화
			~~~ python
			# 사이킷런 스케일러 이용 정규화
			from sklearn.preprocessing import StandardScaler
			
			scaler = StandardScaler()
			meat_consumption["한국인_정규화3"] = scaler.fit_transform(meat_consumption[["한국인"]])
			
			scaler = StandardScaler()
			meat_consumption["일본인_정규화3"] = scaler.fit_transform(meat_consumption[["일본인"]])
			
			# 정규화한 데이터 조회
			meat_consumption.head()
			~~~


###### [[정규화 (Normalization)]]
- 정의
	- 데이터의 범위가 같아지도록 변수별로 값을 비례적으로 조정하는 과정으로, 대표적인 기법으로 Min-Max 정규화가 있다.
- 분석방법

	- Min-Max 정규화
		- MinMaxScaler() 함수를 사용해서 표준화한다
			~~~ python
			# MinMaxScaler() 함수 이용
			from sklearn.preprocessing import MinMaxScaler
			
			scaler = MinMaxScaler()
			meat_consumption["한국인_mm"] = scaler.fit_transform(meat_consumption[["한국인"]])
			scaler = MinMaxScaler()
			meat_consumption["일본인_mm"] = scaler.fit_transform(meat_consumption[["일본인"]])
			
			meat_consumption[["한국인", "일본인", "한국인_mm", "일본인_mm"]].head()
			~~~
		
		- 직접 식을 입력하여 표준화한다.
		
			~~~ python
			# 수식 이용
			Min = np.min(meat_consumption_korean)
			Max = np.max(meat_consumption_korean)
			meat_consumption["한국인_mm2"] = (meat_consumption[["한국인"]] - Min) / (Max - Min)
			
			Min = np.min(meat_consumption_japan)
			Max = np.max(meat_consumption_japan)
			meat_consumption["일본인_mm2"] = (meat_consumption[["일본인"]] - Min) / (Max - Min)
			
			meat_consumption[["한국인", "일본인", "한국인_mm", "일본인_mm", "한국인_mm2", "일본인_mm2"]].head()
			~~~
	
	- Z-score 정규화
		- zscore() 함수를 사용한 표준화
			~~~ python
			# zscore() 함수를 사용한 Z-표준화
			import scipy.stats as ss
			
			meat_consumption["한국인_정규화"] = ss.zscore(meat_consumption_korean)        
			meat_consumption["일본인_정규화"] = ss.zscore(meat_consumption_japan)
			
			# 표준화된 데이터에 대한 히스토그램
			plt.hist(meat_consumption["한국인_정규화"])
			plt.xlabel('Korea')
			plt.show()
			
			plt.hist(meat_consumption["일본인_정규화"])
			plt.xlabel('Japan')
			plt.show()
			~~~

###### [[정규분포변환]]
- 정의
	- 왜도가 0이 아닌 경우 한쪽으로 치우쳐 있으며, 이 분포에서는 중심 경향 측정값(평균, 중앙값, 최빈값)이 동일하지 않다.
	- 데이터분석 필요시 필요에 따라 기존의 데이터의 분포형태를 변경해야하는 경우가 있다.
- 분석방법
	- 최초 데이터 왜도 계산
	  : 두 변수는 양과 음의 왜도를 가진다는 것을 확인하였다.
	~~~ Python
		df = pd.read_csv("http://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/USJudgeRatings.csv")
		df.head(5)
		
		import scipy.stats as ss
		
		# 지정한 컬럼의 왜도 계산
		print(ss.skew(df["CONT"]))
		print(ss.skew(df["PHYS"]))
	~~~

	~~~
		1.0859724796276253 
		-1.5582154642293153
	~~~ 
	- log() 변환
	  : 로그 변환을 통하여 해당 분포를 중심 경향 형태의 값들로 변환한다.
	~~~ Python
		# 변수 변환 - log() 변환
		df["CONT1"] = np.log(df["CONT"]) # 왜도가 양수일 경우
		df["PHYS1"] = np.log(np.max(df["PHYS"]+1) - df["PHYS"]) # 왜도가 음수일 경우
		
		# 왜도 계산
		print(ss.skew(df["CONT1"]))
		print(ss.skew(df["PHYS1"]))
	~~~

	 ~~~
		0.6555571886692603
		0.5824357748750443
	~~~ 
###### 범주화
- 정의
	- 연속형 변수를 범주형 변수로 변환하는 작업
- 분석방법
	- 조건을 사용해서 구간을 직접 지정
		~~~ python
		# 수학점수 (Math Score)
		data = [["철수",52], ["영희",92], ["미영",84], ["시완",71], ["미경",65], 
				["영환",81], ["숙경",66], ["부영",77], ["민섭",73], ["보연",74]]
		df = pd.DataFrame(data,columns = ['이름', '수학점수'])
		
		print(np.mean(df["수학점수"]))
		
		# 히스토그램, 범위 50~100, 5개 구간
		plt.hist(df["수학점수"], bins=5, range=[50,100], rwidth=0.9)
		# rwidth로 그래프 폭을 조절할 수 있음(생략 시 기본값 1)
		plt.show()
		df
		
		# 조건을 사용해서 구간을 직접 지정
		df["등급"] = 0
		
		df.loc[(df["수학점수"]<60), "등급"] = "F"
		df.loc[(df["수학점수"]>=60) & (df["수학점수"]<70), "등급"] = "D"
		df.loc[(df["수학점수"]>=70) & (df["수학점수"]<80), "등급"] = "C"
		df.loc[(df["수학점수"]>=80) & (df["수학점수"]<90), "등급"] = "B"
		df.loc[(df["수학점수"]>=90) & (df["수학점수"]<=100), "등급"] = "A"
		~~~
	- cut() 함수 사용
		~~~ python
		# Cut 함수 사용
		df["등급"] = pd.cut(x=df["수학점수"],
						  bins=[0,60,70,80,90,100],
						  labels=["F","D","C","B","A"],
						  include_lowest=True)
		~~~
	- qcut() 함수 사용
		- cut() 함수는 bins를 직접 수치로 지정하는 반면, qcut() 함수는 나누고자 하는 범주 개수를 정해주면 자동으로 데이터가 채워지도록 범주를 나눔
		~~~ python
		# qcut() 함수 사용
		df["등급_qcut"] = pd.qcut(x=df["수학점수"], q=5, labels=["F","D","C","B","A"])
		~~~

#### 차원 축소 : PCA(Principal Component Anaysis, 주성분 분석)
###### 정의
- 주성분분석이란?
	- 여러 변수들의 변량을 '주성분'이라고 불리는, 서로 상관성이 높은 여러 변수들의 선형 조합으로 새로운 변수들로 요약, 축소하는 기법이다. ![[빅데이터 주성분 분석(PCA)을 통한 차원축소.png]]
###### PCA (차원 축소)
- 변수간의 스케일 차이가 나면 스케일이 큰 변수가 주성분에 영향을 많이 주기 때문에 주성분 분석 전에 표준화나 정규화를 시켜준다.
- 사이킷런의 PCA를 이용해서 쉽게 주성분 분석을 수행할 수 있다.
	- n_component는 PCA로 변환할 차원의 수를 의미
	- 첫번째 PCA 변환요소(차원)만으로 전체 변동성의 73%를 설명이 가능하다. 두번째 요소(차원)는 22.8%를 차지한다.
	- 따라서, 2개 요소(차원)로만 변환해도 원본 데이터의 변동성을 95.8% 설명이 가능하므로, 변수를 4개에서 2개로 줄일 수 있다.
~~~ python
# 데이터 준비하기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         # 맷플롯립 패키지 임포트

iris = pd.read_csv("http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
iris

# 연속형 변수와 범주형 변수 분리
df = iris.drop(["species"], axis=1)
df_species = iris["species"]

df.head()

# 사이킷런의 PCA를 이용해서 쉽게 주성분 분석 수행
# 변수 정규화
from sklearn.preprocessing import StandardScaler

df["sepal_length"] = StandardScaler().fit_transform(df[["sepal_length"]])
df["sepal_width"] = StandardScaler().fit_transform(df[["sepal_width"]])
df["petal_length"] = StandardScaler().fit_transform(df[["petal_length"]])
df["petal_width"] = StandardScaler().fit_transform(df[["petal_width"]])

# PCA 분석
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
p_score = pca.fit_transform(df)
print(p_score.shape)
print(pca.explained_variance_ratio_)
~~~

~~~
(150, 4)
[0.72962445 0.22850762 0.03668922 0.00517871]
~~~

#### [[결측값 처리|결측치 처리]]
###### 결측치 확인 및 제거
- 결측치가 포함되어 있는지를 확인 : isnull()
- 각 컬럼에 대한 결측치 추가 확인 : sum(), info()
- 결측치가 있는 행 전체를 제거 : dropna()
~~~ python
df = pd.read_csv("http://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 결측치 확인
print(df.isnull().sum())    # Column별 결측치 확인 갯수 확인
df.info()                   # 데이터프레임 구조 확인 - Column별 결측치 갯수가 확인 가능하다

# 결측치 제거
df_1 = df.dropna(axis=0)
print(df_1.isnull().sum().sum())        # 데이터프레임 결측치 전체
df_1.shape                              # 데이터프레임 크기
~~~

###### 결측치 대체
- 결측치를 지정값으로 대체 : fillna()
~~~ python
# 결측치 대체 - Age 컬럼의 결측치를 평균으로 대체
print(df["Age"].isnull().sum())   
age_mean = df["Age"].mean()
df["Age"].fillna(age_mean, inplace=True)      # Age 컬럼의 결측치를 평균으로 대체
print(df["Age"].isnull().sum())               # Age 컬럼의 결측치 개수

# 결측치 대체 - Embarked 컬럼의 결측치를 최빈값으로 대체
from scipy.stats import mode

print(df["Embarked"].isnull().sum())
embarked_mode = df["Embarked"].mode()                   # Embarked 컬럼의 최빈값
df["Embarked"].fillna(embarked_mode[0], inplace=True)   # Embarked 컬럼의 결측치를 최빈값으로 대체
print(df["Embarked"].isnull().sum())                    # 대체 후 Embarked 컬럼의 결측치 개수

# 결측치 대체 - Age 컬럼의 결측치를 그룹별 평균값으로 대체
df["Age"].fillna(df.groupby("Pclass")["Age"].transform("mean"), inplace=True)
~~~

#### [[이상값 처리|이상치 처리]]
###### IQR (사분위범위) 방법
~~~ python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt           # 맷플롯립 패키지 임포트

# 정규분포 평균 50, 표준편차 10을 가지는 데이터 200개 생성 -> 데이터프레임으로 변환
data = 10 * np.random.randn(200) + 50
df = pd.DataFrame({"값":data})

# 임의의 이상치 삽입
df.loc[201] = 2
df.loc[202] = 100
df.loc[203] = 10
df.loc[204] = 110

# 분포 시각화
plt.hist(df["값"], bins=20, rwidth=0.8)
plt.show()

# IQR(Intgerquartile Range, 사분위범위) 방법
# 박스플롯
plt.boxplot(df["값"])
plt.show()

Q1 = df["값"].quantile(.25)       # 1사분위수
Q2 = df["값"].quantile(.5)        # 2사분위수
Q3 = df["값"].quantile(.75)       # 3사분위수
IQR = Q3 - Q1                     # 사분위범위

lowerOutlier = df.loc[(df["값"] < (Q1 - IQR * 1.5))]
upperOutlier = df.loc[(df["값"] > (Q3 + IQR * 1.5))]

print(lowerOutlier)
print(upperOutlier)
~~~

#### 평활화
###### 정의 
- 데이터 분포를 매끄럽게 함으로써 중요하지 않는 데이터를 제거, 패턴을 알아내는 방법
###### 단순이동평균
- rolling(n).mean(), n=데이터 개수
~~~ python
df = pd.read_csv("http://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/lynx.csv")      # 1821~1934년 캐나다 시라소니 수에 대한 시계열 데이터

# 단순이동평균(Simple Moving Average) 
# 10년 단순이동평균
df["sma"] = df["value"].rolling(10).mean()

plt.plot(df["value"])
plt.plot(df["sma"])
plt.show()
~~~
![[데이터전처리_평활화_단순이동평균 결과.png]]

###### 지수가중이동평균
- ewm(n).mean(), n=데이터 개수
~~~ python
# 지수가중이동평균(Exponentially-weighted Moving Average)
# : 최근 데이터에 가중치를 부여하여 이동평균을 구하는데, 가중치를 지수함수형태로 사용한다.
# 10년 지수가중이동평균
df["ewm"] = df["value"].ewm(10).mean()

plt.plot(df["value"])
plt.plot(df["ewm"])
plt.show()
~~~
![[데이터전처리_평활화_지수가중이동평균 결과.png]]



## 제2유형 : 머신러닝 (전처리, 모델링 및 평가)
제2유형 작업은 분석 데이터셋을 탐색하고, 데이터를 학습하여 분석모델을 만들어 성능을 평가한다. 학습유형에는 지도학습, 비지도학습 등으로 구분된다.

### [[빅데이터 분석 과정 (Python)]]
#### 분석 과정 이해
- 빅데이터 분석 과정
	1. 필요 패키지 임포트
	2. 데이터 불러오기
	3. 데이터 살펴보기
	4. 데이터 전처리
	5. 분석 데이터셋 준비
	6. 데이터 분석 수행 
	7. 성능평가 및 시각화

##### 필요 패키지 임포트
- 넘파이(numpy) : 선형대수와 통계기능을 제공하는 패키지
- 판다스(pandas) : 파이썬의 대표적인 데이터 처리 패키지로 행렬기반의 2차원 데이터 처리에 특화
- 사이킷런(scikit-learn) : 분석을 위한 머신러닝 알고리즘과 편리한 API를 제공하는 패키지
- 맷플롯립(matplotlib) : 차트, 그래프 등을 지원하는 시각화 패키지

~~~python
## 1. 필요 패키지 임포트(Import)
import numpy as np
import pandas as pd
import sklearn

# 의사결정나무 분류모델을 위한 패키지 임포트
from sklearn.tree import DecisionTreeClassifier

# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~

##### 데이터 불러오기
~~~ python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
~~~

##### 데이터 살펴보기
- 데이터 전체 또는 일부를 출력해서 데이터를 살펴본다.
- 판다스에서 제공하는 함수를 사용하여 전체 데이터의 개수, 컬럼별 평균, 최대값, 최소값 등 데이터 분포를 파악해본다.
~~~ python
## 3. 데이터 살펴보기
df # 데이터 프레임 전체를 출력
df.shape # 데이터프레임의 열과 행의 수를 출력
df.info() # 데이터프레임의 요약정보 출력
df.describe() # 데이터프레임의 기술통계 보여주기
~~~

##### 데이터 전처리
- 이상치, 결측치를 판단하여 처리한다.
- 데이터 인코딩, 단위환산, 자료형 변환, 정규화, 파생변수 생성 등의 작업을 진행하여 최적의 분석데이터로 만들어준다.
~~~ python
## 4. 데이터 전처리
# 텍스트로 되어 있는 Species 컬럼의 데이터를 0,1,2로 변환한다.
print(df["species"].unique())  # 데이터 유니크 값 확인
df["species"].replace({"setosa":0, "versicolor":1, "virginica":2}, inplace=True) # 라벨링
~~~

##### 분석 데이터 준비하기
- 전처리를 마친 데이터는 학습용/테스트용 데이터셋으로 분리한다. (일반적으로 8:2)
~~~ python
## 5. 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=11)
print(X_train.shape) # 학습 데이터(독립변수)
print(X_test.shape) # 테스트 데이터(독립변수)
print(y_train.shape) # 학습 데이터(종속변수)
print(y_test.shape) # 테스트 데이터(종속변수)
~~~

##### 데이터 분석 수행
- 분석 알고리즘을 선택하고, 분석을 수행한다.
- [[데이터 분석 학습 유형]]은 크게 지도학습, 비지도학습으로 구분할 수 있으며, 지도학습은 분류와 예측 문제로 구분해서 모델을 선택한다.
	- 지도학습 - 분류 : [[의사결정 트리 분석|의사결정나무]](분류), [[KNN (K-Neareat Neighbor)|KNN]], [[서포트 벡터 머신(SVM)]], [[로지스틱 회귀분석]], [[랜덤 포레스트]](분류), 인공신경망
	- 지도학습- 회귀(예측) : [[단순 선형 회귀분석|선형회귀분석]], [[다중 선형 회귀분석|다중회귀분석]], [[의사결정 트리 분석|의사결정나무]](회귀), [[랜덤 포레스트]](회귀)
	- 비지도학습 : [[군집분석]], [[연관분석]], 인공신경망
- 해당 데이터의 경우 붓꽃 종류를 구분하는 문제이므로 분류문제이며, 분류를 위한 의사결정나무를 적용
- 학습이 완료된 모델에서 테스트 데이터셋으로 분류(예측)를 수행

~~~ python
## 6. 데이터 분석 수행
dt = DecisionTreeClassifier(random_state=11) # 의사결정나무(DecisionTreeClassifier) 객체 생성
dt.fit(X_train, y_train) # 학습수행
pred = dt.predict(X_test) # 학습이 완료된 dt 객체에서 테스트 데이터셋을 예측 수행
~~~

##### 성능평가 및 시각화
- 각 모델에 적합한 성능평가 방법을 사용하여 평가를 수행한다.
- 의사결정나무를 이용한 분류모델의 성능평가 대표적인 지표는 '정확도' 이다.
- 모델의 정확도는 0.9333 으로, 즉 93% 정확도로 붓꽃의 품종을 분류(예측)함을 알 수 있다.
~~~ python
## 7. 성능평가 및 시각화
# 모델 성능 - 정확도 측정
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)
~~~
~~~
0.933333333333
~~~



#### 사이킷런 패키지 구성
##### 주요 모듈
![[빅데이터 분석_파이썬_사이킷런_주요 모듈.png]]
##### 프레임워크
![[빅데이터 분석_파이썬_사이킷런_프레임워크.png]]

#### [[분석모델 성능 평가]]

##### [[분석모델 성능 평가|분류분석 평가지표]]
###### [[혼동행렬(Confusion Matrix)|혼동행렬]](Confusion Matrix)
- 이진 분류에서 성능 지표로 잘 활용되는 오차행렬
- 이진 분류의 예측 오류가 얼마인지 더불어 어떠한 유형의 예측 오류가 발생하고 있는지 함께 보여준다
- 오차행렬을 통해 정확도, 정밀도, 재현도, F1 Score 등을 구할 수 있다.
~~~ python
# 모델 성능 - 오차 행렬
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
~~~

~~~
array([[ 9, 0, 0], 
	   [ 0, 10, 0], 
	   [ 0, 2, 9]])
~~~
- 결과값의 가운데 대각선 숫자는 정확하게 분류한 숫자를 보여준다.
- 위 예시의 30개 데이터 중 28개는 정확하게 분류한 것을 알 수 있다.
###### 모델 평가지표
- 오차행렬에 기반한 평가지표는 classification_report() 함수를 통해 구할 수 있다.
~~~ python
# 모델 성능 평가 - 평가지표 계산
from sklearn.metrics import classification_report
rpt = classification_report(y_test, pred)
print(rpt)
~~~

~~~
	          precision recall f1-score support 
           0       1.00   1.00     1.00       9 
           1       0.83   1.00     0.91      10 
           2       1.00   0.82     0.90      11 

    accuracy                       0.93      30 
   macro avg       0.94   0.94     0.94      30 
weighted avg       0.94   0.93     0.93      30
~~~


- 정확도(Accuracy)
	- 실제 데이터와 예측 데이터가 얼마나 정확한지를 판단하는 자료이다.
	- 전체 데이터에서 올바르게 분류한 데이터의 비율을 말한다.
	- 오차행렬 기반으로 계산하며, 사이킷런의 accuracy_score() 함수를 통해 구할 수 있다.
- 정밀도(Precision)
	- Positive로 에측한 것중에서 실제 값이 Positive인 비율을 말한다.
	- 오차행렬을 기반으로 계산하며, 사이킷런의 precision_score() 함수를 통해 구할 수 있다.
- 재현율(Recall),민감도(Sensitivity)
	- 실제 Positive인 값 중 Positive로 분류한 비율을 말한다. (실제 예측와 예측값이 일치)
	- 오차행렬을 기반으로 계산하며, 사이킷런의 recall_score() 함수를 통해 구할 수 있다.
- f1-score 
	- 정밀도와 재현율의 조화평균으로, 정밀도와 재현율 중 한쪽만 클 때보다 두 값이 골고루 클때 큰 값이 된다.
	- 사이킷런의 f1_score() 함수를 통해 구할 수 있다. 
- ![[데이터분석 평가지표_혼동행렬_평가지표.png]]

- RoC(Receiver Operation Characteristic Curve) Curve
	- 거짓긍정비율(FPR)과 참긍정비율(TPR) 간의 관계
	- ROC 곡선은 임곗값을 다양하게 조절해 분류 모형의 성능을 비교할 수 있는 그래프로, trade-off 관계인 민감도와 특이도를 기반으로 시각화 한 것이다.
	   ![[데이터분석 평가지표__ROC 곡선_AUC개념.png]]

- AUC(Area Under the Curve)
	- ROC 곡선 아래 면적을 AUC(Area Under Curve)라고 하며,
	- AUC가 0.5일 때 분류 능력이 없다고 평가할 수 있고, 면적이 넓을수록, 즉 1에 가까울수록 분류를 잘하는 모형이라 할 수 있다.

##### [[회귀분석 평가지표]]

###### 개요
- 예측값과 실제값의 차이를 기반으로 한 지표들을 이용해 회귀 모형의 성능을 평가할 수 있다.
- 예를 들어 미래의 주식 가격 예측, TV 판매량 예측, 비디오 게임 매출액 예측 등이 있다.

###### 내용
- 평균절대오차(MAE) (Mean Absolute Error)
	- sklearn.metrics.mean_absolute_error
	- 모형의 예측값과 실제값의 차이를 평균한 값으로 정의하며, 음수오차와 양수오차가 서로 상쇄되는 것을 막기 위해 절댓값을 사용한다.
	- 오차의 크기를 그대로 반영하여 오차 평균 크기를 확인
	- 작을수록 좋지만 너무 작으면 과적합 문제가 있음
~~~ python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(df['실제값'], df['예측값'])
~~~~

~~~
3.9294117647058826
~~~


- 평균제곱오차(MSE) (Mean Squared Error)
	- sklearn.metrics.mean_squared_error(squared=True)
	- 모형의 예측값과 실제값 차이를 제곱하여 평균한 값으로 정의한다.
	- 큰 오차를 더 크게, 작은 오차는 더 작게 평가하여 이상치에 민감
~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'])
~~~~

~~~
41.762745098039225
~~~


- 평균제곱근오차(RMSE) (Root Mean Squared Error)
	- sklearn.metrics.mean_squared_error(squared=False)
	- 평균제곱근오차(MSE)에 루트를 씌운 값이다. 회귀모형의 평가지표로 실무에서도 자주 사용된다.
	- MSE 크기를 줄이기 위한 목적으로 사용
~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'], squared=False)
~~~~

~~~
6.462410161699675
~~~


- 평균절대백분율오차(MAPE)
	- 실제값 대비 오차를 평균한 값으로 평균절대오차(MAE)와 같이 절댓값을 모두 더하기 때문에 실제보다 낮은 값으로 예측되는지 높은 값으로 예측되는지 알 수 없다.

- 결정계수 R2 (R-Squared)
	-  sklearn.metrics.r2_score
	- 주어진 데이터에 회귀선이 얼마나 잘 맞는지, 적합 정도를 평가하는 척도이자 독립변수들이 종속변수를 얼마나 잘 설명하는지 보여주는 지표다.
~~~ python
from sklearn.metrics import r2_score
r2_score(df['실제값'], df['예측값'])
~~~~

~~~
0.5145225055729962
~~~


### [[지도학습]] - 분류
- 정의
	- 정답이 있는 데이터가 주어진 상태에서 학습하는 방법
	- 분류(Classification)이란 데이터들간의 분류 카테고리를 학습, 파악하고 새로운 데이터에 대한 분류 카테고리를 판별하는 과정
- 분류에 사용되는 대표적인 알고리즘
	- [[로지스틱 회귀분석]], 신경망모형, [[의사결정 트리 분석|의사결정나무]](분류트리모형), [[KNN (K-Neareat Neighbor)|KNN]], [[앙상블 분석]], [[서포트 벡터 머신(SVM)]], [[랜덤 포레스트]](분류), 나이브베이즈 등
#### [[의사결정 트리 분석|의사결정나무]] - 분류
##### 알고리즘
- 의사결정을 위한 규칙을 나무 모양으로 조합하여 분류를 수행하는 기법
> 예시) 시장조사, 광고조사, 품질관리 등 다양한 분야에서 활용되고 있음

##### 분석 수행
- 타이타닉 데이터셋에서 탑승자들의 여러 속성 데이터를 기반으로 생존여부(Survived)를 예측한다.
###### 필요 패키지 임포트
~~~ python
## 의사결정나무
import numpy as np
import pandas as pd
import sklearn

# 의사결정나무 분류모델을 위한 패키지 임포트
from sklearn.tree import DecisionTreeClassifier

# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split         
~~~
###### 데이터 불러오기

~~~ python
df = pd.read_csv("http://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
~~~
###### 데이터 살펴보기

~~~ python
## 데이터 살펴보기
df      # 데이터프레임 전체 출력
df.info()
~~~
###### 데이터 전처리
~~~ python
## 데이터 전처리
# Age 컬럼의 결측값을 평균으로 대치한다.
d_mean = df["Age"].mean()
df["Age"].fillna(d_mean, inplace=True)

# Embarked 컬럼의 결측값을 최빈값으로 대치한다.
d_mode = df["Embarked"].mode()[0]
df["Embarked"].fillna(d_mode, inplace=True)

#  Sex 컬럼의 값을 1과 0으로 레이블인코딩 한다.
from sklearn.preprocessing import LabelEncoder
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Embarked 컬럼의 값을 레이블인코딩한다.
from sklearn.preprocessing import LabelEncoder
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

# 파생변수 생성 - SibSp, Parch 값을 더해서 FamilySize 컬럼을 생성한다.
df["FamilySize"] = df["SibSp"] + df["Parch"]

df.describe()
df
~~~
###### 분석 데이터셋 준비
~~~ python
## 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]]
y = df["Survived"]

# 분석 데이터셋 분할(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

~~~
(712, 6)
(179, 6)
(712,)
(179,)
~~~

###### 데이터 분석 수행
- 주어진 데이터로 탑승자의 생존을 구분하는 분류문제이다.
- 분류를 위한 알고리즘 중에서 의사결정나무를 이용한다.
	- 사이킷런의 DecisionTreeClassifier를 이용한다.
	- DecisionTreeClassifier 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train)를 입력해 호출하면 학습이 수행된다.
- 학습이 완료된 dt 객체에서 테스트 데이터셋으로 분류(예측) 수행
	- predict() 함수에 테스트 데이터셋 X_test를 입력값으로 준다.
	- X_test에 대해서 분류(예측)가 수행되며, 그 결과를 지정한 변수(pred)에 저장한다.
~~~ python
## 데이터 분석 수행
# 의사결정나무 객체 생성
dt = DecisionTreeClassifier(random_state=11)
dt.fit(X_train, y_train)        # 학습 수행

# 학습이 완료된 dt객체에서 테스트 데이터셋으로 예측 수행
pred = dt.predict(X_test)
~~~
###### 성능평가 및 시각화
- 분류(예측) 결과(pred)와 실제 분류 결과(y_test)를 비교하여 정확도를 평가한다.
- 사이킷런의 accuracy_score() 함수로 정확도 측정
	- 첫번째 파라미터로 분류데이터셋(y_test), 두번째 파라미터로 분석결과 분류(예측)된 데이터셋(pred)를 입력한다.
	- 모델의 정확도는 0.787인 것을 확인할 수 있으며, 79% 정확도로 생존자를 분류(예측)하였다.
~~~ python
## 성능평가 및 시각화
# 모델 성능 - 정확도 측정
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)
~~~

~~~
0.7877094972067039
~~~

#### [[KNN (K-Neareat Neighbor)|KNN]]
##### 알고리즘
- 개요
	- [[지도학습]]의 한 종류로, 정답이 있는 데이터를 사용하여 분류 작업을 한다.
	- 서로 가까운 점들은 유사하다는 가정하여, 데이터로부터 거리가 가까운 K개의 다른 데이터의 정답(목표값)을 참조하여 분류한다.
- 특징
	- 거리 기반 연산으로, 숫자에 구분된 속성에 우수한 성능을 보인다.
	- 전체 데이터와의 거리를 계산하기 때문에 차원(벡터)의 크기가 크면 계산량이 많아진다.
##### 분석 수행
- 목표
	- iris 데이터셋을 사용하여 꽃잎의 정보를 가지고 붓꽃의 품종(3종류)을 분류(예측)하는 문제를 KNN 알고리즘을 사용해서 해결한다.
- 접근방법
	- KNN 알고리즘은 K값에 따라 정확도가 달라지므로, 적절한 K 값을 찾는 것이 매우 중요하다.
	  >  K=3인 경우 새로운 데이터로 부터 가장 가까운 이웃을 3개 찾고, 그 중에서 가장 개수가 많은 값으로 분류한다. 
	  >  K=6일 경우 6개 중에서 가장 개수가 많은 값으로 분류한다.  
	- 분석데이터 준비 후, K=3, 6, 9의 3가지 경우를 모두 학습하여 정확도를 비교해본다
###### 필요 패키지 임포트
~~~ python
## KNN(K-Nearest Neighbor) 알고리즘
import numpy as np
import pandas as pd
import sklearn

## 1. 필요 패키지 Import
# KNN 분류모델을 위한 패키지 임포트
from sklearn.neighbors import KNeighborsClassifier

# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
~~~
###### 데이터 살펴보기
~~~python
## 3. 데이터 살펴보기
df.info()             # 데이터 결측값 유무 확인
df.describe()         # 데이터프레임의 기술통계 보여주기
df.groupby('species').sum()  # 붓꽃의 종류 확인 
~~~
###### 데이터 전처리
- 4개의 독립변수에 대해서 Min-Max 정규화(모든 값을 0~1사이 값으로 변환)를 실시한다.
~~~python
## 4. 데이터 전처리
# 각 독립변수별 Min-Max 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[["sepal_length"]] = scaler.fit_transform(df[["sepal_length"]])
df[["sepal_width"]] = scaler.fit_transform(df[["sepal_width"]])
df[["petal_length"]] = scaler.fit_transform(df[["petal_length"]])
df[["petal_width"]] = scaler.fit_transform(df[["petal_width"]])
~~~
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X= df[["sepal_length","sepal_width","petal_length","petal_width"]]
y= df["species"]

# 분석 데이터셋 분할(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~
###### 데이터분석 수행
- 주어진 데이터로 붓꽃 종류를 구분하는 분류문제이다.
- 분류를 위한 알고리즘 중에서 KNN을 이용
	- 사이킷런의 KNeighborsClassifier를 사용한다.
	- KNeighborsClassifier 객체를 생성하고, fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train)를 입력해서 호출하면 학습이 수행된다.
	- n_negihbors 값은 3부터 시작해본다. 
- 학습이 완료된 knn 객체에 테스트 데이터 셋으로 분류(예측) 수행
	- 분류(예측) 수행은 predict() 함수에 테스트 데이터셋 X_test를 입력값으로 준다.
	- 분류(예측) 수행 결과를 지정한 변수(pred)에 저장한다.
~~~python
## 6. 데이터분석 수행
# KNeighborsClassifier 객체 생성
knn = KNeighborsClassifier(n_neighbors=3)    # 일반적으로 K 값은 홀수 값을 취하며 3,6,9 를 통상적으로 사용한다.
knn.fit(X_train, y_train)                    # 학습 수행

# 학습이 완료된 knn 객체에서 테스트 데이터셋으로 예측 수행
pred = knn.predict(X_test)
~~~
###### 성능평가 및 시각화
- 분류(예측) 결과(pred)와 실제 분류 결과(y_test)를 비교하여 정확도 평가
- 사이킷런의 accuracy_score() 함수로 정확도 측정
	- 모델의 정확도는 0.933인 것을 확인할 수 있으며, 93% 정확도로 붓꽃종류를 분류(예측)하였다.
~~~python
## 7. 성능평가 및 시각화
# 모델 성능 - 정확도 측정
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)
~~~

~~~
0.9333333333
~~~

- 사이킷런의 [[혼동행렬(Confusion Matrix)]]을 사용한 성능 평가
	- 결과값의 가운데 대각선 숫자는 정확하게 분류한 숫자를 보여준다.
	- 예시의 30개 데이터 중 28개는 정확헤개 분류한 것을 알 수 있다.
~~~ python
# 모델 성능 평가 - Confusion Matrix 계산
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, pred)            # 오차행렬 구하기
print(mat)
~~~

~~~
[[ 9 0 0] 
[ 0 10 0] 
[ 0 2 9]]
~~~

- classification_report() 함수를 이용하여 평가지표를 계산
~~~ python
# 모델 성능 평가 - 평가지표 계산
from sklearn.metrics import classification_report
rpt = classification_report(y_test, pred)       # 평가지표인 precision, recall, f1-score, support를 구해서 보여준다.
print(rpt)
~~~

~~~
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         9
  versicolor       0.83      1.00      0.91        10
   virginica       1.00      0.82      0.90        11

    accuracy                           0.93        30
   macro avg       0.94      0.94      0.94        30
weighted avg       0.94      0.93      0.93        30
~~~


#### [[서포트 벡터 머신(SVM)]]

##### 알고리즘
- 개요
	- 두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때, SVM 알고리즘은 주어진 데이터 집합을 바탕으로하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 이진 선형 분류모델을 만든다.
	- 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데, ==SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘==이다.
- 특징
	- 커널 트릭을 사용함으로써 다양한 데어터의 특성을 분류할 수 있다.
	- 비교적 적은 학습 데이터로도 정확도가 높은 분류를 기대할 수 있다.
##### 분석 수행
- 목표
	- 타이타닉 데이터셋에서 탑승자들의 속성 데이터를 기반으로 생존여부를 분류(예측)한다.
- 접근방법
	- 불필요한 속성은 제거하고 SVM 알고리즘을 이용하며 학습모델을 구축한 후 예측을 수행한다.
	- 전처리 과정에서 Sex와 Embarked 컬럼에 대해서는 원-핫 인코딩을 수행해본다.
		- 원-핫 인코딩 : 범주형 데이터를 숫자형으로 변환하는 방법
###### 필요 패키지 임포트
~~~python
## SVM(Support Vector Machine) 알고리즘
import numpy as np
import pandas as pd
import sklearn


## 1. 필요 패키지 임포트
# 서포트벡터머신 분류모델을 위한 패키지 임포트
from sklearn import svm

# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
~~~
###### 데이터 살펴보기
~~~python
## 3. 데이터 살펴보기
df.info()
~~~
###### 데이터 전처리
~~~python
## 4. 데이터 전처리
# Age 컬럼의 결측값을 평균으로 대치한다.
d_mean = df["Age"].mean()
df["Age"].fillna(d_mean, inplace=True)

# Embarked 컬럼의 결측값을 최빈값으로 대치한다.
d_mode = df["Embarked"].mode()[0]
df["Embarked"].fillna(d_mode, inplace=True)

# SibSp, Parch의 값을 더해서 FamilySize 컬럼(파생변수)을 생성한다.
df["FamilySize"] = df["SibSp"]+df["Parch"]
~~~

- Sex,Embarked 컬럼은 텍스트 값으로 되어 있어 숫자 0,1로 변환하는 원-핫 인코딩을 수행한다.
	- 판다스의 get_dummies() 함수를 사용한다.
	- 원-핫 인코딩을 수행하면 새로운 컬럼이 생성되는데(onehot_sex), 생성된 컬럼 데이터를 원래 데이터 프레임에 결합한다.
~~~python
# Sex 컬럼의 값을 1과 0으로 원-핫 인코딩 한다.
onehot_sex = pd.get_dummies(df["Sex"])
df = pd.concat([df, onehot_sex], axis=1)

# Embarked  컬럼의 값을 원-핫 인코딩 한다.
onehot_embarked = pd.get_dummies(df["Embarked"])
df = pd.concat([df, onehot_embarked], axis=1)

display(df)
~~~
||PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|FamilySize|female|male|C|Q|S|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|1|0|3|Braund, Mr. Owen Harris|male|22.000000|1|0|A/5 21171|7.2500|NaN|S|1|0|1|0|0|1|
|1|2|1|1|Cumings, Mrs. John Bradley (Florence Briggs Th...|female|38.000000|1|0|PC 17599|71.2833|C85|C|1|1|0|1|0|0|
|2|3|1|3|Heikkinen, Miss. Laina|female|26.000000|0|0|STON/O2. 3101282|7.9250|NaN|S|0|1|0|0|0|1|
|3|4|1|1|Futrelle, Mrs. Jacques Heath (Lily May Peel)|female|35.000000|1|0|113803|53.1000|C123|S|1|1|0|0|0|1|
|4|5|0|3|Allen, Mr. William Henry|male|35.000000|0|0|373450|8.0500|NaN|S|0|0|1|0|0|1|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|886|887|0|2|Montvila, Rev. Juozas|male|27.000000|0|0|211536|13.0000|NaN|S|0|0|1|0|0|1|
|887|888|1|1|Graham, Miss. Margaret Edith|female|19.000000|0|0|112053|30.0000|B42|S|0|1|0|0|0|1|
|888|889|0|3|Johnston, Miss. Catherine Helen "Carrie"|female|29.699118|1|2|W./C. 6607|23.4500|NaN|S|3|1|0|0|0|1|
|889|890|1|1|Behr, Mr. Karl Howell|male|26.000000|0|0|111369|30.0000|C148|C|0|0|1|1|0|0|
|890|891|0|3|Dooley, Mr. Patrick|male|32.000000|0|0|370376|7.7500|NaN|Q|0|0|1|0|1|0|

891 rows × 18 columns

###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X = df[["Pclass","Age","Fare","FamilySize","female","male","C","Q","S"]]
y = df["Survived"]

# 분석 데이터셋 분할(7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~
###### 데이터분석 수행
- 주어진 데이터로 탑승자의 생존을 구분하는 문제는 분류문제이다.
- 분류를 위한 알고리즘 중에서 SVM을 이용
	- 사이킷런의 서포트벡터머신인 svm.SVC를 사용한다.
	- SVC 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train)를 입력해 호출하면 학습이 수행된다.
~~~python
## 6. 데이터 분석 수행
# SVM 객체 생성 - 커널 rbf 적용
sv = svm.SVC(kernel="rbf")
sv.fit(X_train, y_train)      # 학습 수행

# 학습이 완료된 dt 객체에서 테스트 데이터셋으로 예측 수행
pred = sv.predict(X_test)
~~~
###### 성능평가 및 시각화
- 사이킷런의 accuracy_score()함수로 정확도 측정
	- 실행결과 생존자 예측 정확도는 약 72%임을 알 수 있다.
- matrics 모듈의 confusion_matrix() 함수를 사용하여 오차행렬 구하기
	- 268개 데이터 중에 194(167+27)개를 정확하게 분류하였음을 알 수 있다.
	- 미생존자를 생존자로 잘못 분류한 FP는 7명, 생존자를 미생존자로 잘못분류한 FN은 67명으로 해석할 수 있다.
- matrics 모듈의 classification_report() 함수를 이용한 평가지표 계산
	- f1-score를 보면 미생존자(0)의 예측 정확도는 0.82이고, 생존자(1)의 예측도는 0.42로 차이가 있다.
~~~python
## 7. 성능평가 및 시각화
# 모델 성능 - 정확도 측정
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)

# 모델 성능 평가 - Confusion Matrix 계산
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, pred)
print(mat)

# 모델 성능 평가 - 평가지표 계산
from sklearn.metrics import classification_report
rpt = classification_report(y_test, pred)
print(rpt)
~~~

~~~
0.7238805970149254

[[167   7]
 [ 67  27]]
 
              precision    recall  f1-score   support

           0       0.71      0.96      0.82       174
           1       0.79      0.29      0.42        94

    accuracy                           0.72       268
   macro avg       0.75      0.62      0.62       268
weighted avg       0.74      0.72      0.68       268
~~~
##### SVM 커널 파라미터 조정
- SVM 알고리즘은 커널을 선택할 수 있으며, C(비용), gamma(허용 표준편차)를 이용해서 결정경계를 조절할 수 있다.
- 선형 SVM은 C를 조절해서 마진의 크기를 조절하지만, 선형분리가 주어진 차원에서 불가능할 경우 커널트릭을 통해 결정경계를 찾는다.
	- 이때 C와 gamma를 조절해서 마진의 크기를 조절한다.
- gamma가 커지면 데이터포인트별로 허용하는 표준편차가 작아지고 결정경계도 작아지면서 구부러진다.
- 파라미터 값에 따른 분석결과를 비교하면서 최적의 모델을 찾는다.

~~~ python
#1. 커널 파라미터 - rbf 적용
sv = svm.SVC(kernel='rbf')

#2. 커널 파라미터 - linear 적용, C=1, gamma=0.1
sv = svm.SVC(kernel='linear', C=1, gamma=0.1)

#1. 커널 파라미터 - rbf 적용, C=0.1, gamma=0.1
sv = svm.SVC(kernel='rbf')
~~~
#### [[로지스틱 회귀분석|로지스틱 회귀]]

##### 알고리즘
- 개요
	- 로지스틱 회귀 알고리즘은 회귀라는 명칭을 갖고 있지만, 회귀가 주로 예측을 하는 것과 달리 정답이 있는 데이터를 사용하여 분류 작업을 한다.
	- 시그모이드(sigmoid) 함수의 출력값을 각 분류 항목에 속하게 확률값으로 사용하며, 이 값은 0~1사이의 실수이다.
- 특징
	- 시그모이드 함수는 입력 값이 클수록 1에 수렴하고, 입력 값이 작으면 0으로 수렴한다.
	- 현재 갖고 있는 데이터를 통해 에러를 줄이는 방향으로 weight와 bias의 최적값을 찾아간다.
##### 분석 수행
- 분석 목표
	- iris 데이터셋을 사용하여 붓꽃의 품종을 분류하는 문제를 로지스틱 회귀 알고리즘을 사용하여 해결
- 접근 방법
	- 로지스틱 회귀는 규제의 유형과 강도에 따라 분류(예측)의 정확도가 달라지므로, 적당한 값을 찾는 것이 중요
	- 규제가 필요한 이유는 학습용 데이터만 과도하게 학습하는 경우 다른 데이터에 대해서 예측력이 낮아지는 과대적합(overfitting) 문제가 발생하는 것을 예방하기 위함
	- 규제의 유형은 LogisticRegression 클래스 내 penalty 매개변수에서 설정할 수 있으며, 기본 값은 L2 규제(릿지 방식)이고, L1 규제(라쏘 방식)을 선택할 수 있다.
	- predict_proba() 메소드를 이용하여 분류 항목에 속할 확률을 확인할 수 있다.
	- decision_fuction() 메소드를 이용하여 모델이 선형 방정식을 확인할 수 있다.

###### 필요 패키지 임포트
~~~python
## 로지스틱 회귀(Logistic Regression)

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import sklearn

# 로지스틱 회귀 분류모델을 위한 패키지 임포트

# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~

###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
~~~
###### 데이터 탐색하기
~~~python
## 3. 데이터 살펴보기
df.info()
df.describe()
~~~
###### 데이터 전처리
~~~python
## 4. 데이터 전처리
# 각 독립변수별 Min-Max 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[["sepal_length"]] = scaler.fit_transform(df[["sepal_length"]])
df[["sepal_width"]] = scaler.fit_transform(df[["sepal_width"]])
df[["petal_length"]] = scaler.fit_transform(df[["petal_length"]])
df[["petal_width"]] = scaler.fit_transform(df[["petal_width"]])
~~~
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
y = df["species"]

# 분석 데이터셋 분할(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~
###### 데이터분석 수행
- 주어진 데이터로 붓꽃 종류를 구분하는 분류 문제이다.
- 분류를 위한 알고리즘 중에서 로지스틱 회귀를 이용
	- 사이킷런의 LogisticRegression를 사용한다.
	- LogisticRegression 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train)를 입력해 호출하면 학습이 수행된다.
- 학습이 완료된 lr 객체에서 테스트 데이터셋으로 분류(예측) 수행
~~~python
## 6. 데이터분석 수행
# LogisticRegression 객체 생성
lr = LogisticRegression()
lr.fit(X_train, y_train)        # 학습 수행

# 학습이 완료된 dt객체에서 테스트 데이터셋으로 예측 수행
pred = lr.predict(X_test)
~~~
###### 성능평가 및 시각화
- 분류(예측) 결과(pred)와 실제 분류 결과(y_test)를 비교하여 정확도를 평가
	- 모델의 분류 정확도는 83%임을 알 수 있다.
~~~python
## 7. 성능평가 및 시각화
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)
~~~

~~~
0.83333333333
~~~
#### [[랜덤 포레스트]]
##### 알고리즘
- 개요
	- 다수의 의사결정 트리들을 배깅하여 분류 또는 회귀를 수행하는 [[앙상블 분석|앙상블 기법]]이다. 
	- 각 트리는 학습 데이터 중 서로 다른 데이터를 샘플링하여 일부 데이터를 제외한 후 최적의 특징을 찾아 트리를 분기한다.
- 특징
	- 다양한 분야에서 비교적 좋은 성능을 보여준다.
	- 트리들이 서로 조금씩 다른 특성을 갖게 되어 일반화 성능을 향상할 수 있다.
	- 샘플링을 하는 과정에서 한 샘플이 중복되어 추출될 수도 있다.
##### 분석 수행
- 분석 목표
	- 타이타닉 데이터셋에서 탑승자들의 생존여부(Survived)를 예측
- 접근 방법
	- 불필요한 속성을 제거하고 전처리과정을 거친 후,
	- 사이킷런의 랜덤 포레스트 알고리즘을 이용하여 학습 모델을 구축한 후 예측을 수행한다.
	- 랜덤 포레스트 알고리즘에서 사용할 트리 모델의 개수(n_estimators)와 개별 트리의 깊이(max_depth) 매개변수 값을 잘 조절하여 예측의 정확도를 높인다.

###### 필요 패키지 임포트
~~~ python
## 랜덤 포레스트(Random Forest) 알고리즘

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import sklearn

# 랜덤 포레스트 분류모델을 위한 패키지 임포트
from sklearn.ensemble import RandomForestClassifier
# 학습 및 테스트 데이터 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~

###### 데이터 불러오기 
~~~ python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
~~~

###### 데이터 탐색하기
~~~ python
## 3. 데이터 살펴보기
df.info()
df.describe()
~~~

###### 데이터 전처리
- 사이킷런의 LabelEncoder() 함수를 사용하여 라벨 인코딩을 수행
	- 텍스트로 되어있는 Sex컬럼은 숫자 0(female), 1(male)로 변환하는 레이블 인코딩을 수행한다.
	- Embarked 역시 인코딩을 수행하며, 사이킷런의 LabelEncoder() 함수를 사용한다.
	- LabelEncoder를 객체로 생성한 후 fit_transform() 함수를 사용해서 구현한다.
~~~ python
## 4. 데이터 전처리
# Age 컬럼의 결측값을 평균으로 대치한다.
d_mean = df["Age"].mean()
df["Age"].fillna(d_mean, inplace=True)

# Embarked 컬럼의 결측값을 최빈값으로 대치한다.
d_mode = df["Embarked"].mode()[0]
df["Embarked"].fillna(d_mode, inplace=True)

# Sex 컬럼의 값을 1과 0으로 레이블인코딩 한다.
from sklearn.preprocessing import LabelEncoder
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Embarked 컬럼의 값에 레이블인코딩 한다.
from sklearn.preprocessing import LabelEncoder
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

# SibSp, Parch의 값을 더해서 FamilySize 컬럼(파생변수)를 생성한다.
df["FamilySize"] = df["SibSp"]+df["Parch"]
~~~

###### 분석 데이터셋 준비
~~~ python
## 5. 분석 데이터셋 준비
# X는 독립변수(설명변수), y는 종속변수(목표변수)
X = df[["Pclass","Age","Fare","Embarked","FamilySize"]]
y = df["Survived"]

# 분석 데이터셋 분할(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

###### 데이터분석 수행
- 주어진 데이터로 탑승자의 생존을 구분하는 분류문제이다.
- 분류를 위한 알고리즘 중에서 랜덤 포레스트를 이용
	- 사이킷런의 랜덤 포레스트인 RandomForestClassifier를 사용한다.
	- RandomForestClassifier 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train)를 입력해 호출하면 학습이 수행된다.
- 학습이 완료된 rf 객체에서 테스트 데이터셋으로 분류(예측) 수행
~~~ python
## 6. 데이터 분석 수행
# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)
rf.fit(X_train, y_train)          # 학습 수행

# 학습이 완료된 rf객체에서 테스트 데이터셋으로 예측 수행
pred = rf.predict(X_test)
~~~

###### 성능평가 및 시각화
- 정확도 측정을 위한 사이킷런의 accuracy_score() 함수 사용
	- 예측 정확도는 86%임을 알 수 있다.
~~~ python
## 7. 성능평가 및 시각화
# 모델 성능 - 정확도 측정
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)
~~~

~~~
0.860333519
~~~

### [[지도학습]] - 회귀(예측)
- 정의
	- 지도학습은 크게 분류와 회귀로 나뉘는데, 분류는 예측값이 카테고리와 같은 이산형 값이고, 회귀는 예측값이 연속형 숫자 값이다.
	- 그 중 회귀는 선형/비선형 여부, 독깁변수의 개수, 종속변수의 개수에 따라 여러가지 유형으로 나누어진다
		
		|독립변수개수|회귀계수|
		|---|---|
		|1개 : 단일 회귀|선형 : 선형 회귀|
		|여러개 : 다중 회귀|비선형 : 비선형 회귀|
- 대표 알고리즘
	- [[단순 선형 회귀분석|선형회귀분석]], 비선형 회귀분석, [[의사결정 트리 분석|의사결정나무]](회귀트리모형), SVR, 신경망 모형, 로지스틱 회귀, [[랜덤 포레스트]](회귀)
#### [[단순 선형 회귀분석]]
##### 알고리즘
- 단순 선형 분석은 가장 단순한 분석으로 한 개의 종속변수 y와 한 개의 독립변수 x로 구성된 선형 회귀이다. 
  (a는 회귀계수, b는 y 절편)
$$y = ax + b$$
- 실제값과 회귀모델에 의해 예측한 값의 차이를 잔차(오류)라고하며, 잔차의 합의 최소가 되는 회귀 계수를 찾아내는게 회귀모델의 목표이다.
- 잔차는 +,-가 될 수 있기 때문에, 잔차의 제곱의 합(Residual Sum of Squares : RSS)을 최소로 하는 최소제곱법을 이용하여 최소가 되는 모델을 만든다.
##### 분석 수행
- 목표
	- UCI 자동차 연비 데이터셋을 사용하여 자동차 연비를 예측하는 모델을 만든다.
- 접근방법
	- 종속변수는 mpg(연비)로 한다.
	- mpg에 대해 나머지 변수들의 상관성을 산점도를 이용하여 시각적으로 분석한다.
	- 상관성이 있다고 판단되는 변수를 선택한 후 각가에 대해 선형 알고리즘을 적용하여 모델을 생성
	- 결정계수 값을 구하여 모델 성능을 평가
###### 필요 패키지 임포트
~~~python
## 단순 선형 회귀분석 알고리즘

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt           # 맷플롯립 패키지 임포트

# 선형 회귀모델을 위한 패키지 임포트
from sklearn.linear_model import LinearRegression
# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv")
~~~
###### 데이터 탐색하기 & 변수 선택
- Matplotlib의 plot() 메소드에 kind='scatter' 옵션을 적용하여 산점도 그리기
	- y축에는 mpg열을 지정하고, x축에 각 열을 하나씩 지정하면서 두 변수의 상관성을 살펴본다.
	- 6개의 변수 중에서 housepower와 weight에 음의 상관관계가 있음을 알 수 있고, 이 두 변수에 선형회귀 모델을 만들어본다.
~~~python
## 3. 데이터 살펴보기 & 변수선택
df
df.info()

# 종속변수 mpg와 다른 변수들 간의 상관관계 분석
df.plot(kind='scatter', x='cylinders', y='mpg')
plt.show()

df.plot(kind='scatter', x='displacement', y='mpg')
plt.show()

df.plot(kind='scatter', x='horsepower', y='mpg')
plt.show()

df.plot(kind='scatter', x='weight', y='mpg')
plt.show()

df.plot(kind='scatter', x='acceleration', y='mpg')
plt.show()

df.plot(kind='scatter', x='model-year', y='mpg')
plt.show()
~~~

![[데이터분석_시각화_산점도_예시1.png]]
###### 데이터 전처리
~~~python
## 4. 데이터 전처리
# 결측값이 있는 행전체 제거 (axis가 1이면 열을 제거)
df = df.dropna(axis=0)
df.info()
~~~
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# 분석 데이터셋 준비 : weight-mpg 분석
X = df[["weight"]]        # 독립변수(설명변수)
y = df["mpg"]             # 종속변수(목표변수)

# 분석 데이터셋 분할(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

~~~
(316, 1)
(80, 1)
(316,)
(80,)
~~~

###### 데이터 분석 수행
- 사이킷런의 선형 회귀분석 모듈인 LinearRegression을 사용
	- LinearRegrassion 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train) 를 입력해 호출하면 학습이 수행된다.
- 최적 선형 회귀식 계수 a, b를 도출
	- 회귀식의 기울기인 a는 lr.coef 에, y절편인 b는 lr.intercept 에 저장된다.
~~~python
## 6. 데이터분석 수행
# LinearRegression 객체 생성
lr = LinearRegression()
lr.fit(X_train, y_train)        # 학습 수행

# 회귀식의 기울기와 y절편 출력
print("기울기 a: ", lr.coef_)
print("y 절편 b: ", lr.intercept_)

# 학습이 완료된 lr객체에서 테스트 데이터셋으로 예측 수행
pred = lr.predict(X_test)
~~~

~~~
기울기 a:  [-0.00774371]
y 절편 b:  46.62501834798047
~~~
###### 성능평가 및 시각화
- 선형 회귀분석 평가는 결정계수인 $R^2$ 점수로 예측 정확도를 판단할 수 있다.
	- 사이킷런의 r2_score() 함수를 사용한다.
	- 첫번째 파라미터로 목표변수에 대한 테스트 데이터셋(y_test), 두번째 파라미터로 분석 결과로 예측된 데이터셋(pred)을 입력한다.
	- 실행결과 $R^2$ 값이 0.7015...로 정확도는 70% 이상임을 알 수 있다,
~~~python
## 7. 성능평가 및 시각화
# 모델 성능 평가 - 테스트 데이터셋
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)
print(score)                        # 테스트 정확도는 70% 이상임이 확인됨
~~~

~~~
0.7015633872576372
~~~

- horsepower-mpg 선형 회귀분석
~~~python
## 추가작업 해보기 (horsepower-mpg 분석)

## 5. 분석 데이터 셋 준비
X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

## 6. 데이터 분석 수행
lr = LinearRegression()
lr.fit(X_train,y_train)

print("기울기 a:", lr.coef_)
print("y 절편 b:", lr.intercept_)
pred = lr.predict(X_test)

## 7. 성능평가 및 시각화
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)
print(score)

~~~

~~~
기울기 a: [-0.16035108]
y 절편 b: 40.313418327064824

0.6039842414538836
~~~
###### (참고) 상관관계 분석
- 독립변수와 종속변수 사이에 선형관계 유무는 산점도를 통해서도 가능하지만, 상관분석을 이용해서 정확하게 파악할 수 있다.
- 상관관계를 구하는 여러가지 방법 중에서 '피어슨 상관계수'를 많이 사용한다.
- 판다스 데이터프레임의 corr() 메소드를 이용한 상관관계 분석
	- 0.7 이상이 강한 상관관계를 가지고 있다고 기준을 정의할 경우 총 4개의 컬럼이 상관관계가 mpg와 상관관계가 있다고 판단할 수 있다.
~~~python
## 8. 상관관계 분석
# 변수들 간의 상관관계 분석
corr = df.corr(method="pearson")
print(corr)
~~~

~~~
                   mpg  cylinders  displacement  horsepower    weight  \
mpg           1.000000  -0.775680     -0.804711   -0.777575 -0.832725   
cylinders    -0.775680   1.000000      0.950706    0.843751  0.896058   
displacement -0.804711   0.950706      1.000000    0.897787  0.932729   
horsepower   -0.777575   0.843751      0.897787    1.000000  0.864350   
weight       -0.832725   0.896058      0.932729    0.864350  1.000000   
acceleration  0.421159  -0.504844     -0.542713   -0.687241 -0.415462   
model-year    0.581144  -0.352554     -0.374620   -0.420697 -0.311774   

              acceleration  model-year  
mpg               0.421159    0.581144  
cylinders        -0.504844   -0.352554  
displacement     -0.542713   -0.374620  
horsepower       -0.687241   -0.420697  
weight           -0.415462   -0.311774  
acceleration      1.000000    0.294588  
model-year        0.294588    1.000000  
~~~

#### [[다중 선형 회귀분석]]
##### 알고리즘
- 하나의 독립변수가 아닌 여러 개의 독립변수를 사용하는 회귀분석 기법이다.
	- [[단순 선형 회귀분석]]이 독립변수를 하나 가지고 있는 선형 회귀분석이라면,
	- 다중 선형 회귀분석은 독립변수가 두 개 이상이고 종속변수가 y 하나인 선형 회귀분석이다.
	  (a, b, ... 는 회귀계수, c는 y절편)
$$y=ax_1 + bx_2 + ... + c$$
##### 분석 수행
- 분석 목표
	- 1999년 미국 캘리포니아 인구가구 통계 데이터셋에서 주택중위가치(median_house_value)에 영향을 주는 변수들을 찾아보고, 이 변수들을 포함하는 다중선형회귀모델을 생성하여 성능을 평가해본다.
- 접근 방법
	- 종속변수는 median_house_value
	- 상관성이 있다고 판단되는 변수를 선택한 후 각가에 대해 다중선형회귀 알고리즘을 적용하여 모델을 생성
	- 결정계수 값을 구해서 모델 성능을 평가
###### 필요 패키지 임포트
~~~python
## 다중 선형 회귀분석을 이용한 예측 문제 해결

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt           # 맷플롯립 패키지 임포트

# 선형 회귀 모델을 위한 패키지 임포트
from sklearn.linear_model import LinearRegression
# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
~~~
###### 데이터 탐색하기 
~~~python
## 3. 데이터 살펴보기
df
~~~
###### 데이터 전처리
- 변수들의 상관관계 분석시에 median_income을 제외하면 median_house_value 와의 상관성이 낮은 것을 알 수 있다.
- 전체 컬럼을 독립변수로 사용해 분석을 수행해 본다.
~~~python
## 4. 데이터 전처리
# 결측값이 있는 행 전체 제거 (axis가 1이면 열을 제거)
df = df.dropna(axis=0)

# ocean_proximity는 범주형 값으로 분석에서 제외
df = df.drop("ocean_proximity", axis=1)

# 변수들 간의 상관관계 분석
corr = df.corr(method="pearson")
print(corr)
~~~

~~~
                    median_house_value  
longitude                    -0.045398  
latitude                     -0.144638  
housing_median_age            0.106432  
total_rooms                   0.133294  
total_bedrooms                0.049686  
population                   -0.025300  
households                    0.064894  
median_income                 0.688355  
median_house_value            1.000000  
~~~
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# median_house_value를 제외한 나머지를 독립변수로 함
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]                  # 종속변수(목표변수)

# 분석 데이터셋 분할(7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

~~~
(14303, 8)
(6130, 8)
(14303,)
(6130,)
~~~

###### 데이터 분석 수행
- 사이킷런의 선형 회귀분석 모듈인 LinearRegression을 사용
	- LinearRegression 객체를 생성하고 fit() 함수에 학습용 데이터(X_train)와 결정값 데이터(y_train) 를 입력해 호출하면 학습이 수행된다.
- 완성된 다중선형 회귀계수를 출력
	- 총 8개의 기울기와 1개의 y절편이 있다.
~~~python
## 6. 데이터 분석 수행
# LinearRegression 객체 생성
lr = LinearRegression()
lr.fit(X_train, y_train)          # 학습 수행

# 회귀식의 기울기와 y절편 출력
print("기울기 : ", lr.coef_)              # 다중회귀이기 때문에 다수의 기울기가 출력됨
print("y절편 : ", lr.intercept_)

# 학습이 완료된 lr객체에서 테스트 데이터셋으로 예측 수행
pred = lr.predict(X_test)
~~~

~~~
기울기 :  [-4.21262308e+04 -4.20623763e+04  1.18784999e+03 -8.57874086e+00
		  1.18123421e+02 -3.55751755e+01  3.73676747e+01  4.03297253e+04]
y절편 :  -3530241.307796566
~~~
###### 성능평가 및 시각화
- 다중 선형회귀분석 평가도 결정계수인 $R^2$ 점수로 예측 정확도를 판단할 수 있다. 결정계수 값이 클수록 모형의 예측능력이 좋다고 판단한다.
	- 사이킷런의 r2_score() 함수를 사용한다.
	- 첫번째 파라미터로 목표변수에 대한 테스트 데이터셋(y_test), 두번째 파라미터로 분석 결과로 예측된 데이터셋(pred)을 입력한다.
	- 실행결과 $R^2$ 값이 0.6445...로 정확도는 64% 이상임을 알 수 있다.
~~~python
## 7. 성능평가 및 시각화
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)
print(score)

# 학습이 완료된 lr객체에서 학습 데이터셋으로 예측 수행
pred = lr.predict(X_train)

# 모델 성능 평가 - 학습 데이터셋
from sklearn.metrics import r2_score
score = r2_score(y_train, pred)
print(score)
~~~

~~~
0.6445130291082337
0.6334125389213838
~~~

#### [[의사결정 트리 분석|의사결정나무]]
##### 알고리즘
- 분류 기능과는 달리 각 항목에서의 범주를 예측하는 것이 아니라 어떠한 값 자체를 예측하는 것
##### 분석 수행
- 분석 목표
	- 1999년 미국 캘리포니아 인구가구 통계 데이터셋에서 주택중위가치(median_house_value)에 영향을 주는 변수들을 찾아보고, 이 변수들을 포함하는 트리모델을 생성하여 성능을 평가해본다.
- 접근 방법
	- 종속변수는 median_house_value로 한다.
	- 상관성이 있다고 판단되는 변수를 선택한 후 각각에 대해 의사결정나무 알고리즘을 적용하여 모델을 생성한다.
	- 평균제곱오차(MSE) 값을 구해서 모델 성능을 평가한다.
###### 필요 패키지 임포트
- 의사결정나무 분석을 위해서 사이킷런의 DecisionTreeRegressor 클래스를 사용한다.
~~~python
## 의사결정나무 알고리즘

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt     # 맷플롯립 패키지 임포트

# 의사결정 모델을 위한 패키지 임포트
from sklearn.tree import DecisionTreeRegressor
# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
~~~
###### 데이터 탐색하기 
~~~python
## 3. 데이터 살펴보기
df
~~~
###### 데이터 전처리
- 변수들의 상관관계 분석시에 median_income을 제외하면 median_house_value 와의 상관성이 낮은 것을 알 수 있다.
- 전체 컬럼을 독립변수로 사용해 분석을 수행해 본다.
~~~python
## 4. 데이터 전처리
# 결측값이 있는 행 전체 제거
df = df.dropna(axis=0)

# ocean_proximity 는 범주형 값이므로 제거
df = df.drop("ocean_proximity", axis=1)

# 변수들간의 상관관계 분석
corr = df.corr(method = "pearson")
print(corr)
~~~

~~~
                    median_house_value  
longitude                    -0.045398  
latitude                     -0.144638  
housing_median_age            0.106432  
total_rooms                   0.133294  
total_bedrooms                0.049686  
population                   -0.025300  
households                    0.064894  
median_income                 0.688355  
median_house_value            1.000000  
~~~
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# median_house_value를 제외한 나머지를 독립변수로 함
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]        # 종속변수(목표변수)

# 분석 데이터셋 분할(7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

print("\n")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

~~~
(14303, 8)
(6130, 8)
(14303,)
(6130,)
~~~
###### 데이터 분석 수행
- 사이킷런의 의사결정나무 분석 모듈인 DecisionTreeRegressor을 사용
- 학습이 완료된 dtr 객체에서 데이터셋으로 예측을 수행
~~~python
## 6. 데이터 분석 수행
# DecisionTreeRegressor 객체 생성
dtr = DecisionTreeRegressor(max_depth=3, random_state=43)
dtr.fit(X_train, y_train)     # 학습 수행

# 학습이 완료된 dtr객체에서 테스트 데이터셋으로 예측 수행
pred = dtr.predict(X_test)
~~~
###### 성능평가 및 시각화
- 의사결정나무 분석의 평가는 평균제곱오차(MSE)로 예측 정확도를 판단할 수 있다. MSE 값이 작을수록 모형의 예측 능력이 좋다고 판단한다.
- 사이킷런의 mean_squared_error() 함수로 정확도 측정
	- 실행결과 MSE 값이 높게 나왔으나 이는 다른 예측 모델을 구현하여 함께 비교해볼 필요가 있다.
- 학습 데이터 셋에 대해서도 예측을 수행하고 성능을 평가해보기
	- 실행결과 MSE 값이 테스트 데이터셋을 이용한 성능평가 값과 큰 차이가 없음을 확인할 수 있다.
~~~python
## 7. 성능평가 및 시각화
# 모델 성능 평가 - 테스트 데이터셋
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print(mse)

# 학습이 완료된 dtr 객체에서 학습 데이터셋으로 예측 수행
pred = dtr.predict(X_train)

# 모델 성능 평가 - 학습 데이터셋
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, pred)
print(mse)
~~~

~~~
6793101269.876856
6684086804.552605
~~~
#### [[랜덤 포레스트]]
##### 알고리즘
- 다수의 [[의사결정 트리 분석|의사결정 트리]]들을 배깅하여 분류 또는 회귀를 수행하는 [[앙상블 분석|앙상블 기법]]이다.
- 각 트리는 학습 데이터 중 서로 다른 데이터를 샘플링하여 일부 데이터를 제외한 후 최적의 특징을 찾아 트리를 분기한다.
##### 분석 수행
- 분석 목표
	- 1999년 미국 캘리포니아 인구가구 통계 데이터셋에서 주택중위가치(median_house_value)에 영향을 주는 변수들을 찾아보고, 이 변수들을 포함하는 트리모델을 생성하여 성능을 평가해본다.
- 접근 방법
	- 종속변수는 median_house_value로 한다.
	- 상관성이 있다고 판단되는 변수를 성택한 후 각각에 대해 랜덤 포레스트 알고리즘을 적용하여 모델을 생성한다.
	- 평균제곱오차(MSE) 값을 구해서 모델의 성능을 평가한다.
###### 필요 패키지 임포트
~~~python
## 랜덤 포레스트 알고리즘
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt       # 맷플롯립 패키지 임포트

## 1. 데이터 임포트
# 랜덤 포레스트 모델을 위한 패키지 임포트
from sklearn.ensemble import RandomForestRegressor
# 학습 및 테스트 데이터셋 분리를 위한 패키지 임포트
from sklearn.model_selection import train_test_split
~~~
###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
~~~
###### 데이터 탐색하기 
~~~python
## 3. 데이터 살펴보기
df.info()
~~~
###### 데이터 전처리
- 변수들의 상관관계 분석시에 median_income을 제외하면 median_house_value 와의 상관성이 낮은 것을 알 수 있다.
- 전체 컬럼을 독립변수로 사용해 분석을 수행해 본다.
~~~python
## 4. 데이터 전처리
# 결측값이 있는 행전체 제거 (axis가 1이면 열을 제거)
df = df.dropna(axis=0)

# ocean_proximity는 범주형 값으로 분석에서 제외
df = df.drop("ocean_proximity", axis=1)

# 변수들 간의 상관관계 분석
corr = df.corr(method="pearson")
print(corr)
~~~

~~~
                    median_house_value  
longitude                    -0.045398  
latitude                     -0.144638  
housing_median_age            0.106432  
total_rooms                   0.133294  
total_bedrooms                0.049686  
population                   -0.025300  
households                    0.064894  
median_income                 0.688355  
median_house_value            1.000000  
~~~

###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# median_house_value를 제외한 나머지를 독립변수로 함
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]        # 종속변수(목표변수)

# 분석 데이터셋 분할(7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
~~~

~~~
(14303, 8)
(6130, 8)
(14303,)
(6130,)
~~~
###### 데이터 분석 수행
- 사이킷런의 랜덤 포레스트 분석 모듈인 RandomForestRegressor 을 사용
- 학습이 완료된 rfr 객체에서 테스트 데이터셋으로 예측을 수행
~~~python
## 6. 데이터 분석 수행
# RandomForestRegressor 객체 생성
rfr = RandomForestRegressor(max_depth=3, random_state=42)
rfr.fit(X_train, y_train)         # 학습 수행
        
# 학습이 완료된 rfr 객체에서 테스트 데이터셋으로 예측 수행
pred = rfr.predict(X_test)
~~~
###### 성능평가 및 시각화
- 랜덤 포레스트 분석의 평가는 평균제곱오차(MSE)로 예측 정확도를 판단할 수 있따. MSE 값이 작을수록 모형의 예측 능력이 좋다고 판단한다.
- 사이킷런의 mean_squared_error() 함수로 정확도 측정
	- MSE 값이 높게 나왔으나 이는 앞에서 구현한 의사결정트리(6793101269)보다 다소 낮은 값으로 예측력이 개선되었음을 확인할 수 있다.
~~~python
## 7. 성능평가 및 시각화
# 모델 성능 평가 - 테스트 데이터셋
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred)
print(mse)
~~~

~~~
6447828605.376922
~~~

### 비지도학습

#### [[군집분석]]
##### 알고리즘
- K-means 알고리즘
	- 서로 유사한 데이터는 동일 그룹으로, 유사하지 않은 데이터는 다른 그룹으로 분류하는 군집분석
	- 대표적인 K-means 알고리즘은 K(클러스터 중심 개수), means(각 클러스터 중심과의 평균 거리)로 구성된다.
	- 데이터셋에서 K개의 centroids를 임시 지정한 뒤 가장 가까운 Centroids가 속한 그룹에 할당, 그리고 다시 centroid를 업데이트하여 반복함으로써 각 클러스터와 거리 차이의 분산을 최소화 하는 방식으로 동작함.
##### 분석 수행
- 분석 목표
	- iris 데이터셋으로 K-means 클러스터링을 사용하여 비슷한 붓꽃끼리 그룹화하고 성능을 평가해본다.
- 접근 방법
	- 변수들 간의 상관성을 시각화해본다.
	- 라벨이 없다는 가정하에 K-means 알고리즘으로 데이터를 그룹화시킨다.
###### 필요 패키지 임포트
~~~python
## 군집분석을 이용한 문제해결 : K-means 알고리즘

## 1. 필요 패키지 임포트(import)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans        # K-Means 패키지 임포트
~~~

###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
df = pd.read_csv("http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
~~~

###### 데이터 살펴보기
~~~python
## 3. 데이터 살펴보기
df
~~~

###### 데이터 전처리
~~~python
## 4. 데이터 전처리
# species 컬럼의 값을 0,1,2로 레이블인코딩 한다.
from sklearn.preprocessing import LabelEncoder
df["species"] = LabelEncoder().fit_transform(df["species"])
df.head()

df_copy = df        # 기존 데이터프레임 복사본 생성
~~~

###### 분석 데이터셋 준비
- 데이터 특징 핵심인자 탐색을 위한 사전 시각화로 pairplot() 함수를 이용
	- 데이터 특징들 간의 상관관계를 표현한 결과에서 petal length를 통해 종류 구분이 가능함을 판단할 수 있음
~~~python
## 5. 분석 데이터셋 준비
# 변수간 상관관계 시각화
import seaborn as sns
from matplotlib import pyplot as plt            # 그래프 시각화
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
sns.pairplot(df, hue="species")                 # 데이터특징들간의 상관관계 표현
plt.show()
~~~
![[군집분석_변수간 상관관계 시각화.png]]
###### 데이터 분석 수행
- 사이킷런의 K-means 모델인 KMeans 클래스(함수) 사용
	- KMeans 객체를 생성하고 fit() 함수에 붓꽃 데이터프레임을 학습시킨다.
- 기존 데이터에 예측된 군집 결과를 붙여서 출력
	- 결과 표를 보면 sepecies와 cluster가 0과 1로 다르게 나오는데, K-means에서는 값 자체보다 유사항 데이터를 같은 값으로 묶어주고 있는지가 중요하다. 
	- 즉, 결과값이 유사한 붓꽃데이터를 같은 종으로 일관성있게 구분하고 있음을 알 수 있다.
~~~python
## 6. 데이터분석 수행
# KMeans 객체 생성
cluster1 = KMeans(n_clusters=3, n_init=10, max_iter=500, random_state=42, algorithm='auto')

# 생성모델로 데이터 학습
cluster1.fit(df)
KMeans(max_iter=500, n_clusters=3, random_state=42)

# 결과 값을 변수에 저장
cluster_center = cluster1.cluster_centers_     # 각 군집의 중심점 결과 저장 
cluster_prediction = cluster1.predict(df)      # 각 예측군집 결과 저장
print(pd.DataFrame(cluster_center))
print(cluster_prediction)

# 기존 데이터에 예측된 군집 결과를 붙인다.
df_copy["cluster"] = cluster_prediction
df_copy
~~~

~~~
          0         1         2         3         4
0  6.622449  2.983673  5.573469  2.032653  2.000000
1  5.006000  3.428000  1.462000  0.246000  0.000000
2  5.915686  2.764706  4.264706  1.333333  1.019608

[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 2 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0]
~~~

||sepal_length|sepal_width|petal_length|petal_width|species|cluster|
|---|---|---|---|---|---|---|
|0|5.1|3.5|1.4|0.2|0|1|
|1|4.9|3.0|1.4|0.2|0|1|
|2|4.7|3.2|1.3|0.2|0|1|
|3|4.6|3.1|1.5|0.2|0|1|
|4|5.0|3.6|1.4|0.2|0|1|
|...|...|...|...|...|...|...|
|145|6.7|3.0|5.2|2.3|2|0|
|146|6.3|2.5|5.0|1.9|2|0|
|147|6.5|3.0|5.2|2.0|2|0|
|148|6.2|3.4|5.4|2.3|2|0|
|149|5.9|3.0|5.1|1.8|2|0|

150 rows × 6 columns
###### 성능평가 및 시각화
- 비지도학습인 K-means는 실제 정답이 없으므로 일반적인 성능평가 대신에 적절한 K개를 설정하였는지 평가할 수 있다.
- K 개수와 inertia 비교시각화로 K=3일 때가 빠르게 줄어들기 시작한 시점임을 알 수 있다.
~~~python
## 7. 성능평가 및 시각화
# 적절한 K에 대해 붓꽃 데이터프레임을 넣어 K와 inertia를 비교
# 값(3)이 적합한 변화시점임을 알 수 있음
scope = range(1,10)
inertias = []

for k in scope:
    model = KMeans(n_clusters=k)
    model.fit(df)
    inertias.append(model.inertia_)

# K 개수와 Inertia 비교 시각화
plt.figure(figsize=(4,4))

plt.plot(scope, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.show()
~~~

![[군집분석_성능평가_K개수와 inertia 비교.png]]
#### [[연관분석]]

##### 알고리즘
- 연관분석은 하나의 거래나 사건에 포함된 항목 간의 관련성을 파악하여 둘 이상의 항목들로 구성된 연관성 규칙을 도출한다.
- 장바구니 분석으로 알려져 있으며, 연관성을 찾아내기 위해 연관성을 비교할 수 있는 규칙이 필요한데 규칙을 발견하기 위해 지지도, 신뢰도, 향상도를 평가 척도로 사용한다.
##### 분석 수행
- 분석 목표
	- 장바구니 구매 물품 데이터 셋으로 물품들 간의 연관관계를 파악하고 성능을 평가해본다.
- 접근 방법
	- 장바구니 구매항목별 관련성이 있는지 알아보기 위해 Apriori 알고리즘으로 연관분석을 적용한다.

###### 필요 패키지 임포트
~~~python
## 연관분석 알고리즘

## 1. 필요 패키지 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# apriori, association_rules 모듈 호출
from mlxtend.frequent_patterns import apriori, association_rules
~~~

###### 데이터 불러오기
~~~python
## 2. 데이터 불러오기
from google.colab import drive
drive.mount('/content/drive')

# pandas의 read_csv 함수 통한 csv 데이터 로드
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/retail_dataset.csv', sep=',')
~~~

###### 데이터 살펴보기
~~~python
## 3. 데이터 살펴보기
df.info()
~~~

###### 데이터 전처리
- 장바구니 목록 값을 수치로 표현하기 위해 항목당 매칭될 경우 1로, 아니면 0으로 표시한다. (원-핫 인코딩 처리)
~~~python
## 4. 데이터 전처리
# 장바구니 데이터 고유항목 구분 출력
items = set()
for col in df:
  items.update(df[col].unique())

items

# 장바구니 목록 값(텍스트)을 수치로 표현-각 항목당 매칭될 경우 1로, 아니면 0으로 표시(one-hot encoding)
itemset = set(items)
encoding = []
for index, row in df.iterrows():
  rowset = set(row)
  labels = {}
  dismatching = list(itemset - rowset)
  matching = list(itemset.intersection(rowset))
  for i in dismatching:
    labels[i] = 0
  for j in matching:
    labels[j] = 1
  encoding.append(labels)
encoding[0]
result = pd.DataFrame(encoding)

result
~~~
|Milk|Bagel|Eggs|Meat|Pencil|Cheese|Wine|Bread|Diaper|
|---|---|---|---|---|---|---|---|---|
|0|0|0|1|1|1|1|1|1|1|
|1|1|0|0|1|1|1|1|1|1|
|2|1|0|1|1|0|1|1|0|0|
|3|1|0|1|1|0|1|1|0|0|
|4|0|0|0|1|1|0|1|0|0|
|...|...|...|...|...|...|...|...|...|...|
|310|0|0|1|0|0|1|0|1|0|
|311|1|0|0|1|1|0|0|0|0|
|312|0|0|1|1|1|1|1|1|1|
|313|0|0|0|1|0|1|0|0|0|
|314|0|1|1|1|0|0|1|1|0|

315 rows × 9 columns
###### 분석 데이터셋 준비
~~~python
## 5. 분석 데이터셋 준비
# 첫째 NaN 열항목 삭제
result = result.drop(result.columns[0], axis=1)         
display(result)
~~~

###### 데이터 분석 수행
- mlxtend의 apriori 클래스 사용
	- apriori() 함수로 장바구니 구매 데이터를 연관분석한다.
- association_rules() 함수를 통해 연관분석 결과에서 규칙(신뢰도 임계치 0.6 기반)을 도출
	- (Baegel, Bread)와 같이 상호연관성이 높은 구매항목들이 출력된다.
	- 신뢰도가 높다는 것은 구매 품목들의 연관성이 높음을 의미한다.
~~~python
## 6. 데이터분석 수행
# apriori 함수 적용
freq_items = apriori(result , min_support=0.2, use_colnames=True)

# association_rules로 규칙 도출(신뢰도 임계치 0.6 기반)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
rules.head()
~~~

||antecedents|consequents|antecedent support|consequent support|support|confidence|lift|leverage|conviction|
|---|---|---|---|---|---|---|---|---|---|
|0|(Cheese)|(Milk)|0.501587|0.501587|0.304762|0.607595|1.211344|0.053172|1.270148|
|1|(Milk)|(Cheese)|0.501587|0.501587|0.304762|0.607595|1.211344|0.053172|1.270148|
|2|(Bagel)|(Bread)|0.425397|0.504762|0.279365|0.656716|1.301042|0.064641|1.442650|
|3|(Eggs)|(Meat)|0.438095|0.476190|0.266667|0.608696|1.278261|0.058050|1.338624|
|4|(Eggs)|(Cheese)|0.438095|0.501587|0.298413|0.681159|1.358008|0.078670|1.563203|


## 제3유형 : 통계 (가설검정, 상관분석, 회귀분석)

### [[가설검정]]
#### 가설검정 정의 
모집단 특성에 대한 주장 또는 가설을 세우고 표본에서 얻은 정보를 이용해 가설이 옳은지 판정하는 과정을 말한다.

가설검정은 일반적으로 두 가지 가설을 설정하여 이루어집니다.
 
이를 귀무가설(H0, null hypothesis)과 대립가설(H1 또는 HA, alternative hypothesis)이라고 합니다.
귀무가설은 일반적인 상태로서 아무런 변화나 효과가 없다는 가설입니다. 대립가설은 귀무가설에 반대되는 주장이나 원하는 변화를 나타내는 가설입니다.

#### 가설검정 절차
![[데이터분석_가설검정 절차.png]]

#### 가설검정 용어 개념
- 귀무가설 : 실험을 통해 기각하고자 하는 어떤 가설로 H0으로 표시한다.
- 대립가설 : 실험을 통해 증명하고자 하는 가설이며 H1 혹은 Ha로 표시한다.
- 검정통계랑 : 가설 검정에 사용되는 표본 통계량으로 결론을 내릴때 사용하는 판단 기준이다.
- 유의수준 : 귀무가설이 참인데도 이를 잘못 기각하는 오류를 범할 확률의 최대 허용한계로 1%, 5%(0.05)를 주로 사용한다.
- 기각역 : 귀무가설을 기각하게 될 검정통계량의 영역
- 채택역 : 귀무가설을 기각할 수 없는 검정통계량의 영역
- 유의확률 : 귀무가설을 지지하는 정도를 나타낸 확률로 p-value라고도 하며, 표본으로부터 얻은 통계량 혹은 이를 치환한 검정통계량의 절대값보다 더 큰 절대값을 또다른 표본으로부터 얻을 수 있는 확률

#### 가설검정 오류
- 제1종 오류 : 귀무가설이 참일때 귀무가설을 기각하는 오류를 의미한다.
- 제2종 오류 : 귀무가설이 거짓일때 귀무가설을 채택하는 오류를 의미한다.

#### 가설검정 방법
![[데이터분석_가설검정의 방법.png]]
![[데이터분석_가설검정 종류.png]]
##### 단일표본 t 검정
###### 개념
하나의 모집단의 평균값을 특정값과 비교하는 경우 사용하는 통계적 분석 방법
- (정규성O) 단일표본 t검정(1sample t-test)
- (정규성X) 윌콕슨 부호순위 검정
###### 예시
병아리의 평균 무게가 270인지 아닌지에 대한 검정
###### 검정 순서
1. 가설설정
2. 유의수준 확인
3. 정규성 검정
4. 검정실시(통계량, p-value 확인)
5. 귀무가설 기각여부 결정(채택/기각)
###### 검정 수행 - 예시1
다음은 22명의 학생들이 국어시험에서 받은 점수이다. 학생들의 평균이 75보다 크다고 할 수 있는가?
- 귀무가설(H0): 모평균은 mu와 같다. (μ = mu), 학생들의 평균은 75이다
- 대립가설(H1): 모평균은 mu보다 크다. (μ > mu), 학생들의 평균은 75보다 크다

가정:
- 모집단은 정규분포를 따른다.
- 표본의 크기가 충분히 크다.

**검정통계량, p-value, 검정결과를 출력하시오**

~~~ python
from scipy import stats

# 데이터
scores = [75, 80, 68, 72, 77, 82, 81, 79, 70, 74, 76, 78, 81, 73, 81, 78, 75, 72, 74, 79, 78, 79]

# 모평균 가설검정
mu = 75  # 검정할 모평균
alpha = 0.05  # 유의수준

# 가설검정
result = stats.ttest_1samp(scores, mu, alternative = 'greater')

# 1. 검정통계량
print(result.statistic)

# 2. p-value
print(result.pvalue)

# 3. 유의성검정
p_value = result.pvalue
if p_value < alpha:
    print("귀무가설을 기각합니다. 모평균은 75보다 큽니다.")
else:
    print("귀무가설을 채택합니다. 모평균은 75보다 크지 않습니다.")
~~~

######  검정 수행 - 예시2
mtcars 데이터셋의 mpg열 데이터의 평균이 20과 같다고 할 수 있는지 검정하시오. (유의수준
5%)

~~~ python
import pandas as pd
import numpy as np

# 데이터 불러오기 mtcars
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/mtcars.csv")
df.head()

import scipy.stats as stats
from scipy.stats import shapiro

# 1. 가설설정
# H0 : mpg 열의 평균이 20과 같다.
# H1 : mpg 열의 평균이 20과 같지 않다.

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 정규성 검정
# H0(귀무가설) : 정규분포를 따른다.
# H1(대립가설) : 정규분포를 따르지 않는다.
statistic, pvalue = stats.shapiro(df['mpg'])
print(round(statistic,4), round(pvalue,4))
result = stats.shapiro(df['mpg'])
print(result)
~~~

~~~
0.9476 0.1229
ShapiroResult(statistic=0.9475648403167725, pvalue=0.1228824257850647)
~~~

~~~ python
# 4.1 (정규성만족 O) t-검정 실시
statistic, pvalue = stats.ttest_1samp(df['mpg'], popmean= 20, alternative='two-sided')
print(round(statistic,4), round(pvalue,4) )
# alternative (대립가설 H1) 옵션 : 'two-sided', 'greater', 'less'
~~~

~~~
0.0851 0.9328
~~~

~~~ python
# 4.2 (정규성만족 X) wilcoxon 부호순위 검정
statistic, pvalue = stats.wilcoxon(df['mpg']-20, alternative='two-sided')
print(round(statistic,4), round(pvalue,4) )
~~~

~~~
249.0 0.7891
~~~

~~~ python
# 5. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 크기 때문에(0.9328) 귀무가설을 채택한다
# 즉, mpg 열의 평균이 20과 같다고 할 수 있다.
~~~

##### 독립표본 t 검정

###### 개념
서로 독립적인 두 그룹의 평균의 차이가 0인지 알아보는 검정 방법으로, 여기서 독립이란 두 모집단에서 각각 추출된 두 표본이 서로 관계가 없다는 것을 의미한다.
- (정규성O) 독립표본 t검정(2sample t-test)
- (정규성X) 윌콕슨의 순위합 검정(ranksums)
###### 예시
남학생과 여학생의 하루 평균 운동시간이 같은지를 알아보고자 함

###### 검정 순서
1. 가설설정
2. 유의수준 확인
3. 정규성 검정 ==(주의) 두 집단 모두 정규성을 따를 경우!==
4. 등분산 검정
5. 검정실시(통계량, p-value 확인) ==(주의) 등분산여부 확인==
6. 귀무가설 기각여부 결정(채택/기각)

###### 검정 수행 - 예시1
어떤 특정 약물을 복용한 사람들의 평균 체온이 복용하지 않은 사람들의 평균 체온과 유의미하게 다른지 검정해보려고 합니다. 검정통계량, p-value, 검정결과를 출력하시오

가정:
- 약물을 복용한 그룹과 복용하지 않은 그룹의 체온 데이터가 각각 주어져 있다고 가정합니다.
- 각 그룹의 체온은 정규분포를 따른다고 가정합니다.
~~~ python
from scipy import stats

# 가설 설정
# H0: 약물을 복용한 그룹과 복용하지 않은 그룹의 평균 체온은 유의미한 차이가 없다.
# H1: 약물을 복용한 그룹과 복용하지 않은 그룹의 평균 체온은 유의미한 차이가 있다.

# 데이터 수집
group1 = [36.8, 36.7, 37.1, 36.9, 37.2, 36.8, 36.9, 37.1, 36.7, 37.1]
group2 = [36.5, 36.6, 36.3, 36.6, 36.9, 36.7, 36.7, 36.8, 36.5, 36.7]

# 가설검정
result = stats.ttest_ind(group1, group2)

# 1. 검정통계량
print(result.statistic)

# 2. p-value
print(result.pvalue)

# 3. 유의성검정
alpha = 0.05  # 유의수준 설정

if p_value < alpha:
    print("귀무가설을 기각합니다. 약물을 복용한 그룹과 복용하지 않은 그룹의 평균 체온은 유의미한 차이가 있습니다.")
else:
    print("귀무가설을 채택합니다. 약물을 복용한 그룹과 복용하지 않은 그룹의 평균 체온은 유의미한 차이가 없습니다.")
~~~

###### 검정 수행 - 예시2
A그룹의 혈압 평균이 B그룹보다 크다고 할 수 있는지 독립표본 t검정을 실시하시오. (유의수준 5%)
- A : A그룹 인원의 혈압, B : B그룹 인원의 혈압
- H0(귀무가설) : A - B ≤ 0 ( or A ≤ B)
- H1(대립가설) : A - B > 0 ( or A > B)

~~~ python
# 데이터 만들기
df = pd.DataFrame( {
'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

# 1. 가설설정
# H0 : A그룹의 혈압 평균이 B그룹보다 작거나 같다. (A <= B)
# H1 : A그룹의 혈압 평균이 B그룹보다 크다. (A > B)

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 정규성 검정 (차이값에 대해 정규성 확인)
# H0(귀무가설) : 정규분포를 따른다.
# H1(대립가설) : 정규분포를 따르지 않는다.
statisticA, pvalueA = stats.shapiro(df['A'])
statisticB, pvalueB = stats.shapiro(df['B'])
print(round(statisticA,4), round(pvalueA,4))
print(round(statisticB,4), round(pvalueB,4))
~~~

~~~
0.9314 0.3559
0.9498 0.5956
~~~

p-value 값이 유의수준(0.05) 보다 크다.
귀무가설(H0) 채택

만약 하나라도 정규분포를 따르지 않는다면 비모수 검정방법을 써야 함
(윌콕슨의 순위합 검정 ranksums)

~~~ python
# 4. 등분산성 검정
# H0(귀무가설) : 등분산 한다.
# H1(대립가설) : 등분산 하지 않는다.
statistic, pvalue = stats.bartlett(df['A'], df['B'])
print(round(statistic,4), round(pvalue,4) )
~~~

~~~
0.0279 0.8673
~~~

p-value 값이 유의수준(0.05) 보다 크다.
귀무가설(H0) 채택 => 등분산성을 따른다고 할 수 있다.

~~~ python
# 5.1 (정규성O, 등분산성 O/X) t검정
statistic, pvalue = stats.ttest_ind(df['A'], df['B'],
equal_var=True,
alternative='greater')
# 만약 등분산 하지 않으면 False로 설정

print(round(statistic,4), round(pvalue,4) )
~~~

~~~
0.8192 0.4207
~~~

~~~ python
# 5.2 (정규성X)윌콕슨의 순위합 검정
statistic, pvalue = stats.ranksums(df['A'], df['B'], alternative='two-sided')
print(round(statistic,4), round(pvalue,4) )
~~~

~~~
0.8462 0.3975
~~~

~~~ python
# 6. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 크기 때문에 귀무가설을 채택한다
# 즉, A그룹과 B그룹의 혈압 평균은 같다고 할 수 있다.

# 답 : 채택
~~~

##### 대응표본 t 검정
###### 개념
동일한 대상에 대해 두가지 관측치가 있는 경우 이를 비교하여 차이가 있는지 검정할 때 사용한다. 
- (정규성O) 대응표본(쌍체) t검정(paired t-test) 
- (정규성X) 윌콕슨 부호순위 검정(wilcoxon)
###### 예시
주로 약의 효과를 알아보거나 실험 전후의 효과를 비교하기 위해 사용한다.
- 대응표본검정을 통하여 p-value를 확인하고 유의수준에 따라 귀무가설 또는 대립가설을 채택한다.
###### 검정 순서
1. 가설설정
2. 유의수준 확인
3. 정규성 검정 ==(주의) 차이값에 대한 정규성==
4. 검정실시(통계량, p-value 확인)
5. 귀무가설 기각여부 결정(채택/기각)
###### 검정 수행 - 예시1
주어진 데이터는 고혈압 환자 치료 전후의 혈압이다. 해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오. (μ  = (치료 후 혈압 - 치료 전 혈압)의 평균))

~~~ python
import pandas as pd
from scipy import stats
df = pd.read_csv("/kaggle/input/bigdatacertificationkr/high_blood_pressure.csv")

df['diff'] = df['bp_post'] - df['bp_pre'] 

# 1. 𝜇 의 표본평균은?(소수 둘째자리까지 반올림)
print(round(df['diff'].mean(),2))

# 2. 검정통계량 값은?(소수 넷째자리까지 반올림)
result = stats.ttest_rel(df['bp_post'], df['bp_pre'],alternative='less')
print(round(result.statistic,4))

# 3. p-값은?(소수 넷째자리까지 반올림)
print(round(result.pvalue,4))

# 4. 가설검정의 결과는? (유의수준 5%)
print('귀무가설 기각, 대립가설 채택')
~~~

###### 검정 수행 - 예시2
다음은 혈압약을 먹은 전,후의 혈압 데이터이다.
혈압약을 먹기 전, 후의 차이가 있는지 쌍체 t 검정을 실시하시오 (유의수준 5%)
- before : 혈압약을 먹기 전 혈압, after : 혈압약을 먹은 후의 혈압
- H0(귀무가설) : after - before = 0
- H1(대립가설) : after - before ≠ 0

~~~ python
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro

# 데이터 만들기
df = pd.DataFrame( {
'before': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
'after' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

# 1. 가설설정
# H0 : 약을 먹기전과 먹은 후의 혈압 평균은 같다(효과가 없다)
# H1 : 약을 먹기전과 먹은 후의 혈압 평균은 같지 않다(효과가 있다)

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 정규성 검정 (차이값에 대해 정규성 확인)
statistic, pvalue = stats.shapiro(df['after']-df['before'])
print(round(statistic,4), round(pvalue,4))
~~~

~~~
0.9589 0.7363
~~~

p-value 값이 유의수준(0.05) 보다 크다.
귀무가설(H0) 채택(정규성검정의 H0 : 정규분포를 따른다)

~~~ python
# 4.1 (정규성O) 대응표본(쌍체) t검정(paired t-test)
statistic, pvalue = stats.ttest_rel(df['after'], df['before'], alternative='two-sided') # alternative='two-sided'
print(round(statistic,4), round(pvalue,4) )

# 4.2 (정규성X) wilcoxon 부호순위 검정
statistic, pvalue = stats.wilcoxon(df['after']-df['before'], alternative='two-sided')
print(round(statistic,4), round(pvalue,4) )
# alternative (대립가설 H1) 옵션 : 'two-sided', 'greater', 'less'
~~~

~~~
-3.1382 0.0086
11.0 0.0134
~~~

~~~ python
# 5. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 작기 때문에 귀무가설을 기각한다.
# 즉, 약을 먹기전과 먹은 후의 혈압 평균은 같지 않다(효과가 있다)

# 답 : 기각
~~~

##### 분산분석 (ANOVA, 일원배치법)

###### 개념
모집단 3개 이상시의 가설검정
- (정규성O) ANOVA분석
- (정규성X) 크루스칼-왈리스 검정(kruskal-wallis test)
###### 검정 순서
1. 가설설정
2. 유의수준 확인
3. 정규성 검정 ==(주의) 집단 모두 정규성을 따를 경우!==
4. 등분산 검정
5. 검정실시(통계량, p-value 확인) ==(주의) 등분산여부 확인==
6. 귀무가설 기각여부 결정(채택/기각)
###### 검정수행 - 예시1
세 가지 다른 교육 방법(A, B, C)을 사용하여 수험생들의 시험 성적을 개선시키는 효과를 평가하고자 한다. 30명의 학생들을 무작위로 세 그룹으로 배정하여 교육을 실시하였고, 시험을 보고 성적을 측정하였습니다. 다음은 각 그룹의 학생들의 성적 데이터입니다.

- 귀무가설(H0): 세 그룹(A, B, C) 간의 평균 성적 차이가 없다.
- 대립가설(H1 또는 Ha): 세 그룹(A, B, C) 간의 평균 성적 차이가 있다.

일원배치법을 수행하여 그룹 간의 평균 성적 차이가 있는지 검정하세요
- f값 (소수 둘째자리)
- p값 (소수 여섯째자리)
- 검정결과 출력

~~~ python
from scipy import stats

# 각 그룹의 데이터
groupA = [85, 92, 78, 88, 83, 90, 76, 84, 92, 87]
groupB = [79, 69, 84, 78, 79, 83, 79, 81, 86, 88]
groupC = [75, 68, 74, 65, 77, 72, 70, 73, 78, 75]

# 일원배치법 수행
result = stats.f_oneway(groupA, groupB, groupC)

# F-value
print(round(result.statistic,2))

# p-value
print(format(result.pvalue,'.6f'))
~~~

###### 검정수행 - 예시2
다음은 A, B, C 그룹 인원 성적 데이터이다.
세 그룹의 성적 평균이 같다고 할 수 있는지 ANOVA 분석을 실시하시오. (유의수준 5%)

- A, B, C : 각 그룹 인원의 성적
- H0(귀무가설) : A(평균) = B(평균) = C(평균)
- H1(대립가설) : Not H0 (적어도 하나는 같지 않다)

~~~ python
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro

# 데이터 만들기
df = pd.DataFrame( {
'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160],
'C' : [130, 120, 115, 122, 133, 144, 122, 120, 110, 134, 125, 122, 122]})

# 1. 가설설정
# H0 : 세 그룹 성적의 평균값이 같다. ( A(평균) = B(평균) = C(평균) )
# H1 : 세 그룹의 성적 평균값이 적어도 하나는 같지 않다. (not H0)

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 정규성 검정
print(stats.shapiro(df['A']) )
print(stats.shapiro(df['B']) )
print(stats.shapiro(df['C']) )

# statistic, pvalue = stats.shapiro(df['A'])
# print(round(statistic,4), round(pvalue,4))
~~~

~~~ 
ShapiroResult(statistic=0.9314376711845398, pvalue=0.35585272312164307)
ShapiroResult(statistic=0.9498201012611389, pvalue=0.5955665707588196)
ShapiroResult(statistic=0.9396706223487854, pvalue=0.45265132188796997)
~~~

세 집단 모두 p-value 값이 유의수준(0.05) 보다 크다.
귀무가설(H0) 채택 => 정규분포를 따른다고 할 수 있다.

약 하나라도 정규분포를 따르지 않는다면 비모수 검정방법을 써야 함

~~~ python
# 5.1 (정규성O, 등분산성 O) 분산분석(F_oneway)
import scipy.stats as stats
statistic, pvalue = stats.f_oneway(df['A'], df['B'], df['C'])
# 주의 : 데이터가 각각 들어가야 함(밑에 예제와 비교해볼 것)

print(round(statistic,4), round(pvalue,4) )
~~~

~~~ 
3.6971 0.0346
~~~


~~~ python
# 5.2 (정규성O, 등분산성 X) Welch-ANOVA 분석
import pingouin as pg # pingouin 패키지 미지원
pg.welch_anova(dv = "그룹변수명", between = "성적데이터", data = 데이터)
pg.welch_anova(df['A'], df['B'], df['C']) # 형태로 분석불가

# 5.3 (정규성X) 크루스칼 왈리스 검정
import scipy.stats as stats
statistic, pvalue = stats.kruskal(df['A'], df['B'], df['C'])
print(round(statistic,4), round(pvalue,4) )
~~~

~~~ 
6.897 0.0318
~~~

~~~ python
# 6. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 작기 때문에 귀무가설을 기각한다.(대립가설채택)
# 즉, A,B,C 그룹의 성적 평균이 같다고 할 수 없다.

# 답 : 기각
~~~


##### 카이제곱검정
###### 개념
- 독립성 검정 : 두 개의 범주형 변수가 서로 독립인지?
- 적합성 검정 : 각 범주에 속할 확률이 같은지?

###### 예시
- 독립성 검정 : 연령대에 따라 먹는 아이스크림 차이가 있는지 독립성 검정을 실시하시오.

	|-|딸기|초코|바닐라|
	|---|---|---|---|
	|10대|200|190|250|
	|20대|220|250|300|
	
- 적합성 검정 : 랜덤 박스에 상품 A, B, C, D가 들어있다고 한다. 랜덤 박스에서 100번 삼품을 꺼냈을때 아래와 같이 상품이 나왔다고 하면 랜덤 박스에는 상품이 동일한 비율로 들어있다고 할 수 있을까?

	|-|A|B|C|D|
	|---|---|---|---|---|
	|관측빈도|30|20|15|35|
	|기대빈도|25|25|25|25|

###### 검정 순서
1. 가설설정
2. 유의수준 확인
3. 검정실시(통계량, p-value 확인, 기대빈도 확인)
4. 귀무가설 기각여부 결정(채택/기각)

###### 검정수행 - 독립성 검정
연령대에 따라 먹는 아이스크림의 차이가 있는지 독립성 검정을 실시하시오.

~~~ python
import pandas as pd
import numpy as np

# 데이터 생성
row1, row2 = [200, 190, 250], [220, 250, 300]
df = pd.DataFrame([row1, row2], columns=['딸기','초코','바닐라'], index=['10대', '20대'])

# 1. 가설설정
# H0 : 연령대와 먹는 아이스크림의 종류는 서로 관련이 없다(두 변수는 서로 독립이다)
# H1 : 연령대와 먹는 아이스크림의 종류는 서로 관련이 있다(두 변수는 서로 독립이 아니다)

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 검정실시(통계량, p-value, 기대빈도 확인)
from scipy.stats import chi2_contingency
statistic, pvalue, dof, expected = chi2_contingency(df)
# 공식문서상에 : statistic(통계량), pvalue, dof(자유도), expected_freq(기대빈도)

# 아래와 같이 입력해도 동일한 결과값
# statistic, pvalue, dof, expected = chi2_contingency([row1, row2])
# statistic, pvalue, dof, expected = chi2_contingency([df.iloc[0],df.iloc[1]])

print(statistic)
print(pvalue)
print(dof) # 자유도 = (행-1)*(열-1)
print(np.round(expected, 2) ) # 반올림하고 싶다면 np.round()

# (참고) print(chi2_contingency(df))
~~~

~~~
1.708360126075226
0.4256320394874311
2 
[[190.64 199.72 249.65]
[229.36 240.28 300.35]]
~~~

~~~ python
# 4. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 크기 때문에 귀무가설을 채택한다.
# 즉, 연령대와 먹는 아이스크림의 종류는 서로 관련이 없다고 할 수 있다.

# 답 : 채택
~~~

~~~ python
# (Case2) 만약 데이터가 아래와 같이 주어진다면?
# (이해를 위한 참고용입니다, 빈도수 카운팅)
df = pd.DataFrame({
'아이스크림' : ['딸기','초코','바닐라','딸기','초코','바닐라'],
'연령' : ['10대','10대','10대','20대','20대','20대']})

# pd.crosstab(index, columns)
pd.crosstab(df['연령'], df['아이스크림'])
~~~

###### 검정수행 - 적합성 검정
랜덤 박스에 상품 A,B,C,D가 들어있다.
다음은 랜덤박스에서 100번 상품을 꺼냈을 때의 상품 데이터라고 할 때
상품이 동일한 비율로 들어있다고 할 수 있는지 검정해보시오.

~~~ python
import pandas as pd
import numpy as np

# 데이터 생성
row1 = [30, 20, 15, 35]
df = pd.DataFrame([row1], columns=['A','B','C', 'D'])

# 1. 가설설정
# H0 : 랜덤박스에 상품 A,B,C,D가 동일한 비율로 들어있다.
# H1 : 랜덤박스에 상품 A,B,C,D가 동일한 비율로 들어있지 않다.

# 2. 유의수준 확인 : 유의수준 5%로 확인

# 3. 검정실시(통계량, p-value)
from scipy.stats import chisquare
# chisquare(f_obs=f_obs, f_exp=f_exp) # 관측빈도, 기대빈도

# 관측빈도와 기대빈도 구하기
f_obs = [30, 20, 15, 35]
# f_obs = df.iloc[0]
f_exp = [25, 25, 25, 25]

statistic, pvalue = chisquare(f_obs=f_obs, f_exp=f_exp)
print(statistic)
print(pvalue)
# 자유도는 n-1 = 3
~~~

~~~
10.0
0.01856613546304325
~~~

~~~ python
# 4. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 작기 때문에 귀무가설을 기각한다.
# 즉, 랜덤박스에 상품 A,B,C,D가 동일한 비율로 들어있지 않다고 할 수 있다.

# 답 : 기각
~~~


### 상관분석

#### 상관분석 정의
두 변수간의 선형관계를 분석하는 기법
#### 상관분석 방법
##### 피어슨 상관계수
###### 정의
- -1 <= r <= 1 값을 가짐 (연속형 데이터 사용)
- 절대값이 1에 가까울수록 강한 선형관계를 가짐
###### 검정 수행

~~~ Python
# 데이터 불러오기
import pandas as pd
import numpy as np

# 실기 시험 데이터셋으로 셋팅하기 (수정금지)
from sklearn.datasets import load_diabetes

# diabetes 데이터셋 로드
diabetes = load_diabetes()
x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.DataFrame(diabetes.target)
y.columns = ['target']

# 상관분석을 할 2가지 변수 설정
x = x['bmi']
y = y['target']
print(x.head())
print(y.head())
~~~

~~~
0 0.061696
1 -0.051474
2 0.044451
3 -0.011595
4 -0.036385
Name: bmi, dtype: float64

0 151.0
1 75.0
2 141.0
3 206.0
4 135.0
Name: target, dtype: float64
~~~

~~~ Python
# 라이브러리 불러오기
from scipy.stats import pearsonr

# 상관계수에 대한 검정실시
r, pvalue = pearsonr(x, y)

# 가설설정
# H0 : 두 변수간 선형관계가 존재하지 않는다 (􀀁 = 0)
# H1 : 두 변수간 선형관계가 존재한다 (􀀁 ≠ 0)

# 1. 상관계수
print(round(r, 2) )

# 2. p-value
print(round(pvalue, 2))

# 3. 검정통계량
# 통계량은 별도로 구해야 함 (T = r * root(n-2) / root(1-r2) )
# r = 상관계수
# n = 데이터의 개수

n = len(x) # 데이터 수
r2 = r**2 # 상관계수의 제곱
statistic = r * ((n-2)**0.5) / ((1-r2)**0.5)

print(round(statistic, 2))

# 4. 귀무가설 기각여부 결정(채택/기각)
# p-value 값이 0.05보다 작기 때문에 귀무가설을 기각한다.(대립가설채택)
# 즉, 두 변수간 선형관계가 존재한다고 할 수 있다.(상관계수가 0이 아니다)

# 답 : 기각
~~~

~~~
0.59
0.0
15.19
~~~


### 회귀분석

#### 회귀분석 방법
##### [[다중 선형 회귀분석|다중선형회귀]](LinearRegression)

###### 정의
독립변수(X)가 종속변수(Y)에 어떻게 영향을 주는지 식으로 표현한 것
특징은 식을 보고 설명이 가능하며, 가장 적은 수의 X로 Y를 잘 예측하는 것이 Best이다.

- 결정계수(R<sup>2</sup>) : 0~1 사이 값을 가지며, 전체 변동에서 회귀식이 설명가능한 변동 비율을 말함
###### 예시
여름 평균 온도(X)와 에어컨 판매 수(Y)의 영향을 식으로 구하라. 

###### 검정 수행
당뇨병 환자의 질병 진행정도 
~~~ python
# 데이터 불러오기
import pandas as pd
import numpy as np
# 실기 시험 데이터셋으로 셋팅하기 (수정금지)
from sklearn.datasets import load_diabetes
# diabetes 데이터셋 로드
diabetes = load_diabetes()
x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.DataFrame(diabetes.target)
y.columns = ['target']

# sklearn 라이브러리 활용
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 독립변수와 종속변수 설정
x = x[ ['age','sex','bmi'] ]
print(x.head())
print(y.head())
~~~

~~~
age sex bmi
0 0.038076 0.050680 0.061696
1 -0.001882 -0.044642 -0.051474
2 0.085299 0.050680 0.044451
3 -0.089063 -0.044642 -0.011595
4 0.005383 -0.044642 -0.036385

target
0 151.0
1 75.0
2 141.0
3 206.0
4 135.0
~~~

회귀식 : y = b0 + b1x1 + b2x2 + b3x3
(x1=age, x2=sex, x3=bmi)

~~~ python
# 모델링
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

# 회귀분석 관련 지표 출력
# 1. Rsq(결정계수) : model.score(x, y)
model.score(x, y)
print(round(model.score(x, y), 2) )
~~~

~~~
0.35
~~~

~~~ Python
# 2. 회귀계수 출력 : model.coef_
print(np.round(model.coef_, 2) ) # 전체 회귀계수
print(np.round(model.coef_[0,0], 2) ) # x1 의 회귀계수
print(np.round(model.coef_[0,1], 2) ) # x2 의 회귀계수
print(np.round(model.coef_[0,2], 2) ) # x3 의 회귀계수
~~~

~~~
[[138.9 -36.14 926.91]]
138.9
-36.14
926.91
~~~

~~~ Python
# 3. 회귀계수(절편) : model.intercept_
print(np.round(model.intercept_, 2) )
~~~

~~~
[152.13]
~~~

회귀식 : y = b0 + b1x1 + b2x2 + b3x3
(x1=age, x2=sex, x3=bmi) ### 결과 : y = 152.13 + 138.9age - 36.14sex + 926.91bmi

##### [[로지스틱 회귀분석|로지스틱 회귀]] (LogisticRegression)

###### 정의
독립변수(X)를 가지고 Y=1일 확률이 얼마인지 예측하는 모형
(확률이 0,5보다 크면 1로, 0.5보다 작으면 0으로 분류)

- Odds : 성공할 확률 / 실패할 확률
	- 예시 : 성공할 확률 0.75, 실패할 확률 0.25 이면, 성공할 확률이 실패할 확률보다 3배 크다. (odds = 3)
- Logit : Odds에 자연로그를 취한 값

###### 검정 수행 

~~~ python
# 데이터 불러오기
import pandas as pd 
import numpy as np 

# Seaborn 의 내장 타이타닉 데이터셋을 불러옵니다. 
import seaborn as sns 
df = sns.load_dataset('titanic')

print(df.head())
~~~

~~~
survived pclass sex age sibsp parch fare embarked class \ 
0 0 3 male 22.0 1 0 7.2500 S Third 
1 1 1 female 38.0 1 0 71.2833 C First 
2 1 3 female 26.0 0 0 7.9250 S Third 
3 1 1 female 35.0 1 0 53.1000 S First 
4 0 3 male 35.0 0 0 8.0500 S Third 

who adult_male deck embark_town alive alone 
0 man True NaN Southampton no False 
1 woman False C Cherbourg yes False 
2 woman False NaN Southampton yes True 
3 woman False C Southampton yes False 
4 man True NaN Southampton no True
~~~

 회귀식 : P(1일 확률) = 1 / ( 1+exp(-f(x)) ) 
 - f(x) = b0 + b1x1 + b2x2 + b3x3 
 - ln(P/1-P) = b0 + b1x1 + b2x2 + b3x3 
   (P=생존할 확률, x1=sex, x2=sibsp, x3=fare)

~~~ python
# 데이터 전처리 
# 변수처리 
# 문자형 타입의 데이터의 경우 숫자로 변경해준다.
# *** 실제 시험에서 지시사항을 따를 것 *** 

# 성별을 map 함수를 활용해서 각각 1 과 0 에 할당한다.( 여성을 1, 남성을 0) 
# ( 실제 시험의 지시 조건에 따를 것) 
df['sex'] = df['sex'].map({'female': 1, 'male': 0 }) 

print(df.head())
~~~

~~~
survived sex sibsp fare 
0 0 0 1 7.2500 
1 1 1 1 71.2833 
2 1 1 0 7.9250 
3 1 1 1 53.1000 
4 0 0 0 8.0500
~~~

(주의) LogisticRegression() 객체안에 반드시 penalty= None 으로 입력해야 함

~~~ python
# 독립변수와 종속변수 설정 
x = df.drop(['survived'], axis=1) # x = df [ ['sex','age','fare'] ] 
y = df['survived']

# 모델링 
from sklearn.linear_model import LogisticRegression # 회귀는 LinearRegression 

# 반드시 penalty= None 으로 입력할 것해야 함, default='l2' 
model1 = LogisticRegression(penalty= None) 
model1.fit(x, y)

# 로지스틱회귀분석 관련 지표 출력 
# 1. 회귀계수 출력 : model.coef_ 
print(np.round(model1.coef_, 4) ) # 전체 회귀계수 
print(np.round(model1.coef_[0,0], 4) ) # x1 의 회귀계수 
print(np.round(model1.coef_[0,1], 4) ) # x2 의 회귀계수 
print(np.round(model1.coef_[0,2], 4) ) # x3 의 회귀계수 

# 2. 회귀계수( 절편) : model.intercept_ 
print(np.round(model1.intercept_, 4) 
~~~

~~~
[[ 2.5668 -0.4017 0.0138]] 
2.5668 
-0.4017 
0.0138 

[-1.6964]
~~~

회귀식 : P(1일 확률) = 1 / ( 1+exp(-f(x)) ) 
- f(x) = b0 + b1x1 + b2x2 + b3x3 
- ln(P/1-P) = b0 + b1x1 + b2x2 + b3x3 
  (P=생존할 확률, x1=sex, x2=sibsp, x3=fare)

결과 : ln(P/1-P) = -1.6964 + 2.5668sex - 0.4017sibsp + 0.0138fare

~~~ python
# 3-1. 로지스틱 회귀모형에서 sibsp 변수가 한단위 증가할 때 생존할 오즈가 몇 배 증가하는지 
# 반올림하여 소수점 셋째 자리까지 구하시오. 

# exp(b2) 를 구하면 된다. 
result = np.exp(model1.coef_[0,1]) # 인덱싱 주의하세요. 
print(round(result, 3)) 

# 해석 : sibsp 변수가 한 단위 증가할 때 생존할 오즈가 0.669 배 증가한다. 
# 생존할 오즈가 33% 감소한다. ( 생존할 확률이 감소한다)
~~~

~~~
0.669
~~~

~~~ python
# 3-2. 로지스틱 회귀모형에서 여성일 경우 남성에 비해 오즈가 몇 배 증가하는지 
# 반올림하여 소수점 셋째 자리까지 구하시오. 

# exp(b1) 를 구하면 된다. 
result2 = np.exp(model1.coef_[0,0]) # 인덱싱 주의하세요. 
print(round(result2, 3)) 

# 해석 : 여성일 경우 남성에 비해 생존할 오즈가 13.024 배 증가한다. 
# 생존할 오즈가 13 배 증가한다. ( 생존할 확률이 증가한다)
~~~

~~~
13.024
~~~




***
# 출처
빅데이터분석기사 실기 (영진닷컴)

***
# 관련 노트
[[MOC_빅데이터 분석 이론]]
빅데이터 분석기사 실기 노트
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

***
# 외부 링크
제3유형 이론 유투브강의 : https://www.youtube.com/watch?v=37GqFZVjc1Y
[제1유형 예제 풀이](https://colab.research.google.com/drive/15h-ZaRjR4JBKWqpeOLO0uQj5j1EtXxz3?hl=ko#scrollTo=oHVIZMGItMjn)
[제2유형 예제 풀이](https://colab.research.google.com/drive/1vgnbs9qIJ8c_qfe2xxS5RUO4uLY1bBCs?hl=ko)
[제3유형 예제 풀이](https://colab.research.google.com/drive/1py2pikeI09QZM2ulUi4p0uvd8V-UZ0OI?hl=ko)