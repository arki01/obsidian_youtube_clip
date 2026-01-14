---
Created: 2024-08-25 14:38
tags: 
share_link: https://share.note.sx/s1ol20bx#CsXYRbACM+z8kZQH4bR80dIRJMB+DhK8GN94UdVx+2A
share_updated: 2024-08-26T23:19:04+09:00
---

# 개요
SKADA 인증 Practitioner 평가에 대해 필요한 역량들을 기준으로 출제범위에 해당되는 데이터분석 이론들로 구성된 내용들이다.
기본적인 AI Math에 해당되는 이론들과 함께, 데이터 분석 수행에 있어 데이터처리>모델링>평가>최적화 순의 이론들을 정리 하였다. 

### 출제범위
![[MOC_SKADA 인증 Practitioner 출제범위.png]]

# 내용
## 개론 (AI Math)
### 기술통계량
#### 대푯값
- 대략적인 분포 위치, 대표적인 값을 정량화하기 위해 사용하는 통계량
##### 평균값 (Mean)
- 표본의 평균값은 표본에서 얻었다는 점에서 '표본평균'이라고도 함
~~~ python
df['가격'].mean()
~~~
##### 중앙값 (Median)
- 크기순으로 값을 정렬했을때 한가운데 위치한 값
- 표본크기가 짝수일 경우 가운데 두값의 평균값
~~~ python
df['가격'].median()
~~~
##### 최빈값(Mode)
- 데이터 중 자주 나타나는 값
~~~ python
df['가격'].mode()
~~~

#### 산포도
- 데이터 분포의 폭
- 즉 '어느 정도 퍼져 있는지'를 파악
##### 분산 (Variance)
- 표본의 각 값과 표본평균이 어느 정도 떨어져 있는지를 표시하는 통계량
~~~ python
df['가격'].var()
~~~
##### 표준편차 (Starndard Devation)
- 표본분산에 제곱근을 취한 값
~~~ python
df['가격'].std()
~~~

##### [[이상값 처리|이상값 (Outlier)]]
- 극단적으로 큰 값이나 작은 값
- 데이터 전처리 과정 중 하나로 이상값(이상치)는 정상의 범주에서 벗어난 값을 의미
###### 검출 방법
- 분산 : 정규분포의 97.5% 이상, 2.5%미만을 이상값이라 할 수 있다.
- 우도 : 관측치가 가장 많이 발견될 것으로 보이는 경위의 확률값을 '우도'라고 하며, 우도가 낮을 수록 이상값에 가깝다고 할 수 있다.
- 근접 이웃 기반 이상치 탐지(NN, [[KNN (K-Neareat Neighbor)|KNN]]) : 정상값의 중심으로 부터 거리가 미리 정해진 임곗값보다 큰 경우 이상치로 본다.
- 사분위수(IQR) : 일반적으로 사분범위에서 1.5 사분범위수(IQR)를 벗어나는 경우를 이상치로 판단한다. (Q1-1.5IQR, Q3+1.5IQR)
  ![[빅데이터 분석_IQR(Inter Quantile Range).png|500]]
###### 처리방법
- 삭제, 대체, 스케일링, [[정규화 (Normalization)|정규화]] 등의 방법이 있다.
  
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

# 이상치 처리
lowerOutlier = df.loc[(df["값"] < (Q1 - IQR * 1.5))]
upperOutlier = df.loc[(df["값"] > (Q3 + IQR * 1.5))]
~~~

##### 변동계수 (Coefficient of Variation, CV)
- 표준 편차와 평균의 비율로 데이터의 상대적 변동성을 측정하는 지표이다.
- 주로 서로 다른 단위나 크기의 데이터를 비교할 때 유용하게 사용된다.
- CV = 표준 편차/평균×100%
### 추론통계
- 표본으로부터 전체 모집단의 모수에 대해 추정하는 통계적 추론 방식
#### 점추정
- 모집단의 모수를 하나의 값을 추정하는 것이다.
#### 구간추정
- 모수가 포함될 것으로 기대되는 구간을 추정하는 것으로, 점추정과 달리 신뢰성 정도를 포함한다.
	- 90% 신뢰구간 z=1.645
	- 95% 신뢰구간 z=1.96
	- 99% 신뢰구간 z=2.576
#### 신뢰구간
- 표본평균으로 모집단평균을 추정할 때 모평균이 포함될 것으로 예상되는 범위
- 95% 신뢰구간을 예를 들면
	- 모집단에서 샘플을 100번 뽑았을 때, 그 100개의 샘플 중 95개가 모수를 포함하게 될 확률
	- 같은 모형에서 반복해서 표본을 얻고, 신뢰구간을 얻을  때, 신뢰구간이 참 모수값을 포함할 확률이 95%가 되도록 만들어진 구간
#### [[가설검정]]
- 모집단 특성에 대한 주장 또는 가설을 세우고 표본에서 얻은 정보를 이용해 가설이 옳은지 판정하는 과정을 말한다.
##### 수행 절차

![[데이터분석_가설검정 절차.png]]

##### 용어 개념
- 귀무가설 : 실험을 통해 기각하고자 하는 어떤 가설로 H0으로 표시한다.
- 대립가설 : 실험을 통해 증명하고자 하는 가설이며 H1 혹은 Ha로 표시한다.
- 검정통계랑 : 가설 검정에 사용되는 표본 통계량으로 결론을 내릴때 사용하는 판단 기준이다.
- 유의수준 : 귀무가설이 참인데도 이를 잘못 기각하는 오류를 범할 확률의 최대 허용한계로 1%, 5%(0.05)를 주로 사용한다.
- 기각역 : 귀무가설을 기각하게 될 검정통계량의 영역
- 채택역 : 귀무가설을 기각할 수 없는 검정통계량의 영역
- 유의확률 : 귀무가설을 지지하는 정도를 나타낸 확률로 p-value라고도 하며, 표본으로부터 얻은 통계량 혹은 이를 치환한 검정통계량의 절대값보다 더 큰 절대값을 또다른 표본으로부터 얻을 수 있는 확률
#### [[혼동행렬(Confusion Matrix)]]
- 이진 분류 모형이 예측한 값과 실제 값의 조합을 교차표 형태로 정리한 행렬
- 이 행렬은 네 가지 지표로 구성된다 : True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
![[데이터분석 평가지표_혼동행렬_개념.png]]
##### 수행 방법
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

##### 성능 지표
- 정확도(Accuracy): 전체 예측 중에서 올바르게 예측한 비율.
- 정밀도(Precision): Positive로 예측한 것 중 실제로 Positive인 비율.
- 재현율(Recall) 또는 민감도(Sensitivity): 실제 Positive인 것 중에서 모델이 Positive로 올바르게 예측한 비율.
- F1 점수(F1 Score): 정밀도와 재현율의 조화 평균으로, 두 지표의 균형을 평가.
  ![[데이터분석 평가지표_혼동행렬_평가지표.png]]

### 확률분포
- 가로축에 확률변수, 세로축에 그확률 변수의 발생 가능성을 표시한 분포
- 현실 세계의 모집단을 수학 세계의 확률분포로 가정
#### 이산 확률분포
- 값을 개수가 셀수 있는(가산) 개가 가지는 확률 분포 (예: 동전 던지기, 주사위 던지기)
- 베르누이 분포, 이항 분포, 포아송 분포, 기하 분포, 초기하 분포, 음이항 분포, 다항 분포
##### 베르누이 분포 (Bernoulli Distribution)
- 1회 실행결과가 1과 0의 2가지 이진값으로만 이루어진 분포를 의미하며, 1과 0을 각각 동전의 앞면과 뒷면으로 가정한다면 동전던지기는 베르누이 분포라 할 수 있다.
- 이진분류 로지스틱리그레션로 베르누이분포를 가정하고 설명할 수 있다.
#### 연속 확률분포
- 확률 밀도 함수를 이용해 분포를 표현할 수 있는 경우 (예: 키나 몸무게 등 연속형 변수를 가질 때)
- 정규 분포, t 분포, 유니폼 분포, 카이제곱 분포, 감마 분포, 지수 분포, f 분포
##### 가우시안 분포 (Gaussian Distribution)
- 정규 분포(Normal Distribution) 라고도 불린다.
- 연속 확률 분포의 하나로, 데이터가 평균을 중심으로 대칭적으로 분포하는 경우를 모델링한다.
- 표준 정규 분포(Standard Normal Distribution) 는 평균이 0이고 표준 편차가 1인 특별한 경우의 정규 분포이다.

##### t-분포
- t-분포는 표본 크기가 작거나 모분산이 알려지지 않은 상황에서 주로 사용되는 분포입니다. 
- 자유도가 낮을수록 꼬리가 두꺼워지고, 자유도가 높아지면 정규분포에 가까워집니다. 
- t-분포는 신뢰구간 계산 및 가설 검정에 널리 사용됩니다.
#### 중심극한정리
- 모집단의 분포와 상관없이, 일정한 크기(n=30)이상의 표본평균 분포는 정규분포를 이룬다라는 개념.
- 임의의 분포를 가진 독립적인 확률 변수들의 표본 평균이, 표본 크기 n이 충분히 크다면, 그 분포는 원래의 분포 형태와 상관없이 정규 분포(가우시안 분포)에 가까워진다는 것

## 데이터 처리와 시각화
### Data Ingestion, EDA
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

#### 상관관계 (Correlation)
##### 공분산
- 두 변수가 서로 비례 또는 반비례하는 지에 대한 부합 정도를 나타내는 값이라 할 수 있음
- 두 편차의 값이 둘다 양수이거나 음수이면 그 곱은 양수이며, 편차의 방향은 같다라고 말할 수 있다. 이 곱을 '교차곱편차' 라고 한다.
- 이 교차곱편차를 n-1로 나눈 값을 공분산이라고 한다.
##### 상관계수 (Correlation Coefficient)
- 표준화된 공분산을 상관계수라고 한다. (범위는 -1~1 사이의 값이다.)
- 피어슨 상관계수 또는 피어슨 r 이라고도 한다.
- r = 1 : 두 변수의 상관관계는 완전한 양의 상관관계이다.
- r = 0 : 두 변수 간의 아무런 상관관계도 없다.
- r = -1 : 두 변수의 상관관계는 완전한 음의 상관관계이다.

###### 분석 수행
~~~python
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
~~~
###### 검정 수행

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
# r^2 = 결정계수
# n = 데이터의 개수

n = len(x) # 데이터 수
r2 = r**2 # 상관계수의 제곱 (경정계수)
statistic = r * ((n-2)**0.5) / ((1-r2)**0.5)  # 검정통계량
 
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


##### 결정계수
- 상관계수를 제곱한 $R^2$ 을 결정계수라 한다.

### Data Processing
#### 개요
###### 필요성 
- 수집한 데이터를 탐색해보면 빠지거나 틀린 값, 단위가 다를 수 있다.
- 목적에 맞게 데이터를 재가공하여 분석하기 좋게 데이터 클리닝 작업도 진행한다.
###### 유형
- 데이터 정제
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

#### 데이터 처리 (기초)
##### 병합

###### merge
- how 파라미터는 조인의 유형을 지정하며, 'inner', 'left', 'right', 'outer' 등의 값을 가집니다.
~~~~ python
### basic1 데이터와 basic3 데이터를 'f4'값을 기준으로 병합

import pandas as pd
b1 = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
b3 = pd.read_csv("../input/bigdatacertificationkr/basic3.csv")

df = pd.merge(left = b1 , right = b3, how = "left", on = "f4")
~~~~

##### 분할 (Slicing)

###### loc
- 인덱스 명 기준
~~~ python
df.loc[2, '메뉴':'가격']
~~~
###### iloc
- 인덱스 번호 기준
~~~ python
df.iloc[2, 0:2]
~~~
###### qcut
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

###### cut
~~~ python
import pandas as pd

# 임의의 데이터 생성
data = {
    '학생': ['학생1', '학생2', '학생3', '학생4', '학생5'],
    '영어점수': [95, 88, 76, 65, 90]
}

# DataFrame 생성
df = pd.DataFrame(data)

# 등급과 점수 범위 설정
grades = ['4등급', '3등급', '2등급', '1등급']
cut_bins = [0, 69, 79, 89, 100]

# 영어점수를 기준으로 영어등급 부여
df['영어등급'] = pd.cut(df['영어점수'], bins=cut_bins, labels=grades)

# 결과 출력
print(df)
~~~

~~~
   학생  영어점수 영어등급
0  학생1    95  1등급
1  학생2    88  2등급
2  학생3    76  3등급
3  학생4    65  4등급
4  학생5    90  1등급
~~~
#####  요약

###### groupby

~~~ python
df.groupby(['원두', '할인율'])['가격'].mean()
~~~

~~~
원두    할인율
과테말라  0.2    4600.0
콜롬비아  0.0    5000.0
      0.5    4100.0
한국    0.0    4100.0
~~~
###### aggregation

~~~ python
import pandas as pd

# 예제 데이터 생성
data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'Sales': [100, 150, 200, 250, 300, 350, 400],
    'Quantity': [10, 15, 20, 25, 30, 35, 40]
})

# 카테고리별 집계
aggregation = data.groupby('Category').agg({
    'Sales': ['sum', 'mean', 'max', 'min'],
    'Quantity': ['sum', 'mean']
})

print(aggregation)
~~~

~~~
         Sales                  Quantity      
           sum   mean  max  min      sum  mean
Category                                      
A          250  125.0  150  100       25  12.5
B          450  225.0  250  200       45  22.5
C         1050  350.0  400  300      105  35.0
~~~

###### pivot
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
~~~

###### pivot table
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

##### 시계열 데이터
###### Datetime
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
###### Timedelta
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

#### 특성 변환 (Feature Engineering)

##### [[표준화 (Standardization)]]
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


##### [[정규화 (Normalization)]]
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

##### [[정규분포변환|로그 변환]]
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
- 예시 문제
~~~ python
import pandas as pd
import numpy as np
from scipy.stats import skew

# 데이터 생성
np.random.seed(42)
data = {
    f'feature_{i}': np.random.normal(loc=0, scale=1, size=100) for i in range(1, 11)
}

# 일부 feature에 skewness를 추가하기 위해 비정상적인 값을 추가
data['feature_1'] = np.random.exponential(scale=2, size=100)  # 양의 skewness
data['feature_2'] = np.random.normal(loc=5, scale=0.5, size=100)  # 약간의 음의 skewness
data['feature_3'] = np.concatenate([np.random.normal(loc=0, scale=1, size=95), np.random.normal(loc=10, scale=1, size=5)])  # 강한 양의 skewness
data['feature_4'] = np.concatenate([np.random.normal(loc=-10, scale=0.5, size=95), np.random.normal(loc=0, scale=1, size=5)])  # 강한 음의 skewness

# 비정상적인 데이터를 추가하여 skewness 절대값이 10을 넘는 feature 생성
data['feature_5'] = np.concatenate([np.random.normal(loc=0, scale=1, size=90), np.random.normal(loc=50, scale=1, size=10)])  # 매우 큰 양의 skewness

# 데이터프레임 생성
x_high_corr = pd.DataFrame(data)

# 조건1. x_high_corr 에서 skeness의 절댓값이 1을 초과하는 feature를 찾으시오.
skewness_values = x_high_corr.skew()
hige_skewness_features = skewness_values[abs(skewness_values)>1].index

# 조건2. 찾은 feature 각각에 1e-10을 더한 후 자연 log 변환을 적용하시오.
for feature in hige_skewness_features:
  x_high_corr[feature] = np.log(x_high_corr[feature] + 1e-10)

# 조건3. 변환된 데이터를 x_log 변수에 저장하시오.
x_log = x_high_corr.copy()

skew_comparison = pd.concat([skewness_values, x_log.skew()], axis=1)
print(skew_comparison)
~~~

~~~
0	1
feature_1	1.751433	-0.558720
feature_2	-0.269298	-0.269298
feature_3	3.213662	0.025040
feature_4	4.030479	NaN
feature_5	2.690219	0.812392
feature_6	-0.034849	-0.034849
feature_7	-0.022612	-0.022612
feature_8	0.145468	0.145468
feature_9	0.397603	0.397603
feature_10	-0.164581	-0.164581
~~~


##### 범주화
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

#### 특성 추출 (Feature Extaction)

##### PCA(Principal Component Anaysis, 주성분 분석)
###### 정의
- 주성분분석이란?
	- 여러 변수들의 변량을 '주성분'이라고 불리는, 서로 상관성이 높은 여러 변수들의 선형 조합으로 새로운 변수들로 요약, 축소하는 기법이다. 
	- ![[빅데이터 주성분 분석(PCA)을 통한 차원축소.png]]
###### 수행 방법
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


#### 데이터 정제
##### [[결측값 처리|결측치 처리]]
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

##### [[이상값 처리|이상치 처리]]
###### IQR (사분위범위) 방법
- 수행예시-1
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
~~~

- 수행예시-2
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 원본 데이터프레임
data = {
    'Feature1': [10, 12, 14, 15, 16, 18, 100, 22, 25, 30],
    'Feature2': [5, 7, 8, 10, 11, 13, 15, 20, 21, 100]
}
df = pd.DataFrame(data)


# 1. 박스플롯 생성
df.boxplot()
plt.show()

# 2. IQR 계산 및 이상치 처리
df_no_outlier = df.copy()
for col in df.columns:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1

  lower = Q1 - 1.5*IQR
  upper = Q3 + 1.5*IQR

  df_no_outlier[col] = df[col].apply(lambda x: np.nan if (x<lower) or (x>upper) else x)


# 3. 박스플롯 결과 출력
df_no_outlier.boxplot()
plt.title('No Outlier')
plt.show()
~~~

#### 범주형 특성 인코딩 (Encoding Categorical Features)

##### 라벨 인코딩 (Label Encoding)
- 카테고리형 데이터(Categorical Data)를 수치형 데이터(Numerical Data)로 변환해주는 전처리 작업
~~~ python
from sklearn.preprocessing import LabelEncoder

cols = list(X_train.columns[X_train.dtypes == object])  # object 컬럼명 추출

for col in cols:
    le = LabelEncoder()
    c_train[col] = le.fit_transform(c_train[col])
    c_test[col] = le.transform(c_test[col])
~~~
##### 원-핫 인코딩 (One-Hot Encoding)
- 설명 : pandas를 통해 아예 데이터프레임 형식으로 반환받는 것이다. 각 column 이름에 변수 특성을 명시해줘서 그냥 array 타입보다는 훨씬 보기 편하다.
~~~ python
c_train = pd.get_dummies(c_train[cols])
c_test = pd.get_dummies(c_test[cols])
~~~

### Data Visualization
#### 시간 시각화
- 막대그래프, 산점도, 선그래프, 계단식 그래프, 영역차트 등으로 표현이 가능하다.
##### 선 그래프 (Line Plot)
- 선 그래프는 데이터 포인트를 선으로 연결하여 시간의 흐름에 따른 데이터의 변화를 시각화합니다.

~~~ python
import matplotlib.pyplot as plt

# 예제 데이터
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [200, 220, 250, 275, 300]

# 선 그래프 그리기
plt.plot(months, sales, marker='o', linestyle='-', color='b')
plt.figure(figsize=(10, 6))
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
~~~

![[데이터분석_시각화_Line Plot.png|500]]

##### 막대 그래프 (Bar Plot)
~~~ python
plt.bar(products, sales, color='skyblue')
~~~

![[데이터분석_시각화_Bar Plot.png|500]]


#### 공간 시각화
- 등치 지역도, 도트 플롯맵, 버블 플롯맵, 등치선도, 입자흐름도, 카토그램 등이 있다.
#### 분포 시각화
- 데이터의 최대값, 최소값, 전체 분포 등을 데이터가 차지하는 영역을 기준으로 시각화 한 것
- 파이차트, 도넛차트, 트리맵, 상자그림, 누적 막대그래프, 누적 영역 차트 등이 있다.

##### 파이 차트 (Pie Chart)
- 파이 차트는 전체를 구성하는 각 부분의 비율을 시각화합니다. 원형 그래프의 각 조각이 비율을 나타내며, 비율 분석에 유용합니다.
~~~ python
# 예제 데이터
departments = ['Marketing', 'Sales', 'R&D', 'HR']
budget = [30000, 40000, 25000, 15000]

# 파이 차트 그리기
plt.figure(figsize=(8, 8))
plt.pie(budget, labels=departments, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Department Budget Distribution')
plt.show()
~~~
![[데이터분석_시각화_산점도_파이차트.png|500]]


##### 상자 그림 (Box Plot)
- 상자 그림은 데이터의 분포를 요약하여 시각화합니다. 중앙값, 사분위수, 이상치 등을 나타내며, 데이터의 변동성과 극단값을 파악하는 데 유용합니다.

~~~ python 
# 예제 데이터
np.random.seed(0)
group1 = np.random.normal(70, 10, 100)
group2 = np.random.normal(80, 15, 100)
group3 = np.random.normal(90, 20, 100)

# 상자 그림 그리기
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Score Distribution by Group')
plt.xlabel('Group')
plt.ylabel('Score')
plt.show()
~~~

![[데이터분석_시각화_Box Plot.png|500]]


#### 관계 시각화
- 데이터 변수 사이의 연관성이나 분포, 패턴을 찾는 시각화 방법을 말한다.
- 산점도 행렬, 버블 차트, 히스토그램 등이 있다.

##### 산점도 (Scatter Plot)
- 산점도는 두 변수 간의 관계를 시각화합니다. 각 데이터 포인트를 좌표 평면에 점으로 표시하여 변수 간의 상관관계를 파악할 수 있습니다.
~~~ python
# 예제 데이터
np.random.seed(0)
x = np.random.rand(100)
y = 2.5 * x + np.random.normal(0, 0.5, 100)

# 산점도 그리기
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, edgecolor='k', color='purple')
plt.title('Scatter Plot of X vs. Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
~~~

![[데이터분석_시각화_산점도.png|500]]



##### 히스토그램
- 히스토그램은 데이터의 분포를 시각화합니다. 데이터의 구간을 나누어 각 구간에 속하는 데이터의 개수를 막대로 표시합니다.
~~~ python
import numpy as np

# 예제 데이터
np.random.seed(0)
scores = np.random.normal(70, 10, 100)  # 평균 70, 표준편차 10

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=10, edgecolor='black', color='lightgreen')
plt.title('Distribution of Exam Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
~~~

![[데이터분석_시각화_히스토그램.png|500]]

#### 비교 시각화
- 다변량 변수를 갖는 자료를 2차원에 효과적으로 표현하여 데이터 간의 차이점 뿐만 아니라 유사성 관계도 확인하는 시각화 방법
- 플로팅바, 간트차트, 히트맵, 평행좌표계, 스타차트 등이 있다.

##### 히트맵 (Heatmap)
- 히트맵은 데이터의 상관관계나 패턴을 색상으로 시각화합니다. 주로 행과 열이 있는 데이터의 값을 색상으로 표현하여 데이터의 패턴을 파악합니다.

~~~ python
import seaborn as sns

# 예제 데이터
data = np.random.rand(10, 12)
heatmap_data = pd.DataFrame(data, columns=[f'Month {i+1}' for i in range(12)], index=[f'Year {i+1}' for i in range(10)])

# 히트맵 그리기
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('Heatmap of Monthly Data')
plt.show()
~~~

![[데이터분석_시각화_히트맵.png|500]]

#### 인포그래픽
- 인포(Information)+그래픽(Graphic)의 합성어로 다양하고 복잡한 데이터를 한눈에 볼 수 있게 표현한 것을 말한다.

## 모델링

### Baseline Modeling
#### 데이터 준비
##### Train/Validation 나누기
~~~ python
# 라이브러리 및 데이터 불러오기
import pandas as pd
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")

# 데이터 전처리 이후 

# 학습용 데이터와 검증용 데이터로 구분
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2022)
~~~


#### 지도학습 - 분류분석

##### [[로지스틱 회귀분석|로지스틱 회귀]]
###### 정의
- 독립변수의 선형결합을 활용하여 사건의 발생 가능성을 예측(확률)하여 이항 분류를 진행하는 분석 방법

###### 적합한 데이터 특성
- 독립 변수와 종속 변수 간의 관계가 선형적으로 결정될 때 가장 효과적. 비선형 분포를 가지는 경우, 로지스틱 회귀의 성능이 제한될 수 있음
- 변수들 간의 스케일 차이가 크면 모델의 수렴 속도에 영향을 줄 수 있습니다. 따라서, 일반적으로 변수들을 정규화(예: 표준화)하는 것이 좋습니다.

###### 수행 예시
~~~ python
## 6. 데이터분석 수행
# LogisticRegression 객체 생성
lr = LogisticRegression()
lr.fit(X_train, y_train)        # 학습 수행

# 학습이 완료된 dt객체에서 테스트 데이터셋으로 예측 수행
pred = lr.predict(X_test)
~~~

###### [[오즈 (Odds)]]
- 사건이 발생활 확률이 사건이 발생하지 않을 확률 의 몇배인지에 대한 개념이다.
	- 오즈(Odds)가 3이라면 성공 확률이 실패 확률의 3배라는 의미 

~~~ python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 원본 데이터프레임
data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 설명 변수와 종속 변수 설정
X = df[['Feature1', 'Feature2']]
X = sm.add_constant(X)  # 상수항 추가
y = df['Target']

# 1. 로지스틱 회귀 모델 학습
model = sm.Logit(y, X)
result = model.fit()

# 2. 모델 계수 출력
coef = result.params

# 3. 오즈 비율 계산
odds_ratios = np.exp(coef)

# 오즈 비율 해석
print("\nInterpretation of Odds Ratios:")
for feature, odds_ratio in odds_ratios.items():
    print(f"{feature}: Odds Ratio = {odds_ratio:.2f}")
~~~
###### 오즈비 (Odds ratio)
- 오즈비는 오즈의 비율로 오즈를 오즈로 나눔으로써 비교할 수 있다. 
	- 비흡연자에 대한 흡연자의 폐암 발병 오즈비를 구하면 4/0.25로 오즈비는 16이 되고, 
	- 이는 흡연자가 비흡연자 보다 폐암 발병 확률이 16배 높다고 해석할 수 있다.
###### 로짓 함수
- 함수의 결과 값이 0에서 1사이의 값을 반환하는 함수
###### 시그모이드(Sigmoid) 함수
- 로짓함수와 역함수 관계이다.

##### [[의사결정 트리 분석|의사결정나무]] - 분류 분석
###### 정의
- 데이터를 학습하여 데이터 내에 존재하는 규칙을 찾아내고, 이 규칙을 나무구조로 모형화해 분류와 예측을 수행하는 방법이다.
- 올바른 분류를 위해서 상위노드에서 하위노드로 갈수록 집단 내에서는 동질성을 가지고, 집단간에는 이질성이 커져야한다.
- 중간 노드가 많다는 것은 규칙이 복잡하다는 이야기이므로, 모델이 과적합되기 쉽기 때문에 나무의 깊이를 적절하게 조절 해야한다
###### 연관개념
- 엔트로피 (Entropy):
	- 데이터의 불확실성을 측정하는 지표. 엔트로피가 높을수록 불확실성이 큼.
- 정보 이득 (Information Gain):
	- 특정 특성을 기준으로 데이터를 분할했을 때의 엔트로피 감소량. 정보 이득이 클수록 해당 특성은 좋은 분할 기준이 됨.
- 과적합 (Overfitting):
	- 모델이 훈련 데이터에 지나치게 맞춰져 새로운 데이터에 대한 예측 성능이 떨어지는 현상. 결정 트리에서 주로 트리의 깊이가 깊어질 때 발생함.
###### 수행 예시
~~~ python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 결정 트리 모델 생성 (엔트로피 기준)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X, y)

# 트리 시각화
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
~~~

![[데이터분석_의사결정나무_시각화.png]]
##### [[랜덤 포레스트]] - 분류 분석
###### 정의
- [[의사결정 트리 분석]] 기반의 알고리즘으로 다수의 의사결정 트리들을 배깅하여 분류 또는 회귀를 수행하는 [[앙상블 분석|앙상블 기법]] 중 하나이다. 
- 각 트리는 학습 데이터 중 서로 다른 데이터를 샘플링하여, 일부 데이터를 제외한 후 최적의 특징을 찾아 트리를 분기한다.
###### 과정
- 배깅의 일종으로 배깅에 변수 랜덤 선택 과정을 추가한 것이다.
- [[부트스트랩(Bootstrap)|부트스트랩]] 방식을 통해 변수를 선택하므로 입력변수가 아주 많은 경우에도 변수를 제거하지 않고 분석하는 것이 가능하다.![[앙상블 분석_랜덤포레스트 과정.png]]

###### 수행 예시
~~~ python
## 6. 데이터 분석 수행
# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)
rf.fit(X_train, y_train)          # 학습 수행

# 학습이 완료된 rf객체에서 테스트 데이터셋으로 예측 수행
pred = rf.predict(X_test)
~~~
##### [[서포트 벡터 머신(SVM)]] - 분류분석

###### 정의
- 서포트 벡터 머신에서는 데이터가 n차원으로 주어졌을때 이러한 데이터를 n-1차원의 초평면으로 분리한다.
- 데이터가 어느 카테고리에 속할지 판단하는 이진 선형 분류 모델을 만드는 기법으로 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 그 중 가장 큰 폭을 가지는 경계를 찾는 알고리즘이다.
![[서포트 벡터의 구성요소.png]]
###### 용어 설명
- Margin: 
	- 초평면과 각 클래스의 가장 가까운 데이터 포인트들 간의 거리를 의미합니다. SVM은 이 Margin을 최대화하는 초평면을 찾습니다. 이 데이터 포인트들을 Support Vectors라고 부릅니다.
- Hyperplane (초평면): 
	- 주어진 데이터셋을 두 클래스로 나누는 (n-1)차원 초평면을 말합니다. 예를 들어, 2차원에서는 직선(line), 3차원에서는 평면(plane)입니다.
	- SVM에서 데이터 포인트를 분리하는 선형 결정 경계(Decision Boundary)를 의미합니다. 이 초평면은 SVM 알고리즘에 의해 정의됩니다.
	- 비선형 데이터의 경우, 커널 트릭을 사용하여 데이터를 고차원 공간으로 변환하고, 그곳에서의 Hyperplane을 찾습니다.
- Hinge Loss (손실 함수):
	- SVM의 목적 함수를 정의하는 데 사용됩니다. SVM의 목표는 Hinge Loss를 최소화하여 최적의 결정 경계를 찾는 것입니다.
###### 구성 요소
- Support Vectors: 
	- 결정 경계를 정의하는데 중요한 역할을 하는 데이터 포인트들입니다. 이들은 초평면과 가장 가까운 포인트들이며, 초평면의 위치가 이들에 의해 결정됩니다.
- Kernel Trick: 
	- 비선형 데이터에 대해 SVM을 적용할 수 있게 해주는 기술입니다. 데이터를 고차원으로 변환하여 선형 분리가 가능하게 합니다. 일반적으로 사용되는 커널에는 선형, 다항식, RBF(Radial Basis Function) 등이 있습니다.
- Regularization Parameter (C): 
	- 오차 허용도를 조정합니다. 작은 C 값은 더 큰 마진을 허용하지만, 잘못 분류된 데이터 포인트가 많을 수 있습니다. 큰 C 값은 마진을 작게 하여 오차를 줄이지만, 과적합(overfitting)의 위험이 있습니다.
	- Soft Margin Classification에서 중요한 역할을 하는 하이퍼파라미터입니다. 이 매개변수는 모델의 정규화(regularization) 강도를 조절합니다.

###### 수행 예시-1 (linear Support Vector Machine)
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 데이터 생성
np.random.seed(42)
X, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM 모델 훈련 : SVC 클래스를 사용해 선형 커널로 SVM 모델을 훈련합니다.
model = SVC(kernel='linear', C=1, random_state=42)
model.fit(X_scaled, y)

# 결정 경계 시각화
def plot_decision_boundary(clf, X, y, ax):
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    ax.set_title("SVM Decision Boundary")

fig, ax = plt.subplots()
plot_decision_boundary(model, X_scaled, y, ax)
plt.show()
~~~

![[데이터 분석_SVM_시각화.png]]

###### 수행 예시-2 (Non-linear Support Vector Machine)
- 선형 분리의 한계: 
	- 데이터가 선형적으로 구분되지 않는 경우, 즉 데이터 포인트를 직선이나 평면으로 구분할 수 없는 경우, 일반적인 선형 SVM은 적합하지 않습니다. 
	- 이럴 때는 데이터의 차원을 증가시켜서 비선형적으로 구분할 수 있는 초평면을 찾을 필요가 있습니다.
- 커널 함수 (Kernel Function): 
	- Kernel Trick을 사용하면 원래의 데이터 공간을 고차원으로 변환하지 않고도 비선형 결정 경계를 찾을 수 있습니다. 
	- 커널 함수는 데이터의 차원을 암묵적으로 증가시키며, 비선형 데이터를 처리할 수 있도록 해줍니다.
	- 대표적인 커널 함수로는 RBF 커널, Polynomial 커널, Sigmoid 커널이 있습니다.

~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 데이터 생성 (비선형적으로 분리 가능)
np.random.seed(42)
X, y = datasets.make_moons(n_samples=200, noise=0.3, random_state=42)

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RBF 커널을 사용하는 SVM 모델 훈련, gamma는 RBF 커널의 폭을 조절하는 매개변수입니다.
model = SVC(kernel='rbf', C=1, gamma=0.5, random_state=42)
model.fit(X_scaled, y)

# 결정 경계 시각화
def plot_decision_boundary(clf, X, y, ax):
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    ax.set_title("Non-linear SVM Decision Boundary (RBF Kernel)")

fig, ax = plt.subplots()
plot_decision_boundary(model, X_scaled, y, ax)
plt.show()
~~~

![[데이터분석_SVM_비선형_시각화.png]]

###### 수행 예시-3 (Hinge Loss 계산)

~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss
from matplotlib.colors import ListedColormap

# 데이터 생성 (비선형적으로 분리 가능)
np.random.seed(42)
X, y = datasets.make_moons(n_samples=200, noise=0.3, random_state=42)

# 레이블을 -1, 1로 변환
y = 2 * y - 1  # 변환된 레이블: 0 -> -1, 1 -> 1

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RBF 커널을 사용하는 SVM 모델 훈련
model = SVC(kernel='rbf', C=1, gamma=0.5, random_state=42)
model.fit(X_scaled, y)

# SVM 모델의 결정 함수 값을 계산
decision_function = model.decision_function(X_scaled)

# Hinge Loss 계산
loss = hinge_loss(y, decision_function)
print("Hinge Loss:", loss)
~~~

~~~
Hinge Loss: 0.26111843832198867
~~~

#### 지도학습 - 회귀분석
##### [[단순 선형 회귀분석|단변량 선형 회귀 (One Variable Linear Regression)]]
###### 정의
- 독립변수와 종속변수 간에 선형적인 관계를 도출하여 독립변수가 종속변수에 미치는 영향을 분석하고, 독립변수를 통해 종속변수를 예측하는 분석기법
###### 분석 수행
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 예제 데이터
# 집의 크기(제곱미터)와 가격(USD)
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)
y = np.array([150000, 180000, 210000, 240000, 270000, 300000])

# 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X, y)

# 예측
y_pred = model.predict(X)

# 회귀 직선 그리기
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('Size (square meters)')
plt.ylabel('Price (USD)')
plt.title('One Variable Linear Regression')
plt.legend()
plt.show()

# 회귀 계수
print(f'Intercept (β0): {model.intercept_:.2f}')
print(f'Slope (β1): {model.coef_[0]:.2f}')
~~~

##### [[다중 선형 회귀분석|다변량 선형 회귀(Multivariable Linear Regression)]]
###### 정의
- 하나의 독립변수가 아닌 여러 개의 독립변수를 사용하는 회귀분석 기법이다.
- 단순 선형 회귀분석이 독립변수를 하나 가지고 있는 선형 회귀분석이라면,
- 다중 선형 회귀분석은 독립변수가 두 개 이상이고 종속변수가 y 하나인 선형 회귀분석이다.
###### 분석 수행
~~~ python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 예제 데이터 (집의 크기, 방의 수, 위치, 가격)
data = {
    'Size': [50, 60, 70, 80, 90, 100],  # 집의 크기 (제곱미터)
    'Rooms': [1, 2, 2, 3, 3, 4],        # 방의 수
    'Location': [1, 2, 2, 3, 3, 1],     # 위치 (1: 저렴, 2: 중간, 3: 비싼)
    'Price': [150000, 180000, 210000, 240000, 270000, 300000]  # 집의 가격 (USD)
}

df = pd.DataFrame(data)

# 독립 변수와 종속 변수 분리
X = df[['Size', 'Rooms', 'Location']]
y = df['Price']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# 회귀 계수
print(f'Intercept (β0): {model.intercept_:.2f}')
print(f'Coefficients (β1, β2, β3): {model.coef_}')
~~~

##### 규제모델 - [[Lasso|Lasso 회귀]]
###### 정의
- 기존 선형회귀식에 규제항(L1 Regularization)이 적용된 모델
- Ridge 회귀와 유사하지만, 계수의 절대값의 합에 패널티를 추가하여 변수 선택 및 모델 간소화를 수행
###### 특징
- L1 정규화는 일부 회귀 계수를 0으로 만들 수 있어 변수 선택(variable selection)에 유용합니다. 즉, 중요하지 않은 특성의 계수를 0으로 만들어 모델을 단순화하고 해석하기 쉽게 합니다.
###### 차이점 (Lasso, Ridge)
- 정규화 방식:
	- Lasso: L1 정규화 (절대값 합의 패널티) → 계수를 0으로 만들 수 있음
	- Ridge: L2 정규화 (제곱합의 패널티) → 계수를 0으로 만들지 않음
- 주요 목적:
	- Lasso: 변수 선택 및 모델 단순화
	- Ridge: 과적합 방지 및 모델 안정성 향상
- 계수의 영향을 받는 방식:
	- Lasso: 일부 계수를 0으로 만들어서 중요하지 않은 변수를 제거
	- Ridge: 계수의 크기를 줄여서 모든 변수를 유지

###### 분석 수행
~~~ python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 예제 데이터 (집의 크기, 방의 수, 위치, 가격)
data = {
    'Size': [50, 60, 70, 80, 90, 100],
    'Rooms': [1, 2, 2, 3, 3, 4],
    'Location': [1, 2, 2, 3, 3, 1],
    'Price': [150000, 180000, 210000, 240000, 270000, 300000]
}

df = pd.DataFrame(data)
X = df[['Size', 'Rooms', 'Location']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lasso 회귀 모델 생성
lasso = Lasso(alpha=1.0)  # alpha는 정규화 강도
lasso.fit(X_train, y_train)

# 예측
y_pred = lasso.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Lasso Regression MSE: {mse:.2f}')
print(f'Coefficients: {lasso.coef_}')
~~~

~~~
Lasso Regression MSE: 0.06
Coefficients: [2999.992    0.      -0.   ]
~~~
##### 규제모델 - [[Ridge|Ridge 회귀]]
###### 정의
- 기존 선형회귀식에 규제항(L2 Regularization)이 적용된 모델
- 선형 회귀의 정규화 버전으로, 회귀 계수에 패널티를 추가하여 과적합을 방지
###### 특징
- L2 정규화는 모든 회귀 계수를 작게 만들어 과적합을 방지하지만, 계수를 정확히 0으로 만들지는 않습니다. 따라서 변수 선택보다는 변수의 계수를 축소하여 모델의 안정성을 높이는 데 유용합니다.

###### 분석 수행
~~~ python
from sklearn.linear_model import Ridge

# Ridge 회귀 모델 생성
ridge = Ridge(alpha=1.0)  # alpha는 정규화 강도
ridge.fit(X_train, y_train)

# 예측
y_pred = ridge.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Ridge Regression MSE: {mse:.2f}')
print(f'Coefficients: {ridge.coef_}')
~~~

~~~
Ridge Regression MSE: 68178.49
Coefficients: [2984.51076691  147.33660748  -22.66717038]
~~~

##### [[의사결정 트리 분석|의사결정나무]] - 회귀
###### 수행 예시
~~~ python
## 6. 데이터 분석 수행
# DecisionTreeRegressor 객체 생성
dtr = DecisionTreeRegressor(max_depth=3, random_state=43)
dtr.fit(X_train, y_train)     # 학습 수행

# 학습이 완료된 dtr객체에서 테스트 데이터셋으로 예측 수행
pred = dtr.predict(X_test)
~~~

#### 지도학습 - 기타
##### 인공신경망 (ANN)
- 인간의 뇌 구조를 모방하여 데이터의 패턴을 학습합니다. 다양한 층(layer)과 노드(node)를 사용하여 복잡한 관계를 모델링합니다.

##### GBM (Gradient Boosting Model)
###### 정의
- Gradient Boosting은 일련의 약한 학습자를 순차적으로 학습시키면서, 이전 모델이 예측하지 못한 부분을 다음 모델이 보완하도록 하는 기법입니다.
- 각 단계에서 모델은 이전 단계의 오차를 줄이기 위해 그라디언트 하강법(Gradient Descent)을 사용해 예측을 개선합니다.
- 주로 회귀 트리(Regression Tree)를 사용하며, 새로운 트리가 추가될 때마다 모델의 성능이 점진적으로 향상됩니다.

###### 수행 예시
~~~ python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gradient Boosting 모델 생성
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = gbm_model.predict(X_test)
print("GBM Accuracy:", accuracy_score(y_test, y_pred))
~~~

##### XGBoost (Extreme Gradient Boosting)
###### 정의
- XGBoost는 Gradient Boosting 알고리즘의 확장으로, 성능과 속도를 극대화하기 위해 여러 가지 최적화 기법을 도입한 모델입니다.
- 정규화(Regularization)를 통해 과적합을 방지하고, 병렬 처리를 통해 연산 속도를 대폭 개선하였습니다.
- XGBoost는 트리 기반의 앙상블 기법으로, 손실 함수를 최소화하기 위해 그라디언트 부스팅을 사용하면서도 추가적인 정규화 항목을 포함하여 모델의 일반화 성능을 높였습니다.
- 또한, 누락된 값을 자동으로 처리하고, 사용자 정의 손실 함수 및 평가 지표를 지원하는 등 유연성이 뛰어납니다.

###### 수행 예시
~~~ python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# XGBoost 모델 설정
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
~~~
- 주요 파라미터 설명
	- objective='multi:softmax' : 다중 클래스 분류를 위해, XGBoost는 입력 데이터를 3개의 클래스 중 하나로 분류합니다. num_class=3는 클래스 수를 지정합니다.
	- learning_rate=0.1 : 학습률을 설정합니다. 모델이 학습하는 속도를 조절합니다.
	- n_estimators=100 : 트리의 개수(부스팅 라운드 수)를 설정합니다.
	- max_depth=5 : 각 트리의 최대 깊이를 설정합니다. 깊이가 깊을수록 더 복잡한 모델을 생성합니다.


##### LightGBM (Light Gradient Boosting Model)
###### 정의
- LightGBM은 Microsoft가 개발한 Gradient Boosting 모델의 변형으로, 특히 대규모 데이터셋에 대한 효율성을 극대화한 알고리즘입니다.
- LightGBM은 트리의 성장을 수직적으로(Leaf-wise) 수행하여, 손실을 가장 많이 줄일 수 있는 리프(leaf)부터 먼저 확장합니다. 이 방식은 전통적인 수평적(Level-wise) 성장을 하는 GBM에 비해 더 적은 계산으로도 높은 성능을 유지할 수 있습니다.
- 또한, LightGBM은 GPU를 지원하여 대규모 데이터셋에 대해 빠른 학습을 가능하게 합니다.
- 과적합을 피하기 위해 트리의 깊이에 제한을 두는 방식으로 일반화 능력을 높입니다.

###### 수행 예시
~~~ python
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# LightGBM 데이터셋 준비
# - 검증 데이터셋을 생성하며, reference=train_data로 지정하여 검증 데이터가 훈련 데이터와 같은 형식과 특성을 가지도록 합니다.
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LightGBM 모델 설정
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# 모델 학습
# - lgb_train과 lgb_test를 valid_sets로 제공하여, 훈련 중에 이 두 데이터셋에서 모델 성능을 평가합니다.
lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=10)

# 예측 및 평가
# - best_iteration은 훈련 동안 조기 종료에 의해 성능이 가장 좋았던 부스팅 라운드를 나타냅니다. 이 라운드는 모델의 성능이 가장 최적이었던 상태를 반영합니다.
# - accuracy_score 함수는 클래스 레이블을 입력으로 받아서 정확도를 계산합니다. 따라서, y_pred는 LightGBM의 predict 메서드에서 반환된 확률 벡터이기 때문에 직접 사용할 수는 없으며, y_pred_classes와 같은 클래스 레이블로 변환된 결과를 사용해야 합니다.
y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
y_pred_classes = [list(x).index(max(x)) for x in y_pred]  # 확률값을 클래스 값으로 변환
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_classes))
~~~

~~~ python
# y_pred_classes를 좀 더 쉽게 작성한다면

y_pred_classes = []

for x in y_pred:
    # x는 각 샘플에 대한 확률 벡터입니다
    # 최대값을 찾고, 그 최대값의 인덱스를 찾습니다
    max_prob = max(x)  # 최대 확률 값
    class_label = list(x).index(max_prob)  # 최대값의 인덱스 (클래스 레이블)
    y_pred_max.append(class_label)  # 결과 리스트에 추가

# y_pred_classes는 각 샘플에 대한 예측 클래스 레이블을 포함합니다
print(y_pred_classes)
~~~


#### 비지도학습
##### [[군집분석]]
###### 정의
- 군집분석(군집화)은 클러스터링의 대표적인 응용 분야입니다.
- 클러스터링(Clustering)은 데이터를 그룹으로 나누는 비지도 학습의 한 기법으로, 유사한 데이터 포인트를 같은 그룹(클러스터)으로 묶는 방법입니다.
###### 계층적 클러스터링 (Hierarchical Clustering)
: 데이터 포인트들을 계층적으로 군집화하는 방법입니다.
- 병합적(Agglomerative) 클러스터링: 
	- 각 데이터 포인트를 개별 클러스터로 시작해 점차적으로 가장 가까운 클러스터를 합치며 클러스터의 수를 줄여가는 방식입니다.
- 분할적(Divisive) 클러스터링: 
	- 전체 데이터를 하나의 클러스터로 시작해 점차적으로 클러스터를 나누어가는 방식입니다.
###### 비계층적 클러스터링 (Non-Hierarchical Clustering)
: 데이터 포인트를 한번에 여러 그룹으로 나누는 방식입니다.
- K-평균 클러스터링(K-Means Clustering): 
	- 가장 널리 사용되는 클러스터링 방법 중 하나로, 데이터를 K개의 클러스터로 나누고 각 클러스터의 중심(centroid)과 가장 가까운 데이터 포인트들을 묶는 방식입니다.
- DBSCAN(Density-Based Spatial Clustering of Applications with Noise): 
	- 밀도 기반 클러스터링 방법으로, 밀도가 높은 지역을 클러스터로 간주하고, 밀도가 낮은 포인트는 노이즈로 처리합니다.
- GMM(Gaussian Mixture Model): 
	- 클러스터들이 가우시안 분포를 따른다고 가정하여 데이터를 클러스터링하는 방법입니다.
##### K-평균 클러스터링(K-Means Clustering)
###### 정의
- K-평균 클러스터링은 데이터셋을 K개의 클러스터로 나누는 비계층적 클러스터링 알고리즘입니다.

###### 형성 과정
- Step 1 랜덤하게 K개(사전의 정의)의 중앙점을 지정
  ![[군집분석_K-means 참고이미지1.png]]
- Step 2 유클리드 거리를 이용하여 군집의 중심점과의 거리 계산 및 가장 거리가 가까운 군집으로 군집 할당
  ![[군집분석_K-means 참고이미지2.png]]
- Step 3 군집 내 데이터의 평균을 계산 후 새로운 중심점으로 배정 및 중심점이 더 이상 이동 안 할 때까지 반복
  ![[군집분석_K-means 참고이미지3.png]]

###### 예시
한 회사가 고객을 몇 개의 그룹으로 나누어 각 그룹에 맞는 마케팅 전략을 수립하고자 한다고 가정해봅시다. 이때 K-평균 클러스터링을 사용해 고객의 나이와 연간 소득을 기준으로 3개의 클러스터(K=3)로 나눌 수 있습니다. 각 고객은 가장 가까운 클러스터 중심에 할당되고, 이를 통해 비슷한 소비 성향을 가진 고객 그룹을 식별할 수 있습니다.

###### 수행 예시
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 간단한 2D 데이터셋 생성
np.random.seed(42)
X = np.vstack([
    np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[8, 8], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
])

# K-평균 클러스터링 적용 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title("K-Means Clustering")
plt.show()
~~~

![[데이터분석_K-평균 클러스터링 (K-Means Clustering)_시각화.png]]


##### 계층적 클러스터링 (Hierarchical Clustering)
###### 정의
- 계층적 클러스터링은 데이터를 계층 구조로 그룹화하는 알고리즘입니다. 
- 이 방법은 클러스터링 결과를 트리 구조(덴드로그램)로 표현할 수 있으며 크게 두 가지로 나뉩니다.
	- 병합적 계층적 클러스터링(Agglomerative Clustering): 처음에 모든 데이터 포인트를 개별 클러스터로 시작하여, 가장 가까운 두 클러스터를 병합하는 방식으로 진행됩니다. 이 과정을 반복하여 클러스터의 수를 점점 줄여나갑니다.
	- 분할적 계층적 클러스터링(Divisive Clustering): 전체 데이터를 하나의 클러스터로 시작하여, 점차적으로 클러스터를 분할하는 방식입니다.

###### 예시
생물학적 데이터에서 여러 종(species)을 계층적으로 군집화하여 진화적 관계를 분석하고자 한다고 가정해봅시다. 병합적 계층적 클러스터링을 사용해, 각 종을 개별 클러스터로 시작해 가장 유사한 두 종을 병합하여 트리 구조를 형성할 수 있습니다. 최종적으로, 덴드로그램을 통해 어떤 종들이 서로 가까운 유전적 관계를 가지고 있는지 시각적으로 확인할 수 있습니다.

###### 수행 예시
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
import pandas as pd

# 간단한 2D 데이터셋 생성
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

# 1. 계층적 클러스터링 적용 (병합적)
linked = linkage(X, method = 'ward')

# 2. 덴드로그램 시각화
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# 3. 클러스터 수를 3으로 설정하여 클러스터 레이블 생성
cluster_labels = fcluster(linked, 3, criterion='maxclust')

# 4. 각 데이터 포인트의 클러스터 레이블 출력
df = pd.DataFrame(
    {'X1':X[:, 0],
     'X2':X[:, 1],
     'Label':cluster_labels}
)
print(df)
~~~

![[데이터분석_계층적 클러스터링 (Hierarchical Clustering)_시각화.png]]

~~~
          X1        X2  Label
0   2.401222  0.772684      1
1   0.438990  4.535929      3
2   2.623619  0.804658      1
3   0.007931  4.176143      3
4   0.347138  3.451777      3
..       ...       ...    ...
95 -0.814086  3.108048      2
96  1.164111  3.791330      3
97  1.494932  3.858488      3
98 -1.151765  1.956648      2
99 -2.262165  3.424500      2
~~~
##### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
###### 정의
- DBSCAN은 밀도 기반 클러스터링 알고리즘으로, 밀도가 높은 지역을 클러스터로 간주하고, 밀도가 낮은 포인트는 노이즈로 처리합니다. 
- DBSCAN은 핵심 포인트를 찾고, 이웃 포인트들을 연결해 클러스터를 형성합니다. 밀도가 낮은 영역에 있는 포인트는 노이즈로 간주되어 클러스터에 속하지 않습니다.
###### 예시
지리적 데이터를 이용해 밀도가 높은 지역에서 발생하는 지진을 분석하고자 한다고 가정해봅시다. DBSCAN을 사용하면, 지진이 자주 발생하는 지역을 클러스터로 식별할 수 있습니다. 이때, 드물게 발생하는 노이즈(희귀한 지진)는 별도의 클러스터로 식별되지 않으며, 이는 지진의 중심지와 여진을 명확하게 구분하는 데 유용합니다.
###### 수행예시
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 비정형 데이터셋 (두 개의 반달형 데이터) 생성
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# DBSCAN 클러스터링 적용 
#  : DBSCAN을 사용해 클러스터링을 수행했습니다. eps=0.2와 min_samples=5로 설정하여 밀도가 높은 영역을 클러스터로 인식하고, 밀도가 낮은 포인트는 노이즈로 처리했습니다.
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.show()
~~~

![[데이터 분석_DBSCAN_시각화.png]]

#### [[앙상블 분석|앙상블학습]]
##### 개념
- 여러 개의 모델(일반적으로 약한 학습자 또는 약한 모델이라고 부름)을 결합하여 최종 예측을 수행하는 방법입니다.
- 개별 모델의 오차를 상쇄하여, 더 강력한 예측 성능을 발휘합니다.
##### 학습 목적
- 모델 성능 향상: 
	- 단일 모델이 가진 한계를 극복하고, 다양한 모델의 강점을 결합함으로써 예측 성능을 향상시킵니다.
- 일반화 능력 개선: 
	- 앙상블 모형은 과적합(overfitting) 문제를 줄이는 데 도움이 됩니다. 여러 모델의 예측을 결합하면 개별 모델이 데이터에 너무 과적합되는 문제를 완화할 수 있습니다.
- 예측 안정성 증가: 
	- 단일 모델은 특정 데이터 분포에 취약할 수 있습니다. 여러 모델을 결합하면 특정 모델의 편향으로 인한 오류를 줄일 수 있어, 예측의 일관성과 안정성이 증가합니다.
##### 주요 기법
###### Bagging
- 데이터의 여러 샘플을 무작위로 선택하여 각각의 모델을 학습시키고, 이들의 예측을 결합하는 방법입니다. 대표적으로 [[랜덤 포레스트|랜덤 포레스트(Random Forest)]]가 있습니다.
- RandomForest와 Bagging을 사용한 앙상블 학습 후 정확도 비교 예:
~~~ python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score

# 데이터셋 생성 (예시로 만든 단순한 데이터셋)
np.random.seed(0)
X = np.random.rand(100, 5)  # 100개의 샘플, 5개의 피처
y = np.random.randint(0, 2, 100)  # 이진 분류 레이블

# 데이터셋을 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 코딩 문제: RandomForest와 Bagging을 사용한 앙상블 학습 후 정확도 비교

# 1. RandomForestClassifier를 사용한 단일 모델 학습
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 2. BaggingClassifier를 사용한 앙상블 모델 학습
bg = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10, random_state=0)
bg.fit(X_train, y_train)

# 3. 두 모델의 예측 및 성능 평가
pred_rf = rf.predict(X_test)
pred_bg = bg.predict(X_test)

# 4. 정확도 비교
acc_rf = accuracy_score(y_test, pred_rf)
acc_bg = accuracy_score(y_test, pred_bg)

print(f"RandomForest 정확도: {acc_rf:.4f}")
print(f"Bagging 정확도: {acc_bg:.4f}")

~~~
![[앙상블 분석_배깅.png|700]]
###### Boosting
- 이전 모델의 오차를 보완하는 방식으로 모델을 순차적으로 학습시켜 강력한 모델을 만드는 방법입니다. 대표적으로 에이다부스트(AdaBoost), 그래디언트 부스팅(Gradient Boosting), XGBoost가 있습니다.
- 그래디언 부스팅 예시:
~~~ python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 그래디언트 부스팅 모델 생성
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
~~~

![[앙상블 분석_부스팅 학습 과정.png]]
###### Stacking
- 여러 모델의 예측 결과를 기반으로 다시 학습을 수행하여 최종 예측을 하는 방법입니다. 여기서 메타 모델(Meta Model)이 최종 결합을 수행합니다.
~~~ python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 개별 모델
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(random_state=42, probability=True))
]

# 스태킹 모델
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = stacking_model.predict(X_test)
print("Stacking Model Accuracy:", stacking_model.score(X_test, y_test))
~~~

### Improving Performance
#### 과적합과 과소적합
##### 과적합 (Overfitting)
###### 정의 
- 과적합은 모델이 훈련 데이터에 너무 잘 맞춰져서, 훈련 데이터에서 높은 성능을 보이지만 새로운 데이터나 검증 데이터에서는 성능이 떨어지는 현상입니다.
- 모델이 데이터의 노이즈나 불필요한 세부 사항까지 학습하여 일반화되지 못하는 경우입니다.
###### 해결방안
- 정규화 (Regularization):
	- L1 정규화 (Lasso): 모델의 일부 계수를 0으로 만들어 변수 선택을 수행하여 과적합을 방지합니다.
	- L2 정규화 (Ridge): 모든 계수의 크기를 줄여 모델의 복잡도를 줄입니다.
- 교차 검증 (Cross-Validation):
	- 데이터를 여러 부분으로 나누어 모델을 평가하여 과적합의 징후를 조기에 발견합니다.
- 훈련 데이터 확대 (Data Augmentation):
	- 데이터의 양을 늘리거나 데이터를 변형하여 더 많은 학습 샘플을 제공하여 모델의 일반화 능력을 향상시킵니다.
###### 수행 예시
-  과적합 문제 해결 방법으로 L1 정규화 (Lasso)와 교차 검증을 사용하여 모델의 성능을 개선하는 방법을 보여줍니다.

~~~ python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 (과적합 가능성 높은 복잡한 모델)
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 성능 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Without Regularization - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')

# Lasso 회귀 모델 생성 (정규화 적용)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 예측
y_train_pred_lasso = lasso.predict(X_train)
y_test_pred_lasso = lasso.predict(X_test)

# 성능 평가
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
print(f'Lasso Regression - Train MSE: {train_mse_lasso:.2f}, Test MSE: {test_mse_lasso:.2f}')

# 교차 검증
cv_scores = cross_val_score(lasso, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation MSE: {-cv_scores.mean():.2f}')
~~~

~~~
Without Regularization - Train MSE: 0.01, Test MSE: 0.02
Lasso Regression - Train MSE: 0.11, Test MSE: 0.11
Cross-validation MSE: 0.18
~~~

##### 과소적합 (Underfitting)
###### 정의 
- 과소적합은 모델이 훈련 데이터와 검증 데이터 모두에서 낮은 성능을 보이는 현상입니다. 
- 모델이 데이터의 패턴을 충분히 학습하지 못하고, 너무 단순하여 문제를 제대로 해결하지 못하는 경우입니다.
###### 해결방안
- 모델 복잡도 증가:
	- 더 복잡한 모델을 사용하거나, 더 많은 특성(feature)을 추가하여 모델이 데이터를 잘 학습할 수 있도록 합니다.
- 특성 엔지니어링 (Feature Engineering):
	- 데이터에서 더 많은 유용한 특성을 추출하거나, 기존 특성의 조합을 통해 모델의 예측 성능을 향상시킵니다.
- 훈련 데이터 확대:
	- 더 많은 훈련 데이터를 수집하여 모델이 데이터의 패턴을 더 잘 학습하도록 합니다.
- 학습 기간 증가 (Increase Training Time):
	- 모델을 충분히 학습시키기 위해 학습 기간을 늘리거나, 더 많은 에포크(epoch)를 사용하여 모델이 데이터 패턴을 학습할 수 있도록 합니다.
###### 수행 예시
- 과소적합 문제를 해결 방법으로 모델 복잡도를 증가시키는 방법을 보여줍니다.
~~~ python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 단순 선형 회귀 모델 생성 (과소적합 가능성 높은 간단한 모델)
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 성능 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Linear Regression - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')

# 다항 회귀 모델 생성 (모델 복잡도 증가)
poly = PolynomialFeatures(degree=3)  # 3차 다항식 특성 추가
poly_model = make_pipeline(poly, LinearRegression())
poly_model.fit(X_train, y_train)

# 예측
y_train_pred_poly = poly_model.predict(X_train)
y_test_pred_poly = poly_model.predict(X_test)

# 성능 평가
train_mse_poly = mean_squared_error(y_train, y_train_pred_poly)
test_mse_poly = mean_squared_error(y_test, y_test_pred_poly)
print(f'Polynomial Regression - Train MSE: {train_mse_poly:.2f}, Test MSE: {test_mse_poly:.2f}')
~~~ 

#### 교차 검증 (Cross-Validation)
- 교차 검증(Cross-Validation)은 머신러닝 모델의 일반화 성능을 평가하고 과적합(overfitting)이나 과소적합(underfitting) 문제를 확인하기 위한 중요한 방법입니다. 
- 데이터의 분할 방법을 통해 모델의 성능을 신뢰성 있게 평가할 수 있도록 도와줍니다.

##### K-겹 교차 검증 (K-Fold Cross-Validation)
###### 정의
- 데이터를 K개의 폴드로 나누어, 각 폴드가 검증 데이터로 사용되도록 하여 K번 모델을 훈련합니다.
- 각 폴드가 검증 데이터로 사용될 때, 나머지 K-1개의 폴드가 훈련 데이터로 사용됩니다.
- 결과적으로 K개의 모델 성능 평가 결과를 평균내어 모델의 성능을 평가합니다.
- 일반적으로 K=5 또는 K=10이 많이 사용됩니다.

![[데이터분석_K-폴드 교차검증.png]]
###### 사용이유
- 총 데이터 갯수가 적은 데이터 셋에 대하여 정확도를 향상시킬 수 있음
- 데이터의 모든 부분이 훈련 및 검증 데이터로 사용되므로 데이터가 적은 경우 유용합니다.
###### 수행 예시
~~~ python
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# 모델 정의
model = LogisticRegression(max_iter=200)

# K-겹 교차 검증 설정 (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # cv에 직접 5를 입력하여도 무방하나, kf와 같이 객체형태로 전달하면 교차 검증의 동작을 세밀하게 조정할 수 있음

# 교차 검증 수행
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# 결과 출력
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean accuracy: {np.mean(cv_scores):.2f}')
print(f'Standard deviation: {np.std(cv_scores):.2f}')
~~~

~~~
Cross-validation scores: [1.         1.         0.93333333 0.96666667 0.96666667]
Mean accuracy: 0.97
Standard deviation: 0.02
~~~

##### Leave-One-Out 교차 검증 (LOOCV)
###### 정의
- 데이터셋에 N개의 샘플이 있을 때, 각 샘플이 단 한 번씩 검증 데이터로 사용됩니다.
- 즉, 총 N번의 반복을 수행하며, 각 반복에서 1개의 샘플을 검증 데이터로 사용하고 나머지 N-1개의 샘플을 훈련 데이터로 사용합니다.
###### 수행 예시
- Iris 데이터셋을 사용하여 로지스틱 회귀 모델을 평가합니다.
~~~ python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

# 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 코딩 문제: LOOCV를 사용한 모델 성능 평가
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 생성
model = LogisticRegression(max_iter=200)

# LOOCV 객체 생성
loo = LeaveOneOut()

# LOOCV 수행
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

# 각 반복의 정확도 출력
for i, score in enumerate(scores):
    print(f"Iteration {i + 1} Accuracy: {score:.4f}")

# 평균 정확도 출력
print(f"Mean Accuracy: {scores.mean():.4f}")
~~~

~~~
Iteration 1 Accuracy: 1.0000
Iteration 2 Accuracy: 1.0000
Iteration 3 Accuracy: 1.0000
...
Iteration 148 Accuracy: 1.0000
Iteration 149 Accuracy: 1.0000
Iteration 150 Accuracy: 1.0000
Mean Accuracy: 0.9667
~~~

#### 홀드아웃 방법 (Hold-Out Method)
###### 정의
- 데이터를 훈련 데이터와 검증 데이터로 단순히 나누어 모델을 훈련하고 평가합니다.
- 일반적으로 70-80%의 데이터를 훈련 데이터로 사용하고 나머지 20-30%를 검증 데이터로 사용합니다.

~~~ python
# 1. 데이터셋 분리 (80% 학습용, 20% 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
~~~



#### 편향과 분산

##### 편향 (Bias)
- 정의: 편향은 모델이 실제 데이터의 관계를 얼마나 잘 학습하고 있는지를 나타냅니다. 모델의 예측 값이 실제 값에서 얼마나 벗어나 있는지를 측정합니다.
- 편향이 높은 경우: 모델이 데이터의 패턴을 충분히 학습하지 못하는 경우, 즉 과소적합(underfitting)이 발생합니다. 모델이 너무 단순하여 데이터의 복잡성을 포착하지 못합니다.
- 예시: 선형 회귀 모델이 복잡한 비선형 데이터를 학습하려고 할 때 높은 편향을 가지며, 이로 인해 모델이 데이터의 패턴을 제대로 캡처하지 못합니다.
##### 분산 (Variance)
- 정의: 분산은 모델의 예측이 데이터의 특정 샘플에 얼마나 민감한지를 나타냅니다. 즉, 데이터의 작은 변화에 대해 모델이 얼마나 변동하는지를 측정합니다.
- 분산이 높은 경우: 모델이 학습 데이터에 너무 잘 맞추어져 있을 때, 즉 과적합(overfitting)이 발생합니다. 모델이 훈련 데이터의 노이즈를 학습하여 새로운 데이터에 대해 잘 일반화하지 못합니다.
- 예시: 복잡한 다항 회귀 모델이 데이터의 노이즈까지 학습하여 훈련 데이터에는 매우 잘 맞지만, 새로운 데이터에는 잘 맞지 않는 경우 높은 분산을 보입니다.
##### Bias-Variance Trade-off
###### 상충 관계
편향과 분산은 서로 상충하는 관계에 있습니다. 모델의 복잡도가 증가하면 편향은 감소하지만 분산은 증가합니다. 반대로 모델의 복잡도가 감소하면 편향은 증가하지만 분산은 감소합니다.
- 간단한 모델: 편향이 높고 분산이 낮습니다. (과소적합 가능성)
- 복잡한 모델: 편향이 낮고 분산이 높습니다. (과적합 가능성)
###### 목표 
최적의 모델을 찾기 위해서는 편향과 분산의 균형을 맞추는 것이 중요합니다. 즉, 모델이 데이터를 잘 학습하면서도 새로운 데이터에 대해서도 잘 일반화할 수 있도록 해야 합니다.

###### 수행 예시
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)

# 모델 복잡도에 따른 예측 성능 평가
degrees = [1, 2, 10]
train_errors = []
test_errors = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1))
    model.fit(X, y)
    
    # 예측
    y_train_pred = model.predict(X)
    y_test_pred = model.predict(X)
    
    # 오차 계산
    train_error = mean_squared_error(y, y_train_pred)
    test_error = mean_squared_error(y, y_test_pred)
    
    train_errors.append(train_error)
    test_errors.append(test_error)
~~~


## 모델 평가와 최적화

### Inferencing

#### 예측 수행
- 학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.
~~~ python
# 새로운 데이터에 대한 예측
y_pred = model.predict(X_test)
~~~
#### 확률 예측 
- 각 클래스에 대한 예측 확률을 반환합니다.
~~~ python
# 예측 확률 (이진 분류일 경우)
y_prob = model.predict_proba(X_test)
~~~

### [[분석모델 성능 평가|Evaluation]]

####  최적화 목표함수
##### 정의
- 최적화 목표 함수는 모델이 학습하는 동안 최적화해야 하는 대상 함수를 말합니다.
- 이 함수는 모델의 성능을 수치적으로 평가하며, 모델이 이 함수를 최소화(혹은 최대화)하도록 학습됩니다.

##### 구분
###### 평가 지표(Metric):

- 정확도(Accuracy), F1-score, ROC-AUC 등과 같은 다양한 평가 지표가 있으며, 모델의 성능을 평가하는 데 사용됩니다. 최적화의 목표는 주어진 평가 지표를 최대화하는 것입니다.
###### 손실 함수(Loss Function):

- 회귀 문제: 평균 제곱 오차(Mean Squared Error, MSE), 평균 절대 오차(Mean Absolute Error, MAE) 등이 사용됩니다. 여기서 목표는 오차를 최소화하는 것입니다.
	- MSE와 MAE 등은 손실함수로 주로 사용되지만, 학습이 끝난 후에도 모델 성능을 평가하는 데 사용하므로 평가지표로도 활용 될 수 있습니다.
- 분류 문제: 크로스 엔트로피 손실(Cross-Entropy Loss) 또는 로그 손실(Log Loss) 등이 사용됩니다. 이 경우 잘못 분류된 샘플의 수나 확률을 최소화하는 것이 목표입니다.
###### 정규화 항(Regularization Term):

- 모델이 과적합되지 않도록 L1 또는 L2 정규화 항을 포함할 수 있습니다. 이 경우 목표는 손실 함수와 정규화 항을 모두 고려하여 최소화하는 것입니다.

#### [[분석모델 성능 평가|분류분석 평가지표]]

- 평가지표 
  ![[데이터분석 평가지표_혼동행렬_평가지표.png]]
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

##### 정확도(Accuracy)
- 전체 샘플 중에서 모델이 올바르게 예측한 샘플의 비율입니다. 모든 클래스가 균등한 비율로 나타나는 경우 유용합니다.
~~~ python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
~~~
##### 정밀도(Precision)
- 모델이 양성(Positive)이라고 예측한 샘플 중 실제로 양성(Positive)인 샘플의 비율입니다.
~~~ python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
~~~
##### 재현율(Recall), 민감도(Sensitivity)
- 실제 양성(Positive) 샘플 중 모델이 올바르게 양성(Positive)으로 예측한 비율입니다. 
~~~ python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
~~~
##### F1-스코어 (F1-Score)
- 정밀도와 재현율의 조화평균으로, 두 지표 간의 균형을 평가합니다. 
~~~ python
# 다중분류 데이터
y_true = pd.DataFrame([2, 2, 3, 3, 2, 1, 3, 3, 2, 1]) # 실제값
y_pred = pd.DataFrame([2, 2, 1, 3, 2, 1, 1, 2, 2, 1]) # 예측값

from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='macro')  # 다중 분류시 컬럼 추가 필요 average= micro, macro, weighted
~~~

##### ROC-AUC (Receiver Operating Characteristic - Area Under Curve) 또는 AUROC
- AUROC는 분류 모델의 성능을 평가하는 중요한 지표 중 하나로, ROC 곡선 아래의 면적(Area Under the ROC Curve)을 의미합니다.
- AUROC 값은 0.5에서 1 사이의 값을 가지며, 1에 가까울수록 모델이 양성과 음성을 잘 구분한다는 것을 의미합니다.
###### RoC(Receiver Operation Characteristic Curve):
- 거짓긍정비율(FPR)과 참긍정비율(TPR) 간의 관계
- ROC 곡선은 임곗값을 다양하게 조절해 분류 모형의 성능을 비교할 수 있는 그래프로, trade-off 관계인 민감도와 특이도를 기반으로 시각화 한 것이다.
  ![[데이터분석 평가지표__ROC 곡선_AUC개념.png]]
###### AUC(Area Under the Curve):
- ROC 곡선 아래 면적을 AUC(Area Under Curve)라고 하며, AUC가 0.5일 때 분류 능력이 없다고 평가할 수 있고, 면적이 넓을수록, 즉 1에 가까울수록 분류를 잘하는 모형이라 할 수 있다. 
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# 1. 데이터셋 로드 및 분리
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 훈련
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 3. 예측 확률 계산
y_prob = model.predict_proba(X_test)[:, 1]

# 4. ROC 커브와 AUC 계산
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# 5. ROC 커브 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
~~~

![[데이터분석_평가지표_ROC-AUC.png]]


##### AUPR (Area Under the Precision-Recall Curve)
- AUPR는 분류 모델의 성능을 평가하는 또 다른 중요한 지표로, Precision-Recall 곡선 아래의 면적을 의미합니다.
- AUPR 값은 0에서 1 사이의 값을 가지며, 1에 가까울수록 모델이 높은 정밀도와 재현율을 유지한다는 것을 의미합니다.

~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

# 1. 데이터셋 로드 및 분리
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 훈련
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 3. 예측 확률 계산
y_prob = model.predict_proba(X_test)[:, 1]

# 4. Precision-Recall 커브와 AUPR 계산
precision, recall, _ = precision_recall_curve(y_test, y_prob)
aupr = average_precision_score(y_test, y_prob)
print(f"AUPR (Area Under the Precision-Recall Curve): {aupr:.4f}")

# 5. Precision-Recall 커브 시각화
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUPR = {aupr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
~~~

~~~
AUPR (Area Under the Precision-Recall Curve): 0.9987
~~~

![[데이터분석_평가지표_AUPR.png]]




#### [[회귀분석 평가지표|손실함수 (Loss Function)]]
- 예측값과 실제값의 차이를 기반으로 한 지표들을 이용해 회귀 모형의 성능을 평가할 수 있으며, 여기서 목표는 오차를 최소화하는 것이다.
- 예를 들어 미래의 주식 가격 예측, TV 판매량 예측, 비디오 게임 매출액 예측 등이 있다.
##### 회귀문제
###### 평균절대오차(MAE) (Mean Absolute Error)
- 예측값과 실제값의 차이의 절대값을 모두 더한 후, 그 평균을 구한 지표입니다. 오차의 크기에 동일한 비중을 부여합니다.
~~~ python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(df['실제값'], df['예측값'])
~~~~

~~~
3.9294117647058826
~~~
###### 평균제곱오차(MSE) (Mean Squared Error)
- 예측값과 실제값의 차이를 제곱한 후, 모든 데이터에 대해 평균을 구한 지표입니다. 오차의 제곱을 사용하기 때문에, 큰 오차에 더 민감하게 반응합니다.
~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'])
~~~~

~~~
41.762745098039225
~~~

###### 평균제곱근오차(RMSE) (Root Mean Squared Error)
- 평균제곱근오차(MSE)에 루트를 씌운 값이다. 회귀모형의 평가지표로 실무에서도 자주 사용된다.
- MSE 크기를 줄이기 위한 목적으로 사용
~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'], squared=False)
~~~~

~~~
6.462410161699675
~~~

###### 결정계수 $R2$ (R-Squared)
- 주어진 데이터에 회귀선이 얼마나 잘 맞는지, 적합 정도를 평가하는 척도이자 독립변수들이 종속변수를 얼마나 잘 설명하는지 보여주는 지표다.
- 1에 가까울수록 모델이 데이터를 잘 설명하며, 0이면 모델이 데이터를 전혀 설명하지 못함을 의미합니다.
~~~ python
from sklearn.metrics import r2_score
r2_score(df['실제값'], df['예측값'])
~~~~

~~~
0.5145225055729962
~~~

##### 분류 문제
- 분류 문제에서의 손실 함수는 모델의 예측이 실제 값과 얼마나 다른지를 측정하는 데 사용됩니다. 모델의 파라미터를 조정하여 이 손실을 최소화하는 것이 학습의 목표입니다.
###### 크로스 엔트로피 손실 (Cross-Entropy Loss)
- 이 손실 함수는 모델의 예측 확률과 실제 클래스 간의 차이를 측정합니다. 
- 모델이 잘못된 클래스에 높은 확률을 할당할수록 손실이 커지며, 정확한 클래스에 높은 확률을 할당할수록 손실이 작아집니다.
###### 로그 손실 (Log Loss)
- 크로스 엔트로피 손실의 한 형태로, 이진 분류 문제에 주로 사용됩니다. 모델이 예측한 확률과 실제 레이블 간의 차이를 로그를 통해 계산합니다.
~~~ python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# 가상의 이진 분류 데이터셋 생성
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 훈련
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 모델의 예측 확률 계산
y_prob = classifier.predict_proba(X_test)

# 로그 손실 계산 (크로스 엔트로피 손실)
loss = log_loss(y_test, y_prob)

print(f"Log Loss (Cross-Entropy Loss): {loss:.4f}")
~~~

### In-depth Modeling

#### 특성 중요도 (Feature Importance)
##### 정의
- 머신러닝 모델에서 각 입력 특성(피처)이 모델의 예측에 얼마나 기여하는지를 나타내는 지표입니다.
- 즉, 각 특성이 모델 예측에 미치는 영향을 정량화한 값입니다.
- 특성 중요도를 사용하여 데이터에서 불필요하거나 덜 중요한 특성을 제거하여 모델의 복잡성을 줄이고 과적합을 방지할 수 있습니다. 
##### 수행 예시
~~~ python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. 랜덤 포레스트 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. 특성 중요도 계산
importances = model.feature_importances_
print(importances[::-1])  # 중요도 순으로 리버스

# 4. 중요도 시각화
indices = np.argsort(importances)[::-1]
print(indices)

plt.figure()
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(feature_names)[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
~~~

~~~
[0.43606478 0.43612951 0.02167809 0.10612762]
[2 3 0 1]
~~~

#### 하이퍼파라미터 튜닝
##### 하이퍼 파라미터 (Hyperparameter)
###### 정의
- 하이퍼파라미터는 모델의 학습 과정 외부에서 설정되는 값으로, 모델의 구조와 학습 전략에 영향을 미칩니다.
- 하이퍼파라미터는 모델의 성능과 일반화 능력에 중요한 영향을 미치며, 적절히 설정하는 것이 모델의 성능을 극대화하는 데 필수적입니다.
###### 예시
- 딥러닝 모델: 학습률, 배치 크기, 에포크 수, 은닉층의 개수와 크기, 활성화 함수
- 랜덤 포레스트: 트리의 개수(n_estimators), 최대 깊이(max_depth), 최소 샘플 분할(min_samples_split)
- 서포트 벡터 머신(SVM): 정규화 파라미터(C), 커널 함수의 종류, 커널 파라미터(예: RBF 커널의 γ)
###### 탐색 방법
- 수동 서치 (Manual Search) : 하이퍼파라미터의 값을 수동으로 설정하거나 경험적 지식을 바탕으로 특정 값들을 선택하여 모델의 성능을 평가하는 방법입니다.
- 그리드 서치 (Grid Search): 모든 가능한 하이퍼파라미터 조합을 체계적으로 탐색합니다.
- 랜덤 서치 (Random Search): 지정된 범위 내에서 무작위로 하이퍼파라미터 조합을 선택하여 탐색합니다.
- 베이지안 최적화 (Bayesian Optimization): 이전의 탐색 결과를 기반으로 다음 하이퍼파라미터 값을 선택하는 방법입니다.
- 하이퍼밴드 (Hyperband): 다양한 하이퍼파라미터 조합을 시도하면서 자원을 적절히 분배하는 방법입니다.

##### 수동 서치 (Manual Search)
###### 정의
- 수동 서치는 하이퍼파라미터의 값을 수동으로 설정하거나 경험적 지식을 바탕으로 특정 값들을 선택하여 모델의 성능을 평가하는 방법입니다.
- 하이퍼파라미터 조합을 빠르게 조정할 수 있어 계산 자원을 절약할 수 있습니다.
- 반복적인 조정이 필요하며, 비효율적일 수 있어 최적의 조합을 놓칠 가능성이 있습니다.

###### 수행 예시

~~~ python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 여러 하이퍼파라미터 조합을 수동으로 시도
hyperparameters = [
    {'n_estimators': 10, 'max_depth': None},
    {'n_estimators': 50, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': 20}
]

best_accuracy = 0
best_params = None

for params in hyperparameters:
    # 모델 정의 및 학습
    rf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    rf.fit(X_train, y_train)
    
    # 성능 평가
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 최적 하이퍼파라미터 업데이트
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print("최적 하이퍼파라미터:", best_params)
print("최고 정확도:", best_accuracy)
~~~

~~~
최적 하이퍼파라미터: {'n_estimators': 10, 'max_depth': None}
최고 정확도: 1.0
~~~
##### 그리드 서치 (Grid Search)
###### 정의
- 그리드 서치는 정의된 하이퍼파라미터의 모든 가능한 조합을 체계적으로 탐색하여 최적의 하이퍼파라미터를 찾는 방법입니다.
- 모든 조합을 탐색하므로 전역 최적 해를 찾을 가능성이 높습니다.
###### 수행 예시
~~~ python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 하이퍼파라미터 범위 설정
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# 랜덤 포레스트 모델 정의
rf = RandomForestClassifier()

# 그리드 서치 설정
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 성능
print("최적 하이퍼파라미터:", grid_search.best_params_)
print("최고 정확도:", grid_search.best_score_)

# 테스트 데이터에서의 성능 평가
test_score = grid_search.score(X_test, y_test)
print("테스트 데이터에서의 정확도:", test_score)
~~~

~~~
최적 하이퍼파라미터: {'max_depth': None, 'n_estimators': 50}
최고 정확도: 0.9428571428571428
테스트 데이터에서의 정확도: 1.0
~~~


# 출처


# 관련 노트

[[MOC_SKADA 인증 Practitioner_객관식]]
[[MOC_SKADA 인증 Practitioner_주관식]]

# 외부 링크

