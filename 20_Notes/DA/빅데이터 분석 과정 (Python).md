---
Created: 2023-10-28 13:08
tags:
  - AI/머신러닝
---

# 개요
빅데이터 분석을 위한 실질적인 과정을 정리하였으며 순서는 아래와 같다.

1. 필요 패키지 임포트
2. 데이터 불러오기
3. 데이터 살펴보기
4. 데이터 전처리
5. 분석 데이터셋 준비
6. 데이터 분석 수행 
7. 성능평가 및 시각화

# 내용

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
	- 지도학습 - 분류 : [[의사결정 트리 분석|의사결정나무]](분류), KNN, [[서포트 벡터 머신(SVM)]], [[로지스틱 회귀분석]], [[랜덤 포레스트]], 인공신경망
	- 지도학습- 회귀(예측) : [[단순 선형 회귀분석|선형회귀분석]], [[다중 선형 회귀분석|다중회귀분석]], [[의사결정 트리 분석|의사결정나무]](회귀)
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

# 출처


# 관련 노트


# 외부 링크

