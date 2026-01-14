---
Created: 2023-10-28 14:04
tags:
  - AI/머신러닝
aliases:
  - KNN
  - 최근접이웃 (K-Neareat Neighbor))
---

# 개요
[[지도학습]]의 한 종류로, 데이터로부터 거리가 가까운 K개의 다른 데이터의 정답(목표값)을 참조하여 분류하는 방법이다.
# 내용

### 알고리즘
- 개요
	- [[지도학습]]의 한 종류로, 정답이 있는 데이터를 사용하여 분류 작업을 한다.
	- 서로 가까운 점들은 유사하다는 가정하여, 데이터로부터 거리가 가까운 K개의 다른 데이터의 정답(목표값)을 참조하여 분류한다.
- 특징
	- 거리 기반 연산으로, 숫자에 구분된 속성에 우수한 성능을 보인다.
	- 전체 데이터와의 거리를 계산하기 때문에 차원(벡터)의 크기가 크면 계산량이 많아진다.
	- 회귀의 경우 종속변수의  평균값으로 예측하며, 분류의 경우 종속변수의 과반범주로 예측한다.
	- 이상치에 크게 영향을 받지 않는다.
	- KNN은 거리 측정 방법에 따라 결과가 크게 달라지며, 주로 유클리드 거리를 주로 사용
- 모델
	- 분류모델 : sklearn.neighbors.KNeighborsClassifier
	- 회귀모델 : sklearn.neighbors.KNeighborsRegressor

~~~ python
import sklearn
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(
	 n_neighbors = 5,       ## 최근접 이웃 수
	 weight = 'uniform',    ## 가중치 함수 (distance 사용시 거리 가중치 적용)
	 p = 2	                ## Minkowski 거리 지수 (1 = 맨하탄, 2 = 유클리드)
)
from sklearn.neighbors import KNeighborsRegressor
KNeighborsRegressor(
	 n_neighbors = 5,       ## 최근접 이웃 수
	 weight = 'uniform',    ## 가중치 함수 (distance 사용시 거리 가중치 적용)
	 p = 2	                ## Minkowski 거리 지수 (1 = 맨하탄, 2 = 유클리드)
)
~~~

### 분석 수행
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
	- n_negihbors 값은 3부터 시갖해본다
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
print("\n")
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
print("\n")
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



# 출처


# 관련 노트


# 외부 링크

