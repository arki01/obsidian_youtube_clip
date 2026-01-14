---
Created: 2023-10-08 13:52
tags:
  - AI/머신러닝
aliases:
  - SVM
---

# 개요
분류와 회귀 분석에 사용되는 [[지도학습]] 알고리즘으로, n-1차원의 초평면으로 분리하여 데이터의 분류를 진행한다.

# 내용
#### 개념
- 분류와 회귀분석에 사용되는 지도학습 알고리즘
- 서포트 벡터 머신에서는 데이터가 n차원으로 주어졌을때 이러한 데이터를 n-1차원의 초평면으로 분리한다.
- 결정 경계와 가장 가까운 데이터를 '서포트 벡터'라고 하며, 서포트 벡터와 결정 경계 사이를 '마진'이라고 한다.
- 데이터가 어느 카테고리에 속할지 판단하는 이진 선형 분류 모델을 만드는 기법으로 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 그 중 가장 큰 폭을 가지는 경계를 찾는 알고리즘이다.
- ![[서포트 벡터의 구성요소.png]]
##### 하드 마진
- 모든 샘플이 마진 바깥쪽에 올바르게 분류되어 있다면 이를 하드마진 분류라고 함
- 하드마진은 선형적으로 구분될 수 있어야 제대로 동작하며 이상치에 민감함
##### 소프트 마진
- 하드 마진의 단점을 극복하기 위해 최대의 마진을 가지는 동시에 여유변수의 합을 최소로 하는 초평면을 찾는 것
- 하이퍼 파라미터인 C값을 조절하여 모델의 적절한 균형 규제
##### 커널 기법
- 선형으로 분리되지 않는 데이터를 저차원에서 고차원으로 매핑하여 해결할 수 있다.
	- 현실세계에서 선형으로 완벽히 구분되는 데이터는 드물다.

#### 알고리즘
- 개요
	- 두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때, SVM 알고리즘은 주어진 데이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 이진 선형 분류모델을 만든다.
	- 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데, ==SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘==이다.
- 특징
	- 커널 트릭을 사용함으로써 다양한 데어터의 특성을 분류할 수 있다.
	- 비교적 적은 학습 데이터로도 정확도가 높은 분류를 기대할 수 있다.
#### 분석 수행
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
sv = svm.SVC(kernel="rbf")    # 분류 모델
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
#### SVM 커널 파라미터 조정
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

# 출처


# 관련 노트


# 외부 링크

