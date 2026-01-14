---
Created: 2024-03-05 22:23
tags:
  - AI/머신러닝
---

# 개요

mySuni 를 통하여 CDS (Citizen Data Scientist) 자격 이수를 위해 수강하는 내용이다. 
크게 데이터 처리의 시각화, 그리고 머신러닝 모델링을 수행할 수 있는 역량을 키우기 위한 이론들을 다룬다.

# 내용
## 데이터 처리의 시각화
### 고급 전처리와 피벗 테이블
#### groupby()
- 판다스(Pandas) .groupby()로 할 수 있는 거의 모든 것!
	- https://teddylee777.github.io/pandas/pandas-groupby/

## 머신러닝 모델링
- 교육자료 : ![[mySuni_CDS_머신러닝 교재.pdf]]
### 머신러닝 개요
#### 머신러닝과 딥러닝의 차이

##### [[머신러닝]]
###### 정의
- 인공지능의 한 분야로 컴퓨터가 학습할 수 있도록 알고리즘과 기술을 개발하는 분야
###### 필요성
- 컴퓨터 성능 향상 및 빅데이터로 인한 새로운 비지니스 창출 가능
###### 문제유형
- 알고리즘 부재로 명시적 문제 해결이 불가능한 문제
- 프로그래밍이 어려운 문제 (예: 음성 인식)
- 지속적으로 변화하는 문제 (예: 자율 주행)
##### [[딥러닝]]
###### 정의
- 여러 층을 가진 인공신경망을 사용하는 머신러닝 방법의 일종
- 머신러닝의 경우 특징 추출을 사람이 하는 반면, 딥려닝은 자동으로 특징을 추출함

###### 머신러닝과 다른점
- 기존 머신러닝에서 수행가능한 분류, 회귀 뿐만 아니라 물체검증, 영상분할, 생성 등 다양한 작업을 할 수 있어 미래 기술로 각광받고 있음

#### [[데이터 분석 모델링 프로세스|머신러닝 프로세스]]

![[머신러닝 프로세스.png]]

#### [[데이터 분석 학습 유형|머신러닝 학습 방법]]

##### [[지도학습]]
- 주어진 데이터에 대한 결과가 있고, 이를 바탕으로 새로운 데이터 추정
- 회귀, 분류에 적합
##### [[비지도학습]]
- 주어진 데이터에 대한 결과가 없어, 데이터의 패턴, 특성, 구조를 찾아서 학습
- 군집 문제에 사용
##### 강화학습
 - 에이전트의 동작이 적절한지에 대한 피드백을 반영하면서 학습
 - 현재의 상태를 인식하여 보상이 최대화되는 행동을 수행

![[머신러닝 학습 유형 (지도학습, 비지도학습).png]]

#### [[분석모델 성능 평가|머신러닝 모델 평가]]

##### [[분류분석 평가지표]]
###### [[혼동행렬(Confusion Matrix)]]
- 예측값이 실제값과 일치하는지 분류하는 분류표![[데이터분석 평가지표_혼동행렬_개념.png]]
  
##### [[회귀분석 평가지표]]
- R-Squared
	-  sklearn.metrics.r2_score
	- 주어진 데이터에 회귀선이 얼마나 잘 맞는지, 적합 정도를 평가하는 척도이자 독립변수들이 종속변수를 얼마나 잘 설명하는지 보여주는 지표다.
~~~ python
from sklearn.metrics import r2_score
r2_score(df['실제값'], df['예측값'])
~~~~

~~~
0.5145225055729962
~~~

- MAE (Mean Absolute Error)
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

- MSE (Mean Squared Error)
	- sklearn.metrics.mean_squared_error(squared=True)
	- 실제값과 예측값의 차이를 제곱해서 평균 계산
	- 큰 오차를 더 크게, 작은 오차는 더 작게 평가하여 이상치에 민감

~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'])
~~~~

~~~
41.762745098039225
~~~


- RMSE (Rood Mean Squared Error)
	- sklearn.metrics.mean_squared_error(squared=False)
	- MSE 크기를 줄이기 위한 목적으로 사용
~~~ python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['실제값'], df['예측값'], squared=False)
~~~~

~~~
6.462410161699675
~~~


#### 데이터 분할
~~~python
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, stratify=Y, random_state=0) # stratify : 층화 분리 적용 (범주별 데이터를 최대한 비슷 비율로 분리)
print('X Train Shape', x_train.shape)
print('X Test Shape', x_test.shape)
print('Y Train Shape', y_train.shape)
print('Y Test Shape', y_test.shape)
~~~
~~~
X Train Shape (112, 4)
X Test Shape (38, 4)
Y Train Shape (112,)
Y Test Shape (38,)
~~~

### [[지도학습]]

##### [[KNN (K-Neareat Neighbor)|최근접이웃 (K-Neareat Neighbor))]]

##### 표준화 및 정규화 (Scaling)
- 특정 알고리즘은 데이터의 스케일(관측 범위)에 많은 영향이 있어 관측 범위를 일정하게 맞춤
- Standardization(표준화), Normalization(정규화) 방법이 존재
###### [[정규화 (Normalization)]]
###### [[표준화 (Standardization)]]

##### 선형 회귀
###### [[단순 선형 회귀분석]]

###### [[다중 선형 회귀분석]]

##### [[다항 회귀 (Polynomial Regression)]]

##### 과대적합 및 과소적합
###### 과소적합(Underfitting)
- 정보를 충분히 학습하지 못해 설명력이 떨어지는 상태
###### 과대적합(Overfitting)
- 정보를 과도하게 학습해 일반화된 설명력이 떨어지는 상태
##### 정칙화와 규제모델
###### 정칙화 (Regularization)
- 정의
	- 다항회귀 시 차수가 높아 질수록 과대적합이 발생하여 예측 성능이 저하되는 문제 발생
	- 과대 적합 박지를 위해 가중치를 크게 제어
	- 기존의 비용 함수에 규제(Penalty) 항을 추가하여 가중치의 크기를 제어 가능
- 규제 강도 (Alpha)
	- Alpha의 값을 아주 작게 하거나 0으로 지정하는 경우 가중치에 대한 규제가 약해지며 과대적합 발생
	- Alpha의 값을 너무 크게 지정하는 경우 가중치에 대한 규제가 강해지며 너무 모델이 단순하여 과소적합 발생
	  ![[지도학습 - 정칙화 (규제강도에 따른 과대,과소적합).png]]
###### 규제 모델 - [[Lasso]]
- 정의
	- 기존 선형회귀식에 규제항(L1 Regularization)이 적용된 모델
	- MSE가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 절댓값 합이 최소가 되도록 적절한 가중치와 편향을 찾음
	- L1규제의 효과로 어떤 특성들은 0이 되어서 모델을 만들 때 사용되지 않음 
	- 모델에서 가장 중요한 특성이 무엇인지 알게 되어 모델 해석력이 좋아짐


~~~ python
# 규제 모델 생성
from sklearn.linear_model import Lasso
model = Lasso(alpha=1) 

# 모델 학습
model.fit(x_train, y_train) # 학습용 데이터만 사용

# 모델 평가 (R Squared)
print('학습 데이터 성능 :', model.score(x_train, y_train))
print('평가 데이터 성능 :', model.score(x_test, y_test))
~~~

~~~
학습 데이터 성능 : 0.41412544493966097
평가 데이터 성능 : 0.27817828862078753
~~~

~~~ python
# 사용된 계수 개수 조회
print('전체 계수 :', len(model.coef_))
print('사용된 계수 : ', len(model.coef_[model.coef_ != 0]))
~~~

~~~
전체 계수 : 285
사용된 계수 :  2.     # 강한 규제로 인해 과소적합 발생
~~~

###### 규제모델 - [[Ridge]]
- 정의
	- 기존 선형회귀식에 규제항(L2 Regularization)이 적용된 모델
	- MSE가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 제곱 합이 최소가 되도록 적절한 가중치와 편향을 찾음
	- L2규제의 효과로 0이 되진 않지만 모델 복잡도를 제어할 수 있음
	- 특성들이 출력에 미치는 영향력이 줄어듦(현재 특성들에 덜 의존)

~~~ python
# 규제 모델 생성
from sklearn.linear_model import Ridge
model = Ridge(alpha=1) 

# 모델 학습
model.fit(x_train, y_train) # 학습용 데이터만 사용

# 모델 평가 (R Squared)
print('학습 데이터 성능 :', model.score(x_train, y_train))
print('평가 데이터 성능 :', model.score(x_test, y_test))
~~~

~~~
학습 데이터 성능 : 0.46333427173027797
평가 데이터 성능 : 0.3571528311583685
~~~


~~~ python
# 규제 강도 완화 
model = Ridge(alpha=0.1) 
model.fit(x_train, y_train) 
print('학습 데이터 성능 :', model.score(x_train, y_train))
print('평가 데이터 성능 :', model.score(x_test, y_test))
~~~

~~~
학습 데이터 성능 : 0.5547492878993845
평가 데이터 성능 : 0.3715849639261286
~~~


##### [[로지스틱 회귀분석|로지스틱 회귀]]
###### 로짓 변환
- 선형 회귀 모델은 특성의 값이 증가할수록 레이블의 값도 증가/감소 하는 형태를 가짐 
- 범주형 레이블을 가지는 경우 특성의 값에 상관없이 0또는 1의 값을 가지는 것을 알 수 있음 
- 분류 모델에서는 선형 모델과 달리 회귀 계수를 통한 직관적 해석이 어려움
- 로짓 변환을 통해 비선형 형태를 선형 형태로 만들어 회귀 계수의 의미를 해석하기 쉽게 할 수 있음
- 로짓 : 오즈에 로그를 씌운 값
- 로짓 변환 : 오즈에 로그를 씌어 변환하는 과정

![[로지스틱 회귀 - 로짓 변환.png]]

###### 오즈 (Odds)
- 실패확률에 대한 성공확률의 비율
	- 오즈(Odds)가 3이라면 성공 확률이 실패 확률의 3배라는 의미 

#### [[서포트 벡터 머신(SVM)]]

#### [[의사결정 트리 분석|의사결정나무]]
#### [[앙상블 분석]]

#### 하이퍼 파라미터 최적화

- 모델에 가장 적합한 하이퍼 파라미터(Hyper Parameter)를 찾기 위해서 사용
    - Hyper Parameter 는 모델의 외적인 요소로 사용자에 의해 결정되는 값
        - 예) 규제 강도, 트리 최대 깊이 등
    - Parameter는 모델의 내적인 요소로 학습을 통해 결정되는 값
        - 예) 회귀 모델의 가중치, 트리 모델의 특성 중요도 등

- 값을 어떻게 설정하느냐에 따라 모델의 성능을 개선시킬 수도, 저하시킬 수도 있음

- Objective Function을 optimize 하는 방향으로 범위 및 간격을 설정
    - Objective Function : 최대화(점수)하거나 최소화(Loss, Cost)해야 하는 값
    - Search Boundary : 탐색 범위 설정
    - Step : 탐색 시 간격
##### GridSearch
- `sklearn.model_selection.GridSearchCV`
- 간단하고 광범위하게 사용되는 hyperparameter 탐색 알고리즘
- 해당 범위 및 Step의 모든 경우의 수를 탐색
- 범위를 넓게 가져갈수록, Step을 작게 설정할 수록 최적해를 찾을 가능성이 높이지지만 시간이 오래 걸림
- 일반적으로 넓은 범위와 큰 Step으로 설정한 후 범위를 좁혀 나가는 방식을 사용하여 시간을 단축
##### RandomSearch

- `sklearn.model_selection.RandomizedSearchCV`
- 정해진 범위 내에서 Random하게 선택 
- 기본적으로는 더 빠르고 효율적이기 때문에 GridSearch보다 권장되는 방법 
- Grid Serach보다 속도가 빠르지만 optimzed solution이 아닐 수 있음 
- Sample의 수가 많다면 Random Sampling을 통해 최적해를 찾을 가능성이 높아짐

##### Bayesian Optimization

- GridSearch와 RandomSearch는 각 Search가 종속되어 있지 않음
    - 서로 간 정보를 사용하지 않음
- 따라서 찾아낸 값이 최적의 값이라고 생각할 수 없음

- Gausain Process 통계학을 기반으로 만들어진 모델로 여러개의 하이퍼 파라미터들에 대해서 Aqusition Fucntion을 적용했을 때 가장 큰 값이 나올 확률이 높은 지점을 찾음

- 목적함수의 '형태'를 학습
    - Prior Distribution에 기반하여 하나의 탐색 함수 가정
        - Exploartion 
            - 매번 새로운 Sampling을 사용해 목적함수를 Test할 시, 해당 정보를 사용하여 새로운 목적함수의 Prior Distribution을 update
        - Exploitation 
            - Posterior distribution에 의해 얻은 global minimum이 나타날 가능성이 높은 위치에서 알고리즘을 테스트

### 비지도학습

#### [[군집분석]]


# 출처


# 관련 노트
[[데이터분석2_테스트 내용]]

# 외부 링크

