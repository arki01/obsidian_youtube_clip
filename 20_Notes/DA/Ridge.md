---
Created: 2024-05-09 22:45
tags:
---

# 개요
회귀분석의 규제모델 중 하나로 기존 선형회귀식에 규제항(L2 Regularization)이 적용된 모델

# 내용

### 정의
- 기존 선형회귀식에 규제항(L2 Regularization)이 적용된 모델
- MSE가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 제곱 합이 최소가 되도록 적절한 가중치와 편향을 찾음
- L2규제의 효과로 0이 되진 않지만 모델 복잡도를 제어할 수 있음
- 특성들이 출력에 미치는 영향력이 줄어듦(현재 특성들에 덜 의존)

### 알고리즘

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


# 출처


# 관련 노트


# 외부 링크

