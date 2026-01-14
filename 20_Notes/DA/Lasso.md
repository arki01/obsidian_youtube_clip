---
Created: 2024-05-09 22:41
tags:
---

# 개요
회귀분석의 규제모델 중 하나로 기존 선형회귀식에 규제항(L1 Regularization)이 적용된 모델

# 내용

### 정의
- 기존 선형회귀식에 규제항(L1 Regularization)이 적용된 모델
- MSE가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 절댓값 합이 최소가 되도록 적절한 가중치와 편향을 찾음
- L1규제의 효과로 어떤 특성들은 0이 되어서 모델을 만들 때 사용되지 않음 
- 모델에서 가장 중요한 특성이 무엇인지 알게 되어 모델 해석력이 좋아짐

### 알고리즘 
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

# 출처


# 관련 노트


# 외부 링크

