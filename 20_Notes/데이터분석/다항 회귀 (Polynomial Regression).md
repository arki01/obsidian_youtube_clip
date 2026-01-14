---
Created: 2024-04-20 22:32
tags: 
aliases:
  - 다항 회귀
---

# 개요
- 역속형 레이블과 특성 간 관계가 선형이 아닌 경우 1차식으로 표현이 불가능하여 2차식 이상의 다항식으로 변형하여 회귀 분석을 진행

# 내용

### 알고리즘
- 기존의 특성에 다항 변환을 적용하여 새로운 특성을 추가
- scikit-learn의 경우 PolynomialFeatures를 사용하여 다항 변환 및 상호교차항을 쉽게 추가 가능

~~~python
# 다항 변환을 위한 데이터 생성
import pandas as pd
X = pd.DataFrame({
    'X':[1, 2, 3, 4, 5]
})

# 다항 변환 적용 (2차항)
from sklearn.preprocessing import PolynomialFeatures
P = PolynomialFeatures(degree=2)     # 2차 항으로 정의
X_POLY = pd.DataFrame(P.fit_transform(X), columns=P.get_feature_names_out())   # 새로운 컬럼 정의
~~~


# 출처


# 관련 노트


# 외부 링크

