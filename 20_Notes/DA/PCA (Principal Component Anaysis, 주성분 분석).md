---
Created: 2024-08-18 13:29
tags: 
aliases:
  - PCA
---

# 개요
여러 변수들의 변량을 '주성분'이라고 불리는, 서로 상관성이 높은 여러 변수들의 선형 조합으로 새로운 변수들로 요약, 축소하는 기법이다. 

# 내용

### 설명
 - 변수간의 스케일 차이가 나면 스케일이 큰 변수가 주성분에 영향을 많이 주기 때문에 주성분 분석 전에 표준화나 정규화를 시켜준다.
- 사이킷런의 PCA를 이용해서 쉽게 주성분 분석을 수행할 수 있다.

![[빅데이터 주성분 분석(PCA)을 통한 차원축소.png]]

## 분석 방법

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

- n_component는 PCA로 변환할 차원의 수를 의미한다.
- 첫번째 PCA 변환요소(차원)만으로 전체 변동성의 73%를 설명이 가능하다. 두번째 요소(차원)는 22.8%를 차지한다.
- 따라서, 2개 요소(차원)로만 변환해도 원본 데이터의 변동성을 95.8% 설명이 가능하므로, 변수를 4개에서 2개로 줄일 수 있다.
- 
# 출처


# 관련 노트


# 외부 링크

