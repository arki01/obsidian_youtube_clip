---
Created: 2024-04-20 20:01
tags:
  - AI/머신러닝
aliases:
  - 표준화
---

# 개요
- 서로 다른 분포를 비교하기 위해 표준에 맞게 통일시키는 방법
- 평균을 0, 표준 편차를 1로 변환 (-1 ~ 1 사이의 데이터로 변환)

# 내용

##### 정의
- 수치로 된 값들을 여러개 사용할 때 각 수치의 범위가 다르면 이를 같은 범위로 변환해서 사용하는데 이를 일반 정규화라고 한다.
> 두 과목의 시험을 가지고 평가하는데 과목 A의 시험은 10점 만점이고, 다른 과목 B는 50점 만점이라고 하자.
> 만약에 A에서는 8점, B에서는 20점을 받았을 때, 이것을 정규화하면 8/10=0.8점, 20/50=0.4점이되고 평점은 0.6이 된다.

##### 분석 방법
1. 표준 정규분포를 갖는 데이터를 생성하고, 이를 데이터 프레임으로 변환한다.
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
2. 시각화를 통해서 히스토그램에 의한 파악을 수행해본다.
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

3. 사이킷런 스케일러를 활용한 표준화
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


# 출처


# 관련 노트


# 외부 링크

표준화 vs 정규화 : https://sungwookoo.tistory.com/35