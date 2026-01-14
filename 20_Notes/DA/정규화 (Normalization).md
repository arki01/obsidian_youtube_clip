---
Created: 2023-10-03 21:41
tags:
  - AI/머신러닝
aliases:
  - 정규화
---

# 개요
- 데이터의 이상값을 특정 범위 내로 변환하는 방법으로 변수의 크기가 너무 크거나 작은 경우 범주를 일치화 시키는 방법이다.
- 최소 최대 정규화를 많이 이용하며, 0~1사이 데이터로 변환
# 내용
대표적인 스케일 방법이는 최소-최대 정규화와 Z-Score 정규화가 있다.

### 방법

#### 최소-최대 정규화(Min-Max Normalization)

##### 정의
- 모든 특성이 0과 1사이에 위치하도록 데이터를 변환한다.
- 스케일이 다른 두 변수를 Min-Max 변환하면 상호간에 비교가 가능하다.

##### 분석 방법
1. MinMaxScaler() 함수를 사용해서 표준화한다
~~~ python
# MinMaxScaler() 함수 이용
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
meat_consumption["한국인_mm"] = scaler.fit_transform(meat_consumption[["한국인"]])
scaler = MinMaxScaler()
meat_consumption["일본인_mm"] = scaler.fit_transform(meat_consumption[["일본인"]])

meat_consumption[["한국인", "일본인", "한국인_mm", "일본인_mm"]].head()
~~~

2. 직접 식을 입력하여 정규화한다.

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

#### Z-Score 정규화(Z-Score Standardization)

##### 정의
- 기존 특성을 평균이 0, 분산이 1인 정규분포로 변환하여 특성의 스케일을 맞춘다.
  ![standard](https://wikimedia.org/api/rest_v1/media/math/render/svg/b0aa2e7d203db1526c577192f2d9102b718eafd5)

##### 분석 방법
1. zscore() 함수를 사용한 표준화
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
2. 직접 수식을 사용한 정규화
~~~ python
# 직접수식 입력을 통한 Z-표준화 
meat_consumption["한국인_정규화2"] = (meat_consumption_korean - np.mean(meat_consumption_korean))/np.std(meat_consumption_korean)     
meat_consumption["일본인_정규화2"] = (meat_consumption_japan - np.mean(meat_consumption_japan))/np.std(meat_consumption_japan)
~~~

#### RobustScaler
- 중앙값이 0, IQR이 1이 되게 변환한다.

### 예제

#### 수치형 변수 변환하기

주어진 데이터에서 'f5'컬럼을 표준화(Standardization (Z-score Normalization))하고 그 중앙값을 구하시오

~~~ PYTHON
# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
df.head(2)
~~~

~~~
	id	age	city	f1	f2	f3	f4	f5
0	id01	2.0	서울	NaN	0	NaN	ENFJ	91.297791
1	id02	9.0	서울	70.0	1	NaN	ENFJ	60.339826
~~~

~~~ python
# 표준화
# 평균이 0, 분산이 1인 정규분포로 변환하여 특성의 스케일을 맞춘다.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['f5'] = scaler.fit_transform(df[['f5']])

df.head()
~~~

~~~
	id	age	city	f1	f2	f3	f4	f5
0	id01	2.0	서울	NaN	0	NaN	ENFJ	1.220815
1	id02	9.0	서울	70.0	1	NaN	ENFJ	0.127343
2	id03	27.0	서울	61.0	1	NaN	ISTJ	-1.394535
3	id04	75.0	서울	NaN	2	NaN	INFP	-0.143667
4	id05	24.0	서울	85.0	2	NaN	ISFJ	-0.970085
~~~

~~~ python
# 중앙값 출력
print(df['f5'].median())    
~~~

~~~
0.260619629559015
~~~

# 출처


# 관련 노트


# 외부 링크

표준화 vs 정규화 : https://sungwookoo.tistory.com/35