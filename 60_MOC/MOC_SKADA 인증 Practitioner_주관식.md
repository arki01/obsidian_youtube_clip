---
Created: 2024-08-18 13:12
tags: 
share_link: https://share.note.sx/fcnaeryy#tiKijJM3wtxCXTgrTnDOrWHY4E/8JetIrWK9XHkWuNI
share_updated: 2024-08-19T13:19:24+09:00
---

# 개요

mySuni 를 통하여 SKADA 자격 이수를 위해 수강하는 내용이다. 
프로젝트를 기반으로 한 EDA, 전처리, 모델 적용 및 고도화, 심화 과정으로 구성되어 있으며, 현업에서 실제 응용하여 활용할 수 있는 수준의 데이트 분석 이론들을 다룬다.

---
# 내용

### 인증소개
![[SKADA 인증시험 상세소개.pdf]]
### 강의자료
![[mySuni_SKADA 인증 Practitioner 시험 강의자료.pdf]]



### Data Ingestion, EDA
#### Data Loading
##### 수행
- 조건 1. 세 종류의 csv 파일을 pandas dataframe 형식으로 읽어온다.
- 조건 2. 불러온 파일은 딕셔너리 변수인 data에 저장한다.
- 조건 3. 해당 파일의 이름(.csv는 제외)을 key로 사용하여 data에 저장한다. (예를 들어, TripA01.csv 파일을 TripA01.csv인 경우 data['TripA01']로 저장)
~~~ python
data_root = '/mnt/elice/dataset/data'
data = dict()

filenames = ['metadata']
for idx in range (1,33):
    filenames.append('TripA{:02d}'.format(idx))
for idx in range (1,38):
    filenames.append('TripB{:02d}'.format(idx))
    
for fn in filenames:
    data[fn] = pd.read_csv(os.path.join(data_root, '{}.csv'.format(fn)))
    # data[fn] = pd.read_csv('/mnt/elice/dataset/data/{}.csv'.format(fn)) 로 정의해도 무방함
~~~

#### Feature Selection
##### 수행
- 조건 1. 모든 주행기록(이름이 Trip으로 시작하는 파일)에서 공통으로 존재하는 피쳐만 사용한다.
~~~ python
# 'TripA01' 의 Feature를 추출
selected_cols = set(data['TripA01'].columns.tolist())  

# set을 활용하여 순차대로 모든 Feature와 겹치는 column만 남기기  
for key in filenames[1:]:
    selected_cols = set.intersection(selected_cols, set(data[key].columns.tolist()))

# 추출한 Feature들에 대해서 data 에 덮어쓰기
for key in filenames[1:]:
    data[key] = data[key][selected_cols]
~~~


### Data Processing

#### 결측치 처리
##### 수행
- 조건 1. 각 주행 기록에서 결측치의 개수가 5개 이상인 경우가 있을 시, 그 주행 기록은 분석 대상에서 제외한다.
- 조건 2. 특정 부분의 결측치가 존재할 시에는 바로 직전 시간의 관측값으로 대체한다.
- 조건 3. 바로 직전의 관측값이 없을 경우, 바로 직후의 관측값으로 대체한다.

~~~ python
removed_keys = set()
for key in filenames[1:]:
    for col in data[key].columns:
        if data[key][col].isnull().values.sum() >= 5:
            removed_keys.add(key)
        else:
	        # 특정 부분의 결측치가 존재할 시에 전/후 관측값 대체
            data[key][col] = data[key][col].fillna(method='ffill').fillna(method='bfill')         

# data에서 결측치의 개수가 5개 이상인 주행기록 제거
for key in removed_keys:
    data.pop(key)
                                                                          
selected_keys = sorted(list(set(filenames[1:]) - removed_keys))  
~~~

#### [[PCA (Principal Component Anaysis, 주성분 분석)]]
##### 이론
- PCA는 고차원 데이터에서 주요한 정보를 유지하면서 차원을 줄이는 기법으로, 데이터의 패턴을 더 쉽게 이해하거나, 계산 효율성을 높이기 위해 사용된다.
- Feature의 수를 줄이기 위해 PCA를 적용하여 데이터의 차원을 축소한다.
  ![[빅데이터 주성분 분석(PCA)을 통한 차원축소.png]]
##### 수행
- n_component는 데이터의 칼럼 수로 입력

~~~ python
from sklearn.decomposition import PCA
pca = PCA(n_components=378)
pca.fit(x_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)  # 최적의 차원을 찾기 위한 누적합
d = np.argmax(cumsum >= 0.95) + 1 # 95% 이상 유지하는 차원의 수 (index는 0부터 시작하므로 +1)
print(d)

pca = PCA(n_components=154)  # 차원 축소수 적용
x_train_pca = pca.fit_transform(x_train)  # 차원 축소 수행
x_test_pca = pca.transform(x_test)  
~~~


#### 로그 변환
##### 수행
~~~ python
import pandas as pd
import numpy as np
from scipy.stats import skew

# 데이터 생성
np.random.seed(42)
data = {
    f'feature_{i}': np.random.normal(loc=0, scale=1, size=100) for i in range(1, 11)
}

# 일부 feature에 skewness를 추가하기 위해 비정상적인 값을 추가
data['feature_1'] = np.random.exponential(scale=2, size=100)  # 양의 skewness
data['feature_2'] = np.random.normal(loc=5, scale=0.5, size=100)  # 약간의 음의 skewness
data['feature_3'] = np.concatenate([np.random.normal(loc=0, scale=1, size=95), np.random.normal(loc=10, scale=1, size=5)])  # 강한 양의 skewness
data['feature_4'] = np.concatenate([np.random.normal(loc=-10, scale=0.5, size=95), np.random.normal(loc=0, scale=1, size=5)])  # 강한 음의 skewness

# 비정상적인 데이터를 추가하여 skewness 절대값이 10을 넘는 feature 생성
data['feature_5'] = np.concatenate([np.random.normal(loc=0, scale=1, size=90), np.random.normal(loc=50, scale=1, size=10)])  # 매우 큰 양의 skewness

# 데이터프레임 생성
x_high_corr = pd.DataFrame(data)

# 조건1. x_high_corr 에서 skeness의 절댓값이 1을 초과하는 feature를 찾으시오.
skewness_values = x_high_corr.skew()
hige_skewness_features = skewness_values[abs(skewness_values)>1].index

# 조건2. 찾은 feature 각각에 1e-10을 더한 후 자연 log 변환을 적용하시오.
for feature in hige_skewness_features:
  x_high_corr[feature] = np.log(x_high_corr[feature] + 1e-10)

# 조건3. 변환된 데이터를 x_log 변수에 저장하시오.
x_log = x_high_corr.copy()

skew_comparison = pd.concat([skewness_values, x_log.skew()], axis=1)
print(skew_comparison)
~~~

~~~
0	1
feature_1	1.751433	-0.558720
feature_2	-0.269298	-0.269298
feature_3	3.213662	0.025040
feature_4	4.030479	NaN
feature_5	2.690219	0.812392
feature_6	-0.034849	-0.034849
feature_7	-0.022612	-0.022612
feature_8	0.145468	0.145468
feature_9	0.397603	0.397603
feature_10	-0.164581	-0.164581
~~~
### Data Visualization
#### 산점도 (Scatter)
##### 수행
- 원본 데이터의 분산에 대한 정보를 가장 많이 갖고 있는 두개의 주성분 벡터를 이용하여 2차원 시각화를 진행한다.
~~~ python
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train["Pass/Fail"])
~~~
![[Pasted image 20240818140959.png]]

#### 박스 플롯 (Box Plot)
##### 수행
~~~~ python
import matplotlib.pyplot as plt
import pandas as pd

summer_r = data['TripA01'].iloc[:,2]
winter_r = data['TripA02'].iloc[:,2]

# Plot the boxplot
plt.boxplot([summer_r, winter_r], labels=['여름', '겨울'])
plt.xlabel('계절')
plt.ylabel('배터리효율')
plt.show()
~~~~

![[Pasted image 20240901121215.png]]

#### SMOTE (Synthetic Minority Over-sampling Technique, 오버샘플링)
##### 이론 
- 데이터 불균형은 모델의 성능에 부정적인 영향을 줄 수 있다. SMOTE(Synthetic Minority Over-sampling Technique)는 불균형한 클래스 분포를 가진 데이터셋에서 소수 클래스의 샘플을 합성하여 데이터 불균형 문제를 완화하는 방법이다.
  ![[데이터분석_샘플링_SMOTE.png|400]]
  ![[데이터분석_샘플링_SMOTE2.png]]

##### 계산과정
1. Minority Class Sample 선택: 소수 클래스에서 랜덤하게 하나의 인스턴스를 선택한다.
2. k-Nearest Neighbors 계산: 선택된 인스턴스와 가장 가까운 거리에 있는 k개의 소수 클래스 샘플을 찾는다.
3. Synthetic Sample 생성: 선택된 인스턴스와 k개의 이웃 중 하나 사이의 선형 보간을 통하여 새로운 인스턴스를 생성한다.

##### 수행
~~~ python
print(y_train.value_counts())

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(x_train_pca, y_train)   # 데이터 불균형 처리를 위한 오버샘플링 수행
print(y_resampled.value_counts())
~~~

~~~
Pass/Fail
0            1170
1             139

Pass/Fail
0            1170
1            1170
~~~

### Improving Performance


#### GridsearchCV (하이퍼 파라미터 탐색)
##### 이론
- 하이퍼파라미터는 모델의 성능에 큰 영향을 줄 수 있으며, Grid Search는 가능한 모든 조합을 시도하여 최적의 하이퍼파라미터를 찾는 방법이다.
##### 수행
~~~ python
model = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200)
model.fit(x_resampled, y_resampled)
y_pred = model.predict(x_test_pca)

param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'gamma': [0.1, 0.5]
}  # XGBClassifier의 하이퍼파라미터 범위(후보군) 정의

# cv는 3-fold cross validation을 사용한다.
# 하이퍼파라미터 scoring 기준은 ROC-AUC가 높은 값을 기준으로 한다.
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc') 
grid_search.fit(x_resampled, y_resampled)

print("\nBest Hyperparameters:", grid_search.best_params_)
~~~

~~~
Best Hyperparameters: {'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
~~~

#### 하이퍼 파라미터 튜닝
##### 수행
- 조건 1. Grid Search 방식을 사용하시오.
- 조건 2. GradientBoostingRegressor의 max_depth, learning_rate, n_estimators를 튜닝하시오.
- 조건 3. 하이퍼파라미터의 후보들은 아래에 정의된 MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS를 활용하시오.
- 조건 4. GradientBoostingRegressor의 random_state는 항상 3064로 고정하시오.

~~~ python
seed = 3064
MAX_DEPTH = [2, 3, 4]
LEARNING_RATE = [0.1, 0.05, 0.01]
N_ESTIMATORS = [60, 80, 100, 120]

best_hyperparameters = None
best_model = None
best_mae = 10000000

# 하이퍼 파라미터 조합 생성
hyperparameters = list(itertools.product(MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS))

for max_depth, lr, n_estimators in hyperparameters:
    model = GradientBoostingRegressor(max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators, random_state=seed)
    
    model.fit(train_x, train_y)
    test_p = model.predict(test_x)
    test_mae = mae_error(test_p, test_y)
    
    if best_mae > test_mae:
        best_hyperparameters = (max_depth, lr, n_estimators)
        best_model = copy.deepcopy(model)
        best_mae = test_mae

# 결과 출력
print('[info] Hyperparameter tuning finished')
print('[info] Best hyperparameters: {}'.format(best_hyperparameters))
print('[info] Best test mae: {}'.format(best_mae))
~~~

~~~
[info] Hyperparameter tuning finished
[info] Best hyperparameters: (4, 0.1, 120)
[info] Best test mae: 0.120635933267477
~~~


#### 스태킹(Stacking)
##### 이론
- 스태킹(Stacking)은 여러 개의 모델의 예측 결과를 입력으로 사용하여 새로운 메타 모델을 학습시키는 앙상블 기법이다. 베이스 모델은 원본 데이터로부터 예측을 수행하며, 메타 모델은 베이스 모델들의 예측 결과를 바탕으로 최종 예측을 수행한다. 이 방법은 다양한 모델의 강점을 결합하여 전체적인 성능을 향상시키는 데 도움을 준다.

##### 수행
~~~ python
# 베이스모델로 10개의 XGBClassifier 모델을 사용
estimators = [
    ("xgb_1", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=41)),
    ("xgb_2", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=42)),
    ("xgb_3", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=43)),
    ("xgb_4", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=44)),
    ("xgb_5", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=45)),
    ("xgb_6", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=46)),
    ("xgb_7", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=47)),
    ("xgb_8", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=48)),
    ("xgb_9", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=49)),
    ("xgb_10", XGBClassifier(gamma=0.5, leanring_rate=0.1, max_depth=4, n_estimators=20, random_state=50)),
]

# 메타모델로는 LogisticRegression 모델을 사용
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(x_resampled, y_resampled)  # 모델 학습
~~~

### In-depth Modeling
#### LIME (Local Interpretable Model-agnostic Explanations)
##### 정의
- LIME (Local Interpretable Model-agnostic Explanations)은 복잡한 머신러닝 모델의 예측을 설명하기 위한 방법 중 하나다. 
- 아래에서는 LIME의 동작 방식을 단순화한 `SimpleLime` 클래스를 구현한다.

##### 수행
1) `_generate_samples` 메서드를 사용하여 100개의 샘플들을 생성한다.
2) 1번에서 생성된 샘플을 분석 대상 모델의 입력으로 사용하여, class 1에 대한 예측값을 얻는다 (힌트 : `predict_proba` 함수 활용)
3) 이 예측값들을 데이터의 새로운 타겟으로 설정하여 새로운 선형 회귀 모델을 훈련시킨다.
4) 선형 모델의 계수를 특성의 중요도로 반환한다.

~~~ python
class SimpleLIME:
    def __init__(self, model: Any, num_samples: int = 100):
        """
        Initialize the SimpleLIME.
        
        Parameters:
        - model: The black-box model we want to explain.
        - num_samples: Number of perturbed samples to generate.
        """
        self.model = model
        self.num_samples = num_samples

    def _generate_samples(self, data_point: np.ndarray) -> np.ndarray:
        """
        Generate perturbed samples around a given data point.
        
        Parameters:
        - data_point: The data point around which to generate samples.
        
        Returns:
        - Perturbed samples.
        """
        noise = np.random.normal(loc=0, scale=1, size=(self.num_samples, data_point.shape[0]))
        samples = data_point + noise
        
        return samples

    def explain(self, data_point: np.ndarray) -> np.ndarray:
        """
        Explain the prediction of a given data point.
        
        Parameters:
        - data_point: The data point to explain.
        
        Returns:
        - feature_importances: Feature importances.
        """
        samples = self._generate_samples(data_point)
        predictions = self.model.predict_proba(samples)[:, 1]
        linear_model = LinearRegression().fit(samples, predictions) # 선형 회귀 모델로는 `LinearRegression`을 사용
        return linear_model.coef_
~~~
---
# 출처

---
# 관련 노트

---
# 외부 링크

