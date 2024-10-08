import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False) # include_bias = False, 1을 제외
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

# 알파 값 설정(lambda)
alpha_values = np.arange(0, 10, 0.01)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=5000) # max_iter의 의미 : minimize가 한번 돌아갈 때 시행하는 횟수.
    scores = cross_val_score(lasso, X_poly, y, cv=kf, scoring='neg_mean_squared_error') 
    # valid_result의 결과, # 'neg_mean_squared_error' : y - y^ 의 평균 * (-1) : 성능이 좋으면, (값이 낮다. * (-1)) => 성능이 좋으면, 값이 높다 가 되기위해 neg를 통한 flip을 함 
    mean_scores.append(np.mean(scores))

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)


# --------------------------수정된 코드-------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)
