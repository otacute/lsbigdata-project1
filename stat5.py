# 패키지 불러오기
import numpy as np
import pandas as pd

tab3 = pd.read_csv('./data/tab3.csv')
tab3

tab1 = pd.DataFrame({"id":np.arange(1,13),
                    "score" : tab3["score"]})
tab1

tab2 = tab1.assign(gender =["female"] * 7  +["male"] * 5)
tab2

# 1 표본 t 검정(그룹 1개)
# 표본 t 검정 (그룹 1개)
# 귀무가설 vs. 대립가설
# H0 : mu = 10 vs Ha : mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp
result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1["score"].mean() # 표본평균
result.pvalue
result.statistic
result.df
# 귀무가설이 참(mu=10)일 때, 11.53(표본평균)이 관찰될 확률이 6.48%이므로,
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05(유의수준)보다 크므로, 귀무가설이 거짓이라 판단하기 힘들다.
# 유의 확률 0.0648이 유의수준 0.05보다 크므로  
# 귀무가설을 기각하지 못한다.

# 95% 신뢰구간 구하기
ci = result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]

# 2 표본 t 검정(그룹 2개) - 분산 같고, 다를 때
# 분산 같은경우 : 독립 2표본 t검정
# 분산 다를경우 : 웰치스 t검정
# 귀무가설 vs. 대립가설
# H0 : mu_m = mu_f vs Ha : mu_m > mu_f
# 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다.
f_tab2 = tab2[tab2["gender"]=="female"]
m_tab2 = tab2[tab2["gender"]=="male"]

from scipy.stats import ttest_ind

result = ttest_ind(m_tab2["score"], f_tab2["score"],
                    equal_var=True, alternative='greater')
                    
# alternative = "less"의 의미는 대립가설이,
# 첫 번째 입력그룹의 평균이 두 번째 입력 그룹 평균보다 작다고 설정된 경우를 나타냄
# result = ttest_ind(f_tab2["score"], m_tab2["score"],
#                     equal_var=True, alternative='less')

result.statistic
result.pvalue
# ci = result.confidence_interval(confidence_level=0.95)
# ci[0]
# ci[1]

# 대응표본 t 검정(짝지을 수 있는 표본)
# 귀무가설 vs. 대립가설
# H0 : mu_before = mu_after vs Ha : mu_after > mu_before
# H0 : mu_d = 0 vs Ha : mu_d > 0
# mu_d = mu_after - mu_before
# 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환
tab3_data = tab3.pivot_table(index='id', 
                             columns='group',
                             values='score').reset_index()
                             
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

from scipy.stats import ttest_1samp

result = ttest_1samp(tab3_data["score_diff"], popmean=0, alternative='greater')
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1["score"].mean() # 표본평균

result.pvalue
result.statistic
result.df

#
# long to wide : pivot_table()
tab3_data = tab3.pivot_table(
    index='id', 
    columns='group',
    values='score'
    ).reset_index()


# wide to long : melt()
long_form = tab3_data.melt(
        id_vars='id', 
        value_vars=['before', 'after'], 
        var_name='group', 
        value_name='score'
        )
        
# 연습 1
df = pd.DataFrame({
    "id" : [1,2,3],
    "A"  : [10,20,30],
    "B"  : [40,50,60]
})
df

df_long = df.melt(id_vars="id",
            value_vars=["A","B"],
            var_name="group",
            value_name="score")
df_long

df_wide_1 = df_long.pivot_table(index = "id",
                    columns = "group",
                    values = "score")
df_wide_1
                    
df_wide_2 = df_long.pivot_table(
            columns = "group",
            values = "score")
df_wide_2

df_wide_3 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "mean")
df_wide_3

df_wide_4 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "max")
df_wide_4

# 연습 2
# import seaborn as sns
# tips = sns.load_dataset("tips")
# tips
# 
# tips.pivot_table(
#         columns = "day",
#         values = "tip")

# 요일별로 펼치고 싶은 경우
# index_list = list(tips.columns.delete(4))
# 
# tips.pivot_table(
#         index = ["index"],
#         columns = "day",
#         values = "tip").reset_index()
# 
# tips["day"].value_counts()

# 2024.08.07(수)
# 교재 Chapter 11. 지도 시각화
# 11-1. 시군구별 인구 단계 구분도 만들기
# JSON (JavaScript Object Notation)
import json
geo = json.load(open('./data/bigfile/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo['features'][0]['properties']

# 11-2. 서울시 동별 외국인 인구 단계 구분도 만들기
import json

geo_seoul = json.load(open('./data/bigfile/SIG_Seoul.geojson', encoding = 'UTF-8'))

# json - dict내에 dict가 있는 구조
type(geo_seoul)
len(geo_seoul)

geo_seoul.keys()
geo_seoul["features"]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
type(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 행정 구역 코드 출력
geo_seoul["features"][0]["properties"]

# 위도, 경도 좌표 출력
geo_seoul["features"][0]["geometry"]

coordinate_list = geo_seoul["features"][0]["geometry"]['coordinates']
coordinate_list
len(coordinate_list)
coordinate_list[0]
len(coordinate_list[0])
coordinate_list[0][0]
len(coordinate_list[0][0])

import numpy as np
coordinate_array = np.array(coordinate_list[0][0])
coordinate_array

x = coordinate_array[:,0]
y = coordinate_array[:,1]

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()






