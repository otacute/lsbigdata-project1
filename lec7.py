# 쉽게 배우는 파이썬 데이터 분석 77p

import pandas as pd
import numpy as np

df = pd.DataFrame({'name' :['김지홍', '이유진', '박동현', '김민지'],
             'english' : [90, 80, 60, 70],
             'math' : [50,60,100,20]})
df

type(df)
type(df["name"])

sum(df["english"])

df2 = pd.DataFrame({'제품' : ['사과','딸기','수박'],
                    '가격' :[1800, 1500, 3000],
                    '판매량' :[24, 38, 13]})
df2

df2['가격'].mean()
df2['판매량'].mean()
