a = (1,2,3) # a = 1,2,3
a

a = 1,2,3
a

# 기본형은 tuple
# list에 비해서 가볍기 때문에

# 사용자 정의함수
# def min_max(numbers):
#   return min(numbers), max(numbers)
# 
# a = [1, 2, 3, 4, 5]
# result = min_max(a)
# type(result) 
# print("Minimum and maximum:", result) 

# >>> type(result)
# <class 'tuple'>
# >>> print("Minimum and maximum:", result)
# Minimum and maximum: (1, 5)
# 반환값의 결과가 tuple

a = [1,2,3]
a

# soft copy
b = a
b

a[1] = 4
a

b
id(a)
id(b)

# deep copy
a = [1,2,3]
a

id(a)

b=a[:]
b=a.copy()

a[1]=4
a
b

id(b)

# 수학함수
import math
x = 4
math.sqrt(x)

# 제곱근 계산
sqrt_val = math.sqrt(4)
sqrt_val

# 지수 계산
exp_val = math.exp(5)
exp_val

# 로그 계산
log_val = math.log(10,10)
log_val

# 팩토리얼 계산 예제
fact_val = math.factorial(5)
fact_val

# 정규분포 확률밀도함수

def my_normal_pdf(x, mu, sigma):
  part_1 = 1 / (sigma * math.sqrt(2* math.pi))
  # part_1 = (sigma * math.sqrt(2* math.pi)**(-1))
  part_2 = math.exp(-(x-mu)**2/(2*sigma**2))
  return part_1 * part_2
my_normal_pdf(3,3,1)

def normal_pdf(x, mu, sigma):
sqrt_two_pi = math.sqrt(2 * math.pi)
factor = 1 / (sigma * sqrt_two_pi)
return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
# 파라미터
mu = 0
sigma = 1
x = 1
# 확률밀도함수 값 계산
pdf_value = normal_pdf(x, mu, sigma)
print("정규분포 확률밀도함수 값은:", pdf_value)

def n_func(x,y,z):
  a = x**2
  b = math.sqrt(y)
  c = math.sin(z)
  d = math.exp(x)
  return (a+b+c)*d

n_func(2,9,math.pi/2)

def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)

my_g(math.pi)

def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...

def     (input):
    contents
    return

# Ctrl + Shift + c : 커멘드 처리
# ! pip install numpy
import pandas as pd
import pandas as pd
import numpy as np

# ** ndarray타입인 이유? numpy의 method를 사용하기 위해

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

type(a)

a[3]
a[2:]
a[1:4]

# 코드 길이가 3인 빈 배열 생성
b = np.empty(3)
b

b[0] = 1
b[1] = 4
b[2] = 10
b

b[2]

vec1 = np.array([1,2,3,4,5])
vec1 = np.arange(10)
vec1

vec1 = np.arange(1,100.1, 0.5)
vec1

# -100부터 0까지 표시하고 싶어요? 어떤 방법들이 있을까?
vec2 = np.arange(-100,1)
vec2

vec2 = np.arange(0, -100, -1)
vec3 = -np.arange(0,100)
vec3
vec2

l_space1 = np.linspace(0, 100, 100)
l_space1

linear_space2 = np.linspace(0, 1, 5, endpoint=False)
linear_space2

?np.linspace

# repeat vs tile
vec1=np.arange(5)
np.repeat(vec1,5)
np.tile(vec1, 3)

# 벡터의 사칙연산 가능
vec1 = np.array([1,2,3,4])
vec1 + vec1 
max(vec1)
sum(vec1)

# 35672이하 홀수들의 합은?
x = np.arange(1,35672,2)
sum(x)

np.arange(1,35672,2).sum()

len(x)
x.shape

# 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

a = np.array([1,2])
b = np.array([1,2,3,4])
a + b
np.tile(a, 2) + b
np.repeat(a, 2) + b

b == 3

# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
sum((np.arange(1, 35672) % 7) == 3)

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape
b.shape

import numpy as np

# 2차원 배열 생성(case1)
matrix = np.array([[ 0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0]])

matrix.shape

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 2차원 배열 생성(case2)
matrix = np.array([[ 0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0]])

matrix.shape

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)







