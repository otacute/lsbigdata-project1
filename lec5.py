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
