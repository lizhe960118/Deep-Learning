# coding=utf-8

# import math
# 调用c内置的math，而非python的math
# 屏蔽了自身的math模块，然后调用了C的头文件math.h，并引入了cosf,sinf,acosf

cdef extern from "math.h":
    float cosf(float theta)
    float sinf(float theta)
    float acosf(float theta)

def spherical_distance(float lon1, float lat1, float lon2, float lat2):
    # 计算地球表面任意两个经纬度之间的距离
    cdef float radius = 3956
    cdef float pi = 3.14159265
    cdef x = pi / 180.0
    cdef a, b, theta, distance
    a = (90.0 - lat1) * x
    b = (90.0 - lat2) * x
    theta = (lon2 - lon1) * x
    distance = acosf(cosf(a) * cosf(b)) + (sinf(a) * sinf(b) * cosf(theta))
    return radius * distance

def f_compute(a, x, N):
    cdef int i
    cdef double s = 0
    cdef double dx = (x - a) / N 
    for i in range(N):
        s += ((a + i * dx) ** 2 - (a + i * dx))
    return s * dx

# 如果是double类型，则用cpdef定义
# cdef extern from "math.h":
#     double cos(double theta)
#     double sin(double theta)

# cpdef double tangent(double x):
#     return sin(x) / cos(x)