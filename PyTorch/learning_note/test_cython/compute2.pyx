# coding=utf-8
import math

# 全部变量都使用静态类型
cpdef float spherical_distance(float lon1, float lat1, float lon2, float lat2):
    # 计算地球表面任意两个经纬度之间的距离
    cdef float radius = 3956
    cdef float pi = 3.14159265
    cdef x = pi / 180.0
    cdef a, b, theta, distance
    a = (90.0 - lat1) * x
    b = (90.0 - lat2) * x
    theta = (lon2 - lon1) * x
    distance = math.acos(math.cos(a) * math.cos(b)) + (math.sin(a) * math.sin(b) * math.cos(theta))
    return radius * distance

def f_compute(a, x, N):
    cdef int i
    cdef double s = 0
    cdef double dx = (x - a) / N 
    for i in range(N):
        s += ((a + i * dx) ** 2 - (a + i * dx))
    return s * dx