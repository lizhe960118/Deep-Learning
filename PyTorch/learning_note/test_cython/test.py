#!/user/bin/env python
# coding=utf-8
import compute
import time 

lon1, lat1, lon2, lat2 = -72.345, 34.323, -61.823, 54.826

start_time = time.clock()
compute.f_compute(3.2, 6.9, 1000000)
end_time = time.clock()
print("runing1 time: %f s" % (end_time - start_time))

start_time = time.clock()
for i in range(1000000):
    compute.spherical_distance(lon1, lat1, lon2, lat2)
end_time = time.clock()
print("runing2 time: %f s" % (end_time - start_time))
