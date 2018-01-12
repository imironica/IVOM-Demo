import numpy
import math

# Compute gaussian kernel

sigma = 1
M_PI = 3.1415
W = 5
kernel = numpy.zeros((W, W))
mean = W / 2
sum = 0.0
for x in range(0, W, 1):
    for y in range(0, W, 1):
        kernel[x][y] = math.exp(-0.5 * (math.pow((x - mean) / sigma, 2.0) + math.pow((y - mean) / sigma, 2.0))) / (
                2 * M_PI * sigma * sigma)
        sum += kernel[x][y]

# Normalize the kernel
for x in range(0, W, 1):
    for y in range(0, W, 1):
        kernel[x][y] /= sum

print(kernel)
