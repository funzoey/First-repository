import numpy
import torch
import matplotlib.pyplot as plt

x = numpy.empty(100)
a = torch.range(0, 100)
b = a*a*2-5*a+7

plt.plot(a, b)
plt.show()
# x1 = torch.ones(100)x
print(b)
# print((x+1)*2)


def big():
    a = 3
    b = 4
    c = 5
    return a, b

# a = [1,2,3]
# b = [4,5,6]
# for (idx, data), (idx1, dat) in zip(enumerate(a), enumerate(b)):
#     print(idx, ' - ', data)
#     print(idx1, ' - ', dat)


# out = big()
# print(out.__len__())
