import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    tmp1 = NAND(x1, x2)
    tmp2 = OR(x1, x2)
    return AND(tmp1, tmp2)


x1, x2 = 1, 1
print(x1, x2, AND(x1, x2), NAND(x1, x2), OR(x1, x2), XOR(x1, x2))

x1, x2 = 1, 0
print(x1, x2, AND(x1, x2), NAND(x1, x2), OR(x1, x2), XOR(x1, x2))

x1, x2 = 0, 1
print(x1, x2, AND(x1, x2), NAND(x1, x2), OR(x1, x2), XOR(x1, x2))

x1, x2 = 0, 0
print(x1, x2, AND(x1, x2), NAND(x1, x2), OR(x1, x2), XOR(x1, x2))
