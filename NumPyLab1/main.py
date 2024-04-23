import numpy as np
import matplotlib.pyplot as plt

# Uppgift 1

a = np.arange(1, 6, 1)                # 1a
b = np.arange(0, 2*np.pi, 0.1)        # 1b
c = np.array([[1, 2, 3], [4, 5, 6]])  # 1c
d_add = [6, 7]
d = np.array(a.tolist() + d_add)      # 1d
e_add = np.arange(-5, 0, 1)
e = np.array(a.tolist() + e_add.tolist())
e_ny = e.reshape(2, 5)                # 1e
f = np.sin(b)                         # 1f


# Uppgift 2

def function_a(x):
    return x*x


def function_b1(y):
    return y * y


def function_b2(y):
    return y@y


def function_c1(z):
    return z*z


def function_c2(z):
    return z@z


x_1 = 2
y = np.array([1, 2, 3])
z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# Uppgift 3

def function_3(x):
    return 1 + x + (4/((x-2)*(x-2)))


def asymptote_1(x):
    return 1 + x


def asymptote_2(y):
    return y == 2


x_1 = np.arange(-10, 11, 0.5)
x_2 = 2
y_1 = function_3(x_1)
y_2 = asymptote_1(x_1)
y_3 = asymptote_2(x_2)
plt.plot(x_1, y_1)
plt.plot(x_1, y_2)
plt.plot(y_3,x_2)
plt.show()

# Får inte andra asymptoten att visa sig.


# Uppgift 4

def function_4(x):
    return 1+np.sin(x)+0.5*np.cos(4*x)


def derivative(x):
    return
