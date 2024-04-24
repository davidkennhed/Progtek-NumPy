import numpy as np
import matplotlib.pyplot as plt
import math

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


x_1 = np.arange(-10, 11, 0.1)
x_2 = 2
y_1 = function_3(x_1)
y_2 = asymptote_1(x_1)
y_3 = asymptote_2(x_2)
"""plt.plot(x_1, y_1)
plt.plot(x_1, y_2)
plt.plot(y_3,x_2)
plt.vlines(2, -10, 10)
plt.axis([-10, 10, -10, 10])
plt.show()"""

# Får inte andra asymptoten att visa sig.


# Uppgift 4
"""
def function_4(x):
    return 1+np.sin(x)+0.5*np.cos(4*x)


def derivative_analytical(x):
    return np.cos(x) - 2 * np.sin(4 * x)


def derivative(x, h):
    return (function_4(x + h) - function_4(x)) / h


print(derivative(5, 0.000000001))
print(derivative_analytical(5))

x = np.arange(0, 10, 0.1)

# plt.plot(x, function_4(x))
plt.plot(x, derivative_analytical(x))
plt.plot(x, derivative(x, h=0.1))
plt.show()
"""

# Uppgift 5
"""
def riemann_sum(f, a, b, n):
    sumval = 0
    h = (b-a)/n
    for i in range(0, n-1):
        current_x = a+(i*h)
        sumval = sumval + f(current_x) * h
    return sumval


def integral_1(x):
    return x / (x**2 + 4)**(1 / 3)


def integral_2(x):
    return x**0.5 * np.log(x)


print(riemann_sum(integral_1, 0, 2, 10))
print(riemann_sum(integral_2, 1, 4, 10))"""

# Uppgift 6


"""def answer_6a(t, a, b):
    return np.e**(-t) * (a * np.cos(t) + b * np.sin(t)) + np.cos(t) + 2 * np.sin(t)

def answer_6b(t):
    return np.cos(t) + 2*np.sin(t)

t = np.arange(0, 10, 1/81)

for A in range(-4, 5):
    for B in range(-4, 5):
        plt.plot(t, answer_6a(t, A, B))

plt.plot(np.arange(0, 10, 1/81), answer_6a(t, 0, 0))
plt.plot(t, answer_6b(t))
plt.show()"""


# Uppgift 7

def sine(x):
    return np.sin(x)


def taylor(x, k):
    sumval = 0
    for i in range(0, k+1):
        sumval += (((-1)**i * x**(1 + 2*i)) / (math.factorial(1 + 2*i)))
    # sumval = sumval + ((-1)**k * x**(1 + 2*k)) / (np.math.factorial(1 + 2*k))
    return sumval

# Något funkar inte när k bli större än 9. K=13 för att uppgiften ska vara löst.
k = 9
x = np.arange(-10, 10, 0.1)
plt.axis([-10, 10, -2, 2])
plt.plot(x, sine(x))
plt.plot(x, taylor(x, k))
plt.show()