import numpy as np
import matplotlib.pyplot as plt
import scipy

# Uppgift 1
"""
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
plt.plot(x_1, y_1)
plt.plot(x_1, y_2)
plt.plot(y_3,x_2)
plt.vlines(2, -10, 10)
plt.axis([-10, 10, -10, 10])
plt.show()"""

# Får inte andra asymptoten att visa sig.


# Uppgift 4

def function_4(x):
    return 1+np.sin(x)+0.5*np.cos(4*x)


def derivative_analytical(x):
    return np.cos(x) - 2 * np.sin(4 * x)


def derivative(x, h):
    return (function_4(x + h) - function_4(x)) / h

"""
print(derivative(5, 0.000000001))
print(derivative_analytical(5))

x = np.arange(0, 10, 0.1)

# plt.plot(x, function_4(x))
plt.plot(x, derivative_analytical(x))
plt.plot(x, derivative(x, h=0.1))
plt.show()"""


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

"""
def answer_6a(t, a, b):
    return np.e**(-t) * (a * np.cos(t) + b * np.sin(t)) + np.cos(t) + 2 * np.sin(t)

def answer_6b(t):
    return np.cos(t) + 2*np.sin(t)

t = np.arange(0, 10, 1/81)

for A in range(-4, 5):
    for B in range(-4, 5):
        plt.plot(t, answer_6a(t, A, B))

plt.plot(np.arange(0, 10, 1/81), answer_6a(t, 0, 0))
plt.plot(t, answer_6b(t))
plt.show()
"""

# Uppgift 7
"""
def sine(x):
    return np.sin(x)


def taylor(x, k):
    sumval = 0
    for i in range(0, k+1):
        sumval = sumval + (((-1)**i * x**(1 + 2*i)) / (math.factorial(1 + 2*i)))
    # sumval = sumval + ((-1)**k * x**(1 + 2*k)) / (np.math.factorial(1 + 2*k))
    return sumval

# Något funkar inte när k bli större än 9. K=13 för att uppgiften ska vara löst.
k = 13
x = np.arange(-10, 10, 0.1)
plt.axis([-10, 10, -2, 2])
plt.plot(x, sine(x))
plt.plot(x, taylor(x, k))
plt.show()"""

# Uppgift 8
"""

def function_8a(x):
    return x - np.cos(x)

# Uppgift 8a
x = np.arange(-10, 10, 0.1)
plt.axis([-np.pi, np.pi, -5, 5])
plt.plot(x, function_8a(x))
plt.plot(x, [0]*len(x))
plt.show()

# Uppgift 8b
def bisect(a, b):
    c = (a + b) / 2
    counter = 0
    while abs(a-b) > 10**-12:
        if function_8a(c) == 0:
            return c, counter
        elif function_8a(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
        counter += 1
    return c, counter


def function_8b(x):
    return x - np.cos(x)

a=0.1
b=1.

bisection, b_counter = bisect(a, b)
print("Bisect: ", str(bisection))
print("Felet: ", str(function_8b(bisection)))


# Uppgift 8c / 8d

def derivative_f8(x):
    return 1 + np.sin(x)

def newton_raphson(a):
    x = a
    x_0 = 0
    counter = 0
    while abs(x - x_0) > 10**-12:
        x_0 = x
        x = x - function_8a(x) / derivative_f8(x)
        counter += 1
    return x, counter


newton, n_counter = newton_raphson(a)

print("Newton-Raphson: ", str(newton))
print("Counter Bisection: ", str(b_counter))
print("Counter Newton-Raphson: ", str(n_counter))


# Uppgift 8e

print("Scipy: ", scipy.optimize.fsolve(function_8a, a))
"""
# Uppgift 9

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Sfär (x^2 + y^2 + z^2 = 100)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
"""
# Kon (sqrt(x^2 + y^2) = z)
u = np.linspace(0, 2* np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), -np.sin(v))
"""
"""

# Pyramid (|x| + |y| = z)
def plot_pyramid(base_center, base_side_length, height):
    # Define the vertices of the base
    base_vertices = np.array([
        [base_center[0] - base_side_length / 2, base_center[1] - base_side_length / 2, base_center[2]],
        [base_center[0] + base_side_length / 2, base_center[1] - base_side_length / 2, base_center[2]],
        [base_center[0] + base_side_length / 2, base_center[1] + base_side_length / 2, base_center[2]],
        [base_center[0] - base_side_length / 2, base_center[1] + base_side_length / 2, base_center[2]]
    ])

    # Define the apex of the pyramid
    apex = np.array([base_center[0], base_center[1], base_center[2] + height])

    # Vertices of each triangular face
    vertices = [
        [base_vertices[0], base_vertices[1], apex],
        [base_vertices[1], base_vertices[2], apex],
        [base_vertices[2], base_vertices[3], apex],
        [base_vertices[3], base_vertices[0], apex],
        [base_vertices[0], base_vertices[1], base_vertices[2]],
        [base_vertices[0], base_vertices[2], base_vertices[3]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for vertex_set in vertices:
        xs = [vertex[0] for vertex in vertex_set]
        ys = [vertex[1] for vertex in vertex_set]
        zs = [vertex[2] for vertex in vertex_set]
        ax.plot(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')



# Example usage
base_center = np.array([0, 0, 0])
base_side_length = 4
height = 5

plot_pyramid(base_center, base_side_length, height)


# Halvsfär (x^2 + y^2 + z^2 = 100, z >= 0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi/2, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio
ax.set_aspect('equal')

plt.show()

# Två spiraler som snurrar runt varandra

ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.plot(y, x, z, label='parametric curve')
ax.legend()

plt.show()
"""

# Uppgift 10a

a = np.array([[4, -1, -9, -4, -6], [1, 1, -1, 4, -5], [0, -3, 4, 7, 0], [3, -5, -5, -3, 7], [9, -1, 4, -8, -9]])
b = np.array([-59, -21, 20, 16, -11])
x = np.linalg.solve(a, b)

print("x1:", x.item(0), "x2:", x.item(1), "x3:", x.item(2),"x4: ", x.item(3), "x5:", x.item(4))

# Uppgift 10b
file = open("Shinkansen.text", "r")


def kilometres(file_name):
    kilometres_list = []
    for line in file:
        elements = line.split()
        kilometres_list.append(elements[1])
    return kilometres_list


def price(file_name):
    price_list = []
    for line in file:
        elements = line.split()
        price_list.append(elements[2])
    return price_list


x = np.array(kilometres("Shinkansen.text"))
y = np.array(price("Shinkansen.text"))
print(x)
print(y)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()
