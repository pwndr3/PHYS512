import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def cheb_fit(fun, order):
    x = np.linspace(0.5, 1, order + 1)
    y = fun(x)
    
    mat = np.zeros([order + 1, order + 1])
    mat[:, 0] = 1
    mat[:, 1] = x
    
    for i in range(1, order):
        mat[:, i + 1] = 2 * x * mat[:, i] - mat[:, i - 1]
    
    u,s,v = np.linalg.svd(mat, 0)
    coeffs = v.T@(np.diag(1.0/s)@(u.T@y))
    
    return coeffs

def cheb_eval(coeffs, x):
    if len(coeffs) == 1:
        return coeffs[0] * np.zeros(len(x))
    elif len(coeffs) == 2:
        return coeffs[0] * np.zeros(len(x)) + coeffs[1] * x
    else:
        T0 = np.ones(len(x))
        T1 = x
        res = np.zeros(len(x))
        for i in range(0, len(coeffs)):
            res += T0 * coeffs[i]
            
            tmp = T1
            T1 = 2 * x * T1 - T0
            T0 = tmp
        return res
    
def legendre_poly_fit(fun, order):
    x = np.linspace(0.5, 1, order + 1)
    y = fun(x)
    
    coeffs = np.polynomial.legendre.legfit(x, y, order)
    
    return coeffs

fun = np.log2
orders = range(3, 11)
acc = np.zeros(len(orders))
for i, order in enumerate(orders):
    coeffs = cheb_fit(fun, order + 1)

    x = np.linspace(0.5, 1, 100)
    y = cheb_eval(coeffs, x)
    y_true = fun(x)
    
    acc[i] = np.max(np.abs(y - y_true))
    
    print(order, acc[i])

plt.plot(orders, acc)
plt.plot(orders, 1e-6 * np.ones(len(orders)))
plt.xlabel('Chebyshev polynomial order')
plt.ylabel('Accuracy')
plt.legend(['Chebyshev polynomial accuracy', r'Accuracy limit ($\times 10^{-6}$)'])
plt.show()

best_order = orders[np.where(acc < 1e-6)[0][0]]
coeffs = cheb_fit(fun, best_order + 1)
x = np.linspace(0.5, 1, 100)
y_cheb = cheb_eval(coeffs, x)
y_true = fun(x)

coeffs = legendre_poly_fit(fun, best_order)
y_poly = np.polynomial.legendre.legval(x, coeffs)

plt.plot(x, (y_cheb - y_true)/1e-6)
plt.plot(x, (y_poly - y_true)/1e-6)
plt.xlabel(r'$x$')
plt.ylabel(r'Residuals ($\times 10^{-6}$)')
plt.legend(['Chebyshev polynomials', 'Legendre polynomials'])
plt.show()