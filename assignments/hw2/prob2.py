import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def cheb_mat(x, order):
    """
    Generate Chebyshev matrix.
    """
    mat = np.zeros([len(x), order + 1])
    mat[:, 0] = 1
    mat[:, 1] = x
    
    for i in range(1, order):
        mat[:, i + 1] = 2 * x * mat[:, i] - mat[:, i - 1]
    
    return mat

if __name__ == '__main__':
    # Define function
    fun = np.log2
    
    x = np.linspace(0.5, 1, 101)
    y = fun(x)
    
    # Scale x for optimal fitting in [-1, 1]
    x_scaled = x * 4 - 3
    
    # a) Orders to evaluate 
    orders = range(3, 11)

    # Fit and extract coefficients
    A = cheb_mat(x_scaled, max(np.max(orders), 50))
    u,s,v = np.linalg.svd(A, 0)
    cheb_coeffs = v.T@(np.diag(1.0/s)@(u.T@y))

    # For each order, get the truncated Chebyshev coefficients
    # and compute the accuracy
    acc = np.zeros(len(orders))
    for i, order in enumerate(orders):
        # Predict
        y_pred = A[:, :order]@cheb_coeffs[:order]
        
        # Get accuracy
        acc[i] = np.mean(np.abs(y_pred - y))
        
        print(order, acc[i])

    # Plot accuracy vs order
    plt.semilogy(orders, acc)
    plt.semilogy(orders, 1e-6 * np.ones(len(orders)))
    plt.xlabel('Chebyshev polynomial order')
    plt.ylabel('Accuracy')
    plt.legend(['Chebyshev polynomial accuracy', r'Accuracy limit ($\times 10^{-6}$)'])
    plt.savefig("images/prob2_cheb_acc.png")

    # b) Extract order to use (where accuracy is less than 1e-6)
    best_order = orders[np.where(acc < 1e-6)[0][0]]
    print("Best order: %d" % best_order)
    
    # Chebyshev
    y_cheb = A[:, :best_order]@cheb_coeffs[:best_order]
    
    print("Chebyshev:")
    print("\tRMS: %e" % np.std(y - y_cheb))
    print("\tMax: %e" % np.max(np.abs(y - y_cheb)))

    # Legendre polynomial
    A = np.polynomial.legendre.legvander(x_scaled, max(np.max(orders), 50))
    
    u,s,v = np.linalg.svd(A, 0)
    poly_coeffs = v.T@(np.diag(1.0/s)@(u.T@y))
    
    y_poly = A[:, :best_order]@poly_coeffs[:best_order]
    
    print("Polynomial:")
    print("\tRMS: %e" % np.std(y - y_poly))
    print("\tMax: %e" % np.max(np.abs(y - y_poly)))

    # Plot residuals
    plt.clf()
    plt.plot(x, (y_cheb - y)/1e-7)
    plt.plot(x, (y_poly - y)/1e-7)
    plt.xlabel(r'$x$')
    plt.ylabel(r'Residuals ($\times 10^{-7}$)')
    plt.legend(['Chebyshev polynomials', 'Legendre polynomials'])
    plt.savefig("images/prob2_residuals.png")