import matplotlib.pyplot as plt
import numpy as np

def my_integrate_no_cache(func, x1, x2, args = (), tol = 1e-3):
    """
    Integrate using variable step size.
    """
    x = np.linspace(x1, x2, 5)
    y = func(x, *args)
    
    area1 = (x2 - x1) * (y[0] + 4*y[2] + y[4])/6
    area2 = (x2 - x1) * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/12
    
    err = np.abs(area1 - area2)
    if err < tol:
        return area2
    else:
        xm = 0.5 * (x1 + x2)
        a1 = my_integrate_no_cache(func, x1, xm, args, tol/2)
        a2 = my_integrate_no_cache(func, xm, x2, args, tol/2)
        return a1 + a2
    
def my_integrate_cache(func, x1, x2, args = (), tol = 1e-3, func_cache = {}):
    """
    Integrate using variable step size with cache for function values.
    """
    x = np.linspace(x1, x2, 5)
    
    y = np.zeros(len(x))
    for i in range(len(y)):
        y[i] = func_cache.get(x[i]) or func(x[i], *args)
        func_cache[x[i]] = y[i]
    
    area1 = (x2 - x1) * (y[0] + 4*y[2] + y[4])/6
    area2 = (x2 - x1) * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/12
    
    err = np.abs(area1 - area2)
    if err < tol:
        return area2
    else:
        xm = 0.5 * (x1 + x2)
        a1 = my_integrate_cache(func, x1, xm, args, tol/2, func_cache)
        a2 = my_integrate_cache(func, xm, x2, args, tol/2, func_cache)
        return a1 + a2
    
def my_integrate_params(func, x1, x2, args = (), tol = 1e-3, y1 = None, y2 = None):
    """
    Integrate using variable step size with passing boundary values as parameters.
    """
    x = np.linspace(x1, x2, 5)
    
    y = np.zeros(len(x))
    
    if y1:
        y[0] = y1
        y[1:] = func(x[1:], *args)
    elif y2:
        y[-1] = y2
        y[:-1] = func(x[:-1], *args)
    else:
        y = func(x, *args)
    
    area1 = (x2 - x1) * (y[0] + 4*y[2] + y[4])/6
    area2 = (x2 - x1) * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/12
    
    err = np.abs(area1 - area2)
    if err < tol:
        return area2
    else:
        xm = 0.5 * (x1 + x2)
        a1 = my_integrate_params(func, x1, xm, args, tol/2, y1=y[0])
        a2 = my_integrate_params(func, xm, x2, args, tol/2, y2=y[-1])
        return a1 + a2

# Lorentzian
def lorentzian(x):
    return 1/(1 + x**2)
    
if __name__ == '__main__':
    fct_count = 0
    
    def func_wrapper(x, func):
        """
        Function wrapper that counts the number of calls.
        """
        global fct_count
        
        if hasattr(x, 'size'):
            fct_count += x.size
        else:
            fct_count += 1
        
        return func(x)
    
    def get_fct_count():
        """
        Returns fct_count and resets count.
        """
        global fct_count
        
        tmp = fct_count
        fct_count = 0
        
        return tmp
    
    def plot_fct_calls_vs_tol(func, a, b, plot_name):
        """
        Plots number of function calls vs tolerance and
        save to file `plot_name`.
        """
        tol_arr = np.power(10, np.linspace(-12, -3, 80))
        fct_count_cache = np.zeros(len(tol_arr))
        fct_count_params = np.zeros(len(tol_arr))
        fct_count_no_cache = np.zeros(len(tol_arr))
    
        for i, tol in enumerate(tol_arr):
            my_integrate_cache(func_wrapper, a, b, args=(func,), tol=tol, func_cache={})
            fct_count_cache[i] = get_fct_count()
            
            my_integrate_params(func_wrapper, a, b, args=(func,), tol=tol)
            fct_count_params[i] = get_fct_count()

            my_integrate_no_cache(func_wrapper, a, b, args=(func,), tol=tol)
            fct_count_no_cache[i] = get_fct_count()
        
        plt.clf()
        plt.loglog(tol_arr, fct_count_cache)
        plt.loglog(tol_arr, fct_count_params)
        plt.loglog(tol_arr, fct_count_no_cache)
        plt.xlabel('Tolerance')
        plt.ylabel('Number of function calls')
        plt.legend(['Using cache', 'Passing values as parameters', 'Without cache'])
        plt.savefig("images/" + plot_name)
        
        print("Average difference cache vs parameters: %f" % np.mean(np.abs(fct_count_cache - fct_count_params)))
        print("Average difference cache vs no cache: %f" % np.mean(np.abs(fct_count_cache - fct_count_no_cache)))
        print("Average difference parameters vs no cache: %f" % np.mean(np.abs(fct_count_params - fct_count_no_cache)))
        print("---")
    
    # Exponential
    plot_fct_calls_vs_tol(np.exp, 0, 1, 'prob1_exp.png')
    
    # Lorentzian
    plot_fct_calls_vs_tol(lorentzian, -1, 1, 'prob1_lorentzian.png')