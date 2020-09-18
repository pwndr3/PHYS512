import numpy as np

def deriv(fun, x, delta):
    """
    Returns derivative of function `fun` evaluated at point `x`
    with the smallest step being `delta`.
    """
    f1 = (fun(x + delta) - fun(x - delta))/(2 * delta) # Step of delta
    f2 = (fun(x + 2 * delta) - fun(x - 2 * delta))/(4 * delta) # Step of 2 delta
    return (f2 - 4 * f1) / (-3)

if __name__ == '__main__':
    # Set up delta to sweep
    n = np.linspace(-16, -2, 40)
    delta_i = 10 ** n

    # Initialize arrays to plot
    errors = np.zeros(len(delta_i))
    
    # 1) exp(x)
    x0 = 1
    truth = np.exp(x0)
    for i, delta in enumerate(delta_i):
        est = deriv(np.exp, x0, delta)
        errors[i] = np.abs(est - truth)
    print("Optimal delta: %s" % delta_i[np.argmin(errors)])
    print("Lowest error: %s" % errors[np.argmin(errors)])
    
    # 2) exp(0.01x)
    def exp_001(x):
        return np.exp(0.01 * x)
    
    x0 = 1
    truth = 0.01 * exp_001(x0)
    for i, delta in enumerate(delta_i):
        est = deriv(exp_001, x0, delta)
        errors[i] = np.abs(est - truth)
    print("Optimal delta: %s" % delta_i[np.argmin(errors)])
    print("Lowest error: %s" % errors[np.argmin(errors)])
