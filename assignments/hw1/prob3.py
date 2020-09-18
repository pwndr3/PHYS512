import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

NUMBER_OF_POINTS_FOR_INTERP = 6

def polyfit(xi, yi, func):
    """
    Performs cubic polynomial fit.
    """
    x = np.linspace(xi[1], xi[-2], 1001)
    y_interp = np.zeros(len(x))
    
    # Fit a cubic polynomial and compute interpolated value for each x
    for i in range(len(x)):    
        ind = np.max(np.where(x[i] >= xi)[0])
        
        # Define the 4 points to use for interpolation
        x_use = xi[ind-1:ind+3]
        y_use = yi[ind-1:ind+3]
        
        # Fit
        pars = np.polyfit(x_use, y_use, 3)
        
        # Compute interpolated point
        pred = np.polyval(pars, x[i])
        
        y_interp[i] = pred

    # Compute true values
    y_true = func(x)
    
    return x, y_interp, np.std(y_interp - y_true)

def splinefit(xi, yi, func):
    """
    Performs spline fit.
    """
    # Fit spline
    spln = interpolate.splrep(xi, yi)
    
    # Get interpolated points in the whole range
    x = np.linspace(xi[0], xi[-1], 1001)
    y_interp = interpolate.splev(x, spln)
    
    # Compute true values
    y_true = func(x)
    
    return x, y_interp, np.std(y_interp - y_true)

def ratfit(xi, yi, func, n = 4, m = 5, pinv = False):
    """
    Performs rational function fit.
    """
    # Evaluate rational function
    def rat_eval(p,q,x):
        top=0
        for i in range(len(p)):
            top=top+p[i]*x**i
        bot=1
        for i in range(len(q)):
            bot=bot+q[i]*x**(i+1)
        return top/bot

    # Fit rational function
    def rat_fit(x,y,n,m):
        assert(len(x)==n+m-1)
        assert(len(y)==len(x))
        mat=np.zeros([n+m-1,n+m-1])
        for i in range(n):
            mat[:,i]=x**i
        for i in range(1,m):
            mat[:,i-1+n]=-y*x**i
        if pinv:
            pars=np.dot(np.linalg.pinv(mat),y)
        else:    
            pars=np.dot(np.linalg.inv(mat),y)
        print("Matrix: ", mat)
        print("Determinant: ", np.linalg.det(mat))
        p=pars[:n]
        q=pars[n:]
        print("p: ", p)
        print("q: ", q)
        print('###')
        return p,q

    # Fit
    p, q = rat_fit(xi, yi, n, m)

    # Get interpolated points in the whole range
    x = np.linspace(xi[0], xi[-1], 1001)
    y_interp = rat_eval(p, q, x)
    
    # Compute true values
    y_true = func(x)
    
    return x, y_interp, np.std(y_interp - y_true)

if __name__ == '__main__':
    # 1) Cosine
    # Define points for interpolation
    interp_range = [-np.pi/2, np.pi/2]
    xi = np.linspace(interp_range[0], interp_range[1], NUMBER_OF_POINTS_FOR_INTERP)
    yi = np.cos(xi)
    
    pol_x, pol_y, pol_err = polyfit(xi, yi, np.cos) 
    spline_x, spline_y, spline_err = splinefit(xi, yi, np.cos) 
    rat_x, rat_y, rat_err = ratfit(xi, yi, np.cos, NUMBER_OF_POINTS_FOR_INTERP // 2, NUMBER_OF_POINTS_FOR_INTERP // 2 + 1) 
    
    # Plot function
    plt.plot(pol_x, pol_y)
    plt.plot(spline_x, spline_y)
    plt.plot(rat_x, rat_y)
    plt.plot(xi, yi, "*")
    plt.xlabel('$x$')
    plt.ylabel(r'$y(x) = \cos(x)$')
    plt.legend(['Polynomial', 'Cubic spline', 'Rational', 'True'])
    plt.savefig('images/prob3_cos_func.png')
    
    # Plot errors
    plt.clf()
    plt.bar(['Polynomial', 'Cubic spline', 'Rational'], [pol_err, spline_err, rat_err])
    plt.ylabel('RMS error')
    plt.savefig('images/prob3_cos_errs.png')
    
    # 2a) Lorentzian with inv (not pinv)
    def lorentzian(x):
        return 1.0 / (1.0 + x**2)
        
    interp_range = [-1, 1]
    xi = np.linspace(interp_range[0], interp_range[1], NUMBER_OF_POINTS_FOR_INTERP)
    yi = lorentzian(xi)
    
    pol_x, pol_y, pol_err = polyfit(xi, yi, lorentzian) 
    spline_x, spline_y, spline_err = splinefit(xi, yi, lorentzian) 
    rat_x, rat_y, rat_err = ratfit(xi, yi, lorentzian, NUMBER_OF_POINTS_FOR_INTERP // 2, NUMBER_OF_POINTS_FOR_INTERP // 2 + 1) 
    
    # Plot function
    plt.clf()
    plt.plot(pol_x, pol_y)
    plt.plot(spline_x, spline_y)
    plt.plot(rat_x, rat_y)
    plt.plot(xi, yi, "*")
    plt.xlabel('$x$')
    plt.ylabel(r'$y(x) = \frac{1}{1 + x^2}$')
    #plt.ylim(-20, 20) # For poor rational fitting
    plt.legend(['Polynomial', 'Cubic spline', 'Rational', 'True'])
    plt.savefig('images/prob3_lorentzian_func.png')
    
    # Plot errors
    plt.clf()
    plt.bar(['Polynomial', 'Cubic spline', 'Rational'], [pol_err, spline_err, rat_err])
    plt.ylabel('RMS error')
    plt.savefig('images/prob3_lorentzian_errs.png')
    
    # 2b) Lorentzian with pinv
    rat_x, rat_y, rat_err = ratfit(xi, yi, lorentzian, NUMBER_OF_POINTS_FOR_INTERP // 2, NUMBER_OF_POINTS_FOR_INTERP // 2 + 1, pinv=True) 
    
    # Plot function
    plt.clf()
    plt.plot(pol_x, pol_y)
    plt.plot(spline_x, spline_y)
    plt.plot(rat_x, rat_y)
    plt.plot(xi, yi, "*")
    plt.xlabel('$x$')
    plt.ylabel(r'$y(x) = \frac{1}{1 + x^2}$')
    plt.legend(['Polynomial', 'Cubic spline', 'Rational', 'True'])
    plt.savefig('images/prob3_lorentzian_func_pinv.png')
    
    # Plot errors
    plt.clf()
    plt.bar(['Polynomial', 'Cubic spline', 'Rational'], [pol_err, spline_err, rat_err])
    plt.ylabel('RMS error')
    plt.savefig('images/prob3_lorentzian_errs_pinv.png')