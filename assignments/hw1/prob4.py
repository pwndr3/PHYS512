import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def electric_field(theta, R, z):
    """
    Returns electric field function to integrate in cgs
    units and with sigma = 1.
    """
    num = (z - R * np.cos(theta)) * (2 * np.pi * R * np.sin(theta)) * R
    den = (z**2 + R**2 - 2 * z * R * np.cos(theta))**(3/2)
    
    return num / (den + 1e-16)

def my_integrate(func, x1, x2, args = (), tol = 1e-3, max_recursion = 10):
    """
    Integrate using variable step size
    """
    x = np.linspace(x1, x2, 5)
    y = func(x, *args)
    
    area1 = (x2 - x1) * (y[0] + 4*y[2] + y[4])/6
    area2 = (x2 - x1) * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/12
    
    if max_recursion == 0:
        return area2
    
    err = np.abs(area1 - area2)
    if err < tol:
        return area2
    else:
        xm = 0.5 * (x1 + x2)
        a1 = my_integrate(func, x1, xm, args, tol/2, max_recursion - 1)
        a2 = my_integrate(func, xm, x2, args, tol/2, max_recursion - 1)
        return a1 + a2

if __name__ == '__main__':
    # Define shell radius
    R = 1
    
    # Loop over z and integrate for each
    zs = np.concatenate((np.linspace(0.0, R, endpoint=False), np.linspace(R, 5.0)))
    Es_np = np.zeros(len(zs))
    Es_myintegrate = np.zeros(len(zs))
    for i, z in enumerate(zs):
        Es_np[i] = integrate.quad(electric_field, 0, np.pi, args=(R, z,))[0]
        Es_myintegrate[i] = my_integrate(electric_field, 0, np.pi, args=(R, z,))
        
    # Plot electric field vs z
    plt.plot(zs, Es_np)
    plt.plot(zs, Es_myintegrate)
    plt.xlabel('Distance from radius')
    plt.ylabel('Electric field')
    plt.legend(['np.integrate.quad', 'Own integrator (variable step size)'])
    plt.savefig('images/prob4_electric_field.png')
    
    print("Integration error: %f" % np.std(Es_np - Es_myintegrate))