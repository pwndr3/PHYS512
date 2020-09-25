import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def u238_decay(t, y):
    """
    U238 decay chain
    """
    def from_days(d):
        return d / 365.0
    def from_hours(h):
        return from_days(h / 24.0)
    def from_minutes(m):
        return from_hours(m / 60.0)
    def from_seconds(s):
        return from_minutes(s / 60.0)

    # Half lives in years
    half_lives = [
        4.468e9,                # U-238
        from_days(24.1),        # Th-234
        from_hours(6.7),        # Pa-234
        245500,                 # U-234
        75380,                  # Th-230
        1600,                   # Ra-226
        from_days(3.8235),      # Rn-222
        from_minutes(3.1),      # Po-218
        from_minutes(26.8),     # Pb-214
        from_minutes(19.9),     # Bi-214
        from_seconds(164.3e-6), # Po-214
        22.3,                   # Pb-210
        5.015,                  # Bi-210
        from_days(138.376)      # Po-210
    ]
    
    dydx = np.zeros(len(half_lives) + 1)
    
    dydx[0] = -y[0] / half_lives[0]
    
    for i in range(1, len(dydx) - 1):
        dydx[i] = y[i - 1] / half_lives[i - 1] - y[i] / half_lives[i]
    
    dydx[-1] = y[-2] / half_lives[-2]
    
    return dydx

if __name__ == '__main__':
    y0 = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
    t0 = 0
    t1 = 5e10

    # U-238 decay products
    ans = integrate.solve_ivp(u238_decay, [t0, t1], y0, method='Radau', t_eval=np.linspace(0, t1, 1001))
    for i in range(ans.y.shape[0]):
        plt.plot(ans.t/1e10, ans.y[i,:])
    plt.xlabel(r'Time ($\times 10^{10}$ years)')
    plt.ylabel('Number relative to initial U-238 number')
    plt.legend(['U-238', 
               'Th-234', 
               'Pa-234', 
               'U-234', 
               'Th-230', 
               'Ra-226', 
               'Rn-222', 
               'Po-218', 
               'Pb-214', 
               'Bi-214', 
               'Po-214', 
               'Pb-210', 
               'Bi-210', 
               'Po-210',
               'Pb-206'])
    plt.savefig('images/prob3_u238_decay.png')

    # Pb-206/U-238 number ratio
    plt.clf()
    plt.plot(ans.t/1e10, ans.y[-1,:]/ans.y[0,:])
    plt.xlabel(r'Time ($\times 10^{10}$ years)')
    plt.ylabel('Pb-206/U-238 number ratio')
    plt.savefig('images/prob3_pb206_u238.png')

    # Th-230/U-234 number ratio
    plt.clf()
    t1 = 1e6
    ans = integrate.solve_ivp(u238_decay, [t0, t1], y0, method='Radau', t_eval=np.linspace(0, t1, 1001))
    plt.plot(ans.t/1e6, ans.y[4,:]/ans.y[3,:])
    plt.xlabel(r'Time ($\times 10^6$ years)')
    plt.ylabel('Th-230/U-234 number ratio')
    plt.savefig('images/prob3_th230_u234.png')
    
    