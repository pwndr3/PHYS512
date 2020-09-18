import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

if __name__ == '__main__':
    # Load data (T, V)
    data = np.loadtxt('lakeshore.txt', usecols=(0,1))

    # Sort to have increasing voltage
    data = data[data[:,1].argsort()]
    
    # Interpolate with odd points
    interp = interpolate.PchipInterpolator(data[::2,1], data[::2,0])
    
    # Plot
    plt.plot(data[:,1], data[:,0])
    plt.plot(data[:,1], interp(data[:,1]))
    plt.xlabel('Voltage (V)')
    plt.ylabel('Temperature (K)')
    plt.legend(['Real data', 'Interpolated data'])
    plt.savefig('images/prob2_lakeshore_interp.png')
    
    # Compute error on even points
    print("Error: %f" % np.std(data[1::2,0] - interp(data[1::2,1])))