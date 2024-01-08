# Fitting curves

from Lecture_12_part_1 import generate_measurement_data
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def cf_model(xdata, a0, a1, a2, a3):
    yhat = a0*xdata**3 + a1*xdata**2 + a2*xdata +a3
    return yhat

def cf_model_exp(xdata, a0,a1,a2):
    yhat = a0 * np.exp(a1*xdata) +a2
    return yhat

if __name__ == '__main__':
    x, y = generate_measurement_data()
    A = [1,2,3,4]
    coeffs = curve_fit(cf_model, x, y, p0=A)[0]
    print(coeffs)
    yhat = cf_model(x, *coeffs)
    plt.plot(x,y,'x')
    plt.plot(x,yhat,'-',label='curve fit poly')
    
    A = [1,2,3]
    coeffs = curve_fit(cf_model_exp, x, y, p0=A)[0]
    yhat = cf_model_exp(x, *coeffs)
    plt.plot(x,y,'x')
    plt.plot(x,yhat,'-',label='curve fit exp')
    plt.legend()
    plt.show()