import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants.constants as const
from uncertainties import ufloat
from uncertainties import unumpy

#Koordinatensystem erstellen:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2

#Messwerte laden:
t1, t2, time = np.loadtxt("content/Messwerte.txt", unpack=True)

#Koordinatensystem schön machen:
plt.clf()
plt.xlabel(r'Zeit t in [s]')
plt.ylabel(r'Temperatur in [K]')
plt.grid()

#Messwerte plotten:
plt.plot(time, t1, 'r.', ms=12, label=r'Messwerte $T_1(K)$')
plt.plot(time, t2, 'b.', ms=12, label=r'Messwerte $T_2(K)$')

# Fitvorschrift
def f(x, A, B, C):
    return A*(x**2) + B*x + C                               #jeweilige Fitfunktion auswaehlen:
                                                            #Fuer polynomieller Plot: A*x**2+B*x+C
#1. Kurve für die Werte T1:
params, covar = curve_fit(f, time, t1)
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(time, f(time, *params), "r--", label=r'Fit $T_1$')

#2.Kurve für die Werte T2:
params, covar = curve_fit(f, time, t2)
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(time, f(time, *params), "b--", label=r'Fit $T_2$')

#Legende und anzeigen:
plt.legend(loc='best')
plt.savefig('build/tempfit.pdf')
