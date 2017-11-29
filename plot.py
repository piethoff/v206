import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants.constants as const
from uncertainties import ufloat
from uncertainties import unumpy
import matplotlib as mpl

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

#Koordinatensystem erstellen:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2

#Messwerte laden:
t1, t2, time = np.loadtxt("content/Messwerte.txt", unpack=True)

#Koordinatensystem schön machen:
plt.clf()
plt.xlabel(r'Zeit $t/\si{\second}')
plt.ylabel(r'Temperatur $/\si{\kelvin}')
plt.grid()

#Messwerte plotten:
plt.plot(time, t1, 'r.', ms=12, label=r'Messwerte $T_1$')
plt.plot(time, t2, 'b.', ms=12, label=r'Messwerte $T_2$')

# Fitvorschrift
def f(x, A, B, C, D):
    return A*(x**3) + B*(x**2) + C*x + D                               #jeweilige Fitfunktion auswaehlen:
def g(x, A, B, a):
    return A/(1+B*(x**a))
def h(x, A, B, C, a):
    return A*(x**a)/(1+B*(x**a))+C

                                                            #Fuer polynomieller Plot: A*x**2+B*x+C
#1. Kurve für die Werte T1:
##params, covar = curve_fit(g, time, t1, p0=(295, 1, 2))
params, covar = curve_fit(f, time, t1, p0=(1, 1, 1, 295))
#params, covar = curve_fit(h, time, t1, p0=(0, 0, 295, 2))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

##plt.plot(time, g(time, *params), "r--", label=r'Fit $T_1$')
plt.plot(time, f(time, *params), "r--", label=r' $F_{T_1}$')
#plt.plot(time, h(time, *params), "r--", label=r'Fit $T_1$')

#2.Kurve für die Werte T2:
##params, covar = curve_fit(g, time, t2, p0=(295, 1, 2))
params, covar = curve_fit(f, time, t2, p0=(1, 1, 1, 295))
#params, covar = curve_fit(h, time, t2, p0=(0, 0, 295, 2))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

##plt.plot(time, g(time, *params), "b--", label=r'Fit $T_2$')
plt.plot(time, f(time, *params), "b--", label=r'$F_{T_2}$')
#plt.plot(time, h(time, *params), "b--", label=r'Fit $T_2$')

#Legende und anzeigen:
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('build/tempfit.pdf')

plt.clf()

data = np.genfromtxt("content/Messwerte2.txt", unpack=True)

data[1] += 273.15
data[1] = 1/data[1]
data[4] += 1
#data[4] = np.log(data[4])

def p(T, A, LR):
    return np.exp(-LR*T+A)

plt.plot(data[1], data[4],'o', label="Messwerte $p_2$")
params, covar = curve_fit(p, data[1], data[4], p0=(9, 2000))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print(uparams)

plt.yscale("log")
plt.ylabel(r"$p/\si{\bar}$")
plt.xlabel(r"$\frac{1}{T}/\si{\per\kelvin}$")
plt.plot(data[1], p(data[1], *params),'-', label="lin. Reg.")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("build/dampfdruck.pdf")
