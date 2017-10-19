import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

(wlDECam, u, g, r, i, z, Y, atmDECam) = np.loadtxt("DECam_transmission_total.txt").transpose()


(wlatm, atm) = np.loadtxt("atm_ctio.dat").transpose()
wlatm = wlatm / 10. # microm
atmf = interp1d(wlatm, atm, bounds_error = False, fill_value = 0)

(wl, B, V, R, I, QE, mirror) = np.loadtxt("KMTNet_transmission.dat").transpose()
wl = wl # microm
B = B / 100.
V = V / 100.
R = R / 100.
I = I / 100.
QE = QE / 100.
mirror = mirror / 100.

Bf = interp1d(wl, B)
Vf = interp1d(wl, V)
Rf = interp1d(wl, R)
If = interp1d(wl, I)
QEf = interp1d(wl, QE)
mirrorf = interp1d(wl, mirror, bounds_error = False, fill_value = 0)

print (wl)
fig, ax = plt.subplots()

ax.plot(wl, B, label = 'B', c = 'b')
ax.plot(wl, V, label = 'V', c = 'g')
ax.plot(wl, R, label = 'R', c = 'r')
ax.plot(wl, I, label = 'I', c = 'brown')
ax.plot(wl, QE, label = 'QE', c = 'gray')
ax.plot(wl, mirror, label = 'mirror', c = 'k')
ax.plot(wlatm, atm, label = 'atmosphere', c = 'k', ls = ":")

colors = {'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'}
final = {}
for label, func in zip(['B', 'V', 'R', 'I'], [Bf, Vf, Rf, If]):
    ax.plot(wlDECam, func(wlDECam) * QEf(wlDECam) * mirrorf(wlDECam) * atmf(wlDECam), label = label, c = colors[label])
    final[label] = np.abs(func(wlDECam) * QEf(wlDECam) * mirrorf(wlDECam))

ax.legend(loc = 1, fontsize = 9)
plt.show()

np.savetxt("KMTNet_transmission_total.dat", np.array([wlDECam, final['B'], final['V'], final['R'], final['I'], atmDECam]).transpose(), fmt = "%.3f", header = "wavelength  B V R I atm")


def findvals(wl, band):

    th = np.max(band) / 3.
    
    # find lower limit
    for w, b in zip(wl, band):
        if b > th:
            wl1 = w
            break

    # find upper limit
    for w, b in zip(wl[::-1], band[::-1]):
        if b > th:
            wl2 = w
            break

    # find center of mass
    mask = (wl >= wl1) & (wl <= wl2)
    cw = np.sum(wl[mask] * band[mask]) / np.sum(band[mask])

    # typical value
    t = np.sum(wl[mask] * band[mask]) / np.sum(wl[mask])
    q = np.sum(wl[mask] * QE[mask]) / np.sum(wl[mask])
    m = np.sum(wl[mask] * mirror[mask]) / np.sum(wl[mask])
        
    return wl1, cw, wl2, wl2 - wl1, m, t, q, atmf(cw)

print("B cw, w1, dw, mirror, filter, QE ", findvals(wl, Bf(wl)))# * QEf(wl) * mirrorf(wl)))
print("V cw, w1, dw, mirror, filter, QE ", findvals(wl, Vf(wl)))# * QEf(wl) * mirrorf(wl)))
print("R cw, w1, dw, mirror, filter, QE ", findvals(wl, Rf(wl)))# * QEf(wl) * mirrorf(wl)))
print("I cw, w1, dw, mirror, filter, QE ", findvals(wl, If(wl)))# * QEf(wl) * mirrorf(wl)))


