import numpy as np
import matplotlib.pyplot as plt

(wl, B, V, R, I, QE, mirror) = np.loadtxt("KMTNet_transmission.dat").transpose()

fig, ax = plt.subplots()

ax.plot(wl, B, label = 'B', c = 'b')
ax.plot(wl, V, label = 'V', c = 'g')
ax.plot(wl, R, label = 'R', c = 'r')
ax.plot(wl, I, label = 'I', c = 'brown')
ax.plot(wl, QE, label = 'QE', c = 'gray')
ax.plot(wl, mirror, label = 'mirror', c = 'k')

ax.legend(loc = 1, fontsize = 9)
plt.show()

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
        
    return wl1, cw, wl2, wl2 - wl1, m, t, q

print "B", findvals(wl, B)
print "V", findvals(wl, V)
print "R", findvals(wl, R)
print "I", findvals(wl, I)
