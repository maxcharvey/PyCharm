import numpy as np
from matplotlib import pyplot as plt

alpha = 2e-4
beta = 7e-4
k = 8.3e17
taulist = [2, 20, 200]

SA_ocean = 3.58e14
AL = SA_ocean * 0.85
AH = SA_ocean * 0.15
D = 3000
VL = AL * D
VH = AH * D
Fw = 0.25
Sref = 35
E = AH * Fw * Sref

duration = 1000
steps = 1000
stepsize = duration / steps
times = np.linspace(0, duration, steps)

TL = np.empty_like(times)
SL = np.empty_like(times)
TH = np.empty_like(times)
SH = np.empty_like(times)
Q = np.empty_like(times)

TL[0] = 298.
SL[0] = 37.
TH[0] = 273.
SH[0] = 33.
TatL = 30. + 273.
TatH = 0. + 273.

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)


for i in range(3):
    tau = taulist[i]
    print(tau)
    for t in range(0, steps - 1):
        delT = TL[t] - TH[t]
        delS = SL[t] - SH[t]
        Q[t] = k * (alpha * delT - beta * delS)
        if Q[t] > 0:
            TL[t + 1] = TL[t] + stepsize * (-Q[t] * delT - (VL / tau) * (TL[t] - TatL)) / VL
            TH[t + 1] = TH[t] + stepsize * (Q[t] * delT - (VH / tau) * (TH[t] - TatH)) / VH
            SL[t + 1] = SL[t] + stepsize * (-Q[t] * delS + E) / VL
            SH[t + 1] = SH[t] + stepsize * (Q[t] * delS - E) / VH

    delT = TL[steps - 1] - TH[steps - 1]
    delS = SL[steps - 1] - SH[steps - 1]
    Q[steps - 1] = k * (alpha * delT - beta * delS)

    axs[0].plot(times, TL - 273, '--', label=r'$T_L$')
    axs[0].plot(times, TH - 273, '-.', label=r'$T_H$')

    axs[1].plot(times, SL, '--', label=r'$S_L$')
    axs[1].plot(times, SH, '-.', label=r'$S_H$')

    axs[2].plot(times, Q)

axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('Temperature ( ̊C)')
axs[1].set_ylabel('Salinity')
axs[2].set_ylabel('Latitudinal flow strength ($m^{3}yr^{-1}$)')
axs[2].set_xlabel('Time (years)')

plt.show()
