import numpy as np
from matplotlib import pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15),sharex=True)

no_steps=365
f = np.zeros(no_steps)
TL = np.zeros(no_steps)
TH = np.zeros(no_steps)
EL = np.zeros(no_steps)
EH = np.zeros(no_steps)
TL[0] = 298
TH[0] = 280
D = 0.85

A = 203.3
B = 2.71
Ca = 1.02e7
EL[0] = A+B*(TL[0]-273)
EH[0] = A+B*(TH[0]-273)
IL = 300
IH = 170
max_t = 365
steps = np.linspace(0,max_t,no_steps)
stepsize = max_t/no_steps

def plot2():
    for t in range(0,len(steps)-1):
        f[t] = 2 * D * (TL[t] - TH[t])
        EL[t + 1] = A + B * (TL[t] - 273)
        EH[t + 1] = A + B * (TH[t] - 273)

        TL[t + 1] = TL[t] + ((IL - EL[t + 1] - f[t]) / Ca * 86400) * stepsize
        TH[t + 1] = TH[t] + ((IH - EH[t + 1] + f[t]) / Ca * 86400) * stepsize

    axs[0].plot(steps,TL-273,'b--',)
    axs[0].plot(steps,TH-273,'b-.')

    axs[1].plot(steps,EL,'b--')
    axs[1].plot(steps,EH,'b-.')

    axs[2].plot(steps[:-1],f[:-1])

    axs[1].plot([0,max_t],[IL,IL],'k--')
    axs[1].plot([0,max_t],[IH,IH],'k-.')
    axs[0].set_ylabel('Temperature ( ̊C)')
    axs[1].set_ylabel('Heat Flux (Wm$^{-2}$)')
    axs[2].set_ylabel('Latitudinal Heat Flow (Wm$^{-2}$)')
    axs[2].set_xlabel('Time (Days)')

    plt.show()

    
if __name__ == "__main__":
    plot2()
