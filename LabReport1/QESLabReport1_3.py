import numpy as np
from matplotlib import pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 12), sharex=True)
cols = ['r', 'm', 'g', 'c', 'b']
for i, D in enumerate(np.linspace(0.35, 1.35, 5)):
    print(D)
    for tl, th in [[305, 283], [300, 278], [295, 273]]:
        no_steps = 365
        f = np.zeros(no_steps)
        TL = np.zeros(no_steps)
        TH = np.zeros(no_steps)
        EL = np.zeros(no_steps)
        EH = np.zeros(no_steps)
        TL[0] = tl
        TH[0] = th

        A = 202
        B = 2.71
        Ca = 1.02e7
        EL[0] = A + B * (TL[0] - 273)
        EH[0] = A + B * (TH[0] - 273)
        IL = 300
        IH = 170
        max_t = 365
        steps = np.linspace(0, max_t, no_steps)
        stepsize = max_t / no_steps

        for t in range(0, len(steps) - 1):
            f[t] = 2 * D * (TL[t] - TH[t])
            EL[t + 1] = A + B * (TL[t] - 273)
            EH[t + 1] = A + B * (TH[t] - 273)

            TL[t + 1] = TL[t] + ((IL - EL[t + 1] - f[t]) / Ca * 86400) * stepsize
            TH[t + 1] = TH[t] + ((IH - EH[t + 1] + f[t]) / Ca * 86400) * stepsize

        axs[0].plot(steps, TL - 273, '--', lw=1, c='C' + str(i))
        axs[0].plot(steps, TH - 273, '-.', lw=1, c='C' + str(i))

        axs[1].plot(steps, EL, '--', lw=1, c='C' + str(i))
        axs[1].plot(steps, EH, '-.', lw=1, c='C' + str(i))

        if tl == 305:
            axs[2].plot(steps[:-1], f[:-1], lw=1, c='C' + str(i), label=D)
        else:
            axs[2].plot(steps[:-1], f[:-1], lw=1, c='C' + str(i))

axs[1].plot([0, max_t], [IL, IL], 'k--', label='Low latitude incoming radiation')
axs[1].plot([0, max_t], [IH, IH], 'k-.', label='High latitude incoming radiation')
axs[0].set_ylabel('Temperature ( ̊C)')
axs[1].set_ylabel('Heat Flux (Wm$^{-2}$)')
axs[2].set_ylabel('Latitudinal Heat Flow (Wm$^{-2}$)')
axs[2].set_xlabel('Time (Days)')
axs[0].fill_between(steps, 27, 15, color='lightgray')
axs[0].fill_between(steps, -5, 5, color='lightgray')
axs[2].legend(title='Values of D', ncols=3, fancybox=True)
axs[1].legend()

for i in axs:
    i.set_xlim(0, 350)

plt.savefig('QESLabReport11', dpi=600)

plt.show()
