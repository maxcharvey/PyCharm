import numpy as np
from matplotlib import pyplot as plt


def plot():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

    D = np.logspace(-2, 2, num=1000)
    IL = 300
    IH = 170
    B = 2.71

    Change_T = (IL - IH)/(4*D + B)
    TAV = 282.5
    TL = TAV + (0.5 * Change_T)
    TH = TAV - (0.5 * Change_T)

    F = 2 * D * (TL - TH)

    Entropy = (F/TH) - (F/TL)

    axs[0].plot(D, TH-273)
    axs[0].plot(D, TL-273)
    axs[1].plot(D, Entropy)
    axs[0].axvline(D[np.argmax(Entropy)], c='k', ls='--')

    for ax in axs:
        ax.set_xscale('log')

    axs[0].fill_between(D, 27, 15, color='C1', alpha=0.2)
    axs[0].fill_between(D, -5, 5, color='C0', alpha=0.2)

    axs[1].text(s=f'D at MEP={D[np.argmax(Entropy)]:.2f}', x=0.75, y=0.9, transform=axs[1].transAxes)

    plt.show()

if __name__ == "__main__":
    plot()
