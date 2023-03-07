import matplotlib.pyplot as plt
from QESLabReport4_1 import make_dictionaries, ocean_model
from tools import plot


def ocean_mixing():

    dicts = make_dictionaries()
    time, finished_dicts = ocean_model(dicts, 3000, 0.5)
    lolat, hilat, deep, atmos = finished_dicts
    fig, axs = plot.boxes(time, ['pCO2'], atmos)

    dicts[0]['tau_M'] /= 2
    dicts[1]['tau_M'] /= 2

    time, finished_dicts = ocean_model(dicts, 3000, 0.5)
    lolat, hilat, deep, atmos = finished_dicts
    fig, axs = plot.boxes(time, ['pCO2'], atmos, axs=axs, ls='--')

    plt.show()


if __name__ == '__main__':
    ocean_mixing()
