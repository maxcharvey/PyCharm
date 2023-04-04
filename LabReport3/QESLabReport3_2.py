# import cbsyst
from tools import plot
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('./OceanTools')

# global variables
V_ocean = 1.34e18  # volume of the ocean in m3
SA_ocean = 358e12  # surface area of the ocean in m2
fSA_hilat = 0.15  # fraction of ocean surface area in 'high latitude' box

# variables used to calculate Q
Q_alpha = 1e-4
Q_beta = 7e-4
Q_k = 8.3e17

# salinity balance - the total amount of salt added or removed to the surface boxes
Fw = 0.1  # low latitude evaporation - precipitation in units of m yr-1
Sref = 35  # reference salinity in units of g kg-1
E = Fw * SA_ocean * (1 - fSA_hilat) * Sref  # amount of salt removed from the low latitude box

# set up initial values for the boxes
init_hilat = {
    'name': 'hilat',
    'depth': 200,  # box depth, m
    'SA': SA_ocean * fSA_hilat,  # box surface area, m2
    'T': 15.,  # initial water temperature, Celcius
    'S': 34.,  # initial salinity
    'tau_M': 100.,  # timescale of surface-deep mixing, yr
    'T_atmos': 0.,  # air temperature, Celcius
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': -E,  # salt added due to evaporation - precipitation, kg m-3 yr-1
}
init_hilat['V'] = init_hilat['SA'] * init_hilat['depth']  # box volume, m3

init_lolat = {
    'name': 'lolat',
    'depth': 100,  # box depth, m
    'SA': SA_ocean * (1 - fSA_hilat),  # box surface area, m2
    'T': 15.,  # initial water temperature, Celcius
    'S': 35.,  # initial salinity
    'T_atmos': 25.,  # air temperature, Celcius
    'tau_M': 250.,  # timescale of surface-deep mixing, yr
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': E,  # salinity balance, PSU m3 yr-1
}
init_lolat['V'] = init_lolat['SA'] * init_lolat['depth']  # box volume, m3

init_deep = {
    'name': 'deep',
    'V': V_ocean - init_lolat['V'] - init_hilat['V'],  # box volume, m3
    'T': 5.,  # initial water temperature, Celcius
    'S': 34.5,  # initial salinity
}


def ocean_model(lolat, hilat, deep, tmax, dt):
    """Run the ocean model for a given time period and return the results for each box.

    Parameters
    ----------
    lolat, hilat, deep : dict
        dictionaries containing the box properties
    tmax : int or float
        The maximum time to run the model for (yr)
    dt : float
        The time step of the model (yr)

    Returns
    -------
    tuple of (time, lolat, hilat, deep)
    """

    time = np.arange(0, tmax + dt, dt)  # the time axis for the model

    # set which variables will change with time
    model_vars = ['T', 'S']

    # create copies of the input dictionaries so we don't modify the originals
    lolat = lolat.copy()
    hilat = hilat.copy()
    deep = deep.copy()

    # turn all time-evolving variables into arrays containing the start values
    for box in [lolat, hilat, deep]:
        for k in model_vars:
            box[k] = np.full(time.shape, box[k])

    fluxes = {}

    ### LOOP STARTS HERE ###
    for i in range(1, time.size):
        last = i - 1  # index of last model step

        # calculate circulation flux, Q
        dT = lolat['T'][last] - hilat['T'][last]
        dS = lolat['S'][last] - hilat['S'][last]
        Q_T = Q_k * (Q_alpha * dT - Q_beta * dS)  # m3 yr-1

        # calculate mixing fluxes for T and S
        for var in model_vars:
            fluxes[f'Q_{var}_deep'] = Q_T * (hilat[var][last] - deep[var][last]) * dt  # mol dt-1
            fluxes[f'Q_{var}_hilat'] = Q_T * (lolat[var][last] - hilat[var][last]) * dt  # mol dt-1
            fluxes[f'Q_{var}_lolat'] = Q_T * (deep[var][last] - lolat[var][last]) * dt  # mol dt-1

            fluxes[f'vmix_{var}_hilat'] = hilat['V'] / hilat['tau_M'] * (
                    hilat[var][last] - deep[var][last]) * dt  # mol dt-1
            fluxes[f'vmix_{var}_lolat'] = lolat['V'] / lolat['tau_M'] * (
                    lolat[var][last] - deep[var][last]) * dt  # mol dt-1

        # calculate temperature exchange with each surface box
        for box in [hilat, lolat]:
            boxname = box['name']
            # temperature exchange with atmosphere
            fluxes[f'dT_{boxname}'] = box['V'] / box['tau_T'] * (box['T_atmos'] - box['T'][last]) * dt  # mol dt-1

        # update deep box
        for var in model_vars:
            deep[var][i] = deep[var][last] + (
                    fluxes[f'Q_{var}_deep'] + fluxes[f'vmix_{var}_hilat'] + fluxes[f'vmix_{var}_lolat']
            ) / deep['V']

        # update surface boxes
        for box in [hilat, lolat]:
            boxname = box['name']
            box['S'][i] = box['S'][last] + (fluxes[f'Q_S_{boxname}'] - fluxes[f'vmix_S_{boxname}'] + box['E'] * dt) / \
                          box['V']
            box['T'][i] = box['T'][last] + (
                    fluxes[f'Q_T_{boxname}'] - fluxes[f'vmix_T_{boxname}'] + fluxes[f'dT_{boxname}']) / box['V']

    return time, lolat, hilat, deep


time, lolat, hilat, deep = ocean_model(init_lolat, init_hilat, init_deep, 1000, 0.5)

orig_hilat = init_hilat.copy()
orig_lolat = init_lolat.copy()
orig_hilat['tau_M'] = init_hilat['tau_M']/2
orig_lolat['tau_M'] = init_lolat['tau_M']/2
time, lolat, hilat2, deep = ocean_model(orig_lolat, orig_hilat, init_deep, 1000, 0.5)

orig_hilat['tau_M'] = init_hilat['tau_M']*2
orig_lolat['tau_M'] = init_lolat['tau_M']*2
time, lolat, hilat3, deep = ocean_model(orig_lolat, orig_hilat, init_deep, 1000, 0.5)

init_hilat = orig_hilat

fig, axs = plot.boxes(time, ['T', 'S'], hilat, ls='solid', label='100')
fig1, axs = plot.boxes(time, ['T', 'S'], hilat2, axs=axs, ls='dotted', label='50')
fig2, axs = plot.boxes(time, ['T', 'S'], hilat3, axs=axs, ls='dashed', label='200')

plt.legend(title=r'Hilat $\tau_m$ values (years)')
axs[0].set_xlim(0, 200)
axs[1].set_xlim(0, 200)
axs[0].set_ylabel('Temperature ( ̊C)')
axs[1].set_ylabel('Salinity (PSS)')
axs[1].set_xlabel('Time (years)')



model_vars = ['T', 'S']
for var in model_vars:
    print(var)
    for box in [hilat, lolat, deep]:
        print(f"  {box['name']}: {box[var][-1]:.2f}")

plt.savefig('QESLabReport13', dpi=600)

plt.show()
