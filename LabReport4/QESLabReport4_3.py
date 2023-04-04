import matplotlib.pyplot as plt
from QESLabReport4_1 import ocean_model
from tools import plot
import numpy as np

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

E = Fw * SA_ocean * (
        1 - fSA_hilat) * Sref  # amount of salt removed from the low latitude box,  g kg-1 yr-1, ~ kg m-3 yr-1

init_hilat = {
    'name': 'hilat',
    'depth': 200,  # box depth, m
    'SA': SA_ocean * fSA_hilat,  # box surface area, m2
    'T': 3.897678,  # initial water temperature, Celcius
    'S': 34.37786,  # initial salinity
    'T_atmos': 0.,  # air temperature, Celcius
    'tau_M': 100.,  # timescale of surface-deep mixing, yr
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': -E,  # salt added due to evaporation - precipitation, kg m-3 yr-1
    'DIC': 2.30018233641265,
    # 1/3 of the ocean's DIC in moles of carbon (split equally between the three boxes)
    'TA': 3.1e18 / V_ocean,
    'tau_CO2': 2
}
init_hilat['V'] = init_hilat['SA'] * init_hilat['depth']  # box volume, m3

init_lolat = {
    'name': 'lolat',
    'depth': 100,  # box depth, m
    'SA': SA_ocean * (1 - fSA_hilat),  # box surface area, m2
    'T': 23.60040,  # initial water temperature, Celcius
    'S': 35.37898,  # initial salinity
    'T_atmos': 25.,  # air temperature, Celcius
    'tau_M': 250.,  # timescale of surface-deep mixing, yr
    'tau_T': 2.,  # timescale of temperature exchange with atmosphere, yr
    'E': E,  # salinity balance, PSU m3 yr-1
    'DIC': 2.2388213938431694,
    # 1/3 of the ocean's DIC in moles of carbon (split equally between the three boxes)
    'TA': 3.1e18 / V_ocean,
    'tau_CO2': 2
}

init_lolat['V'] = init_lolat['SA'] * init_lolat['depth']  # box volume, m3

init_deep = {
    'name': 'deep',
    'V': V_ocean - init_lolat['V'] - init_hilat['V'],  # box volume, m3
    'T': 5.483637,  # initial water temperature, Celcius
    'S': 34.47283,  # initial salinity
    'DIC': 2.2956873603696915,
    # 1/3 of the ocean's DIC in moles of carbon (split equally between the three boxes)
    'TA': 3.1e18 / V_ocean
}

init_atmos = {
    'name': 'atmos',
    'mass': 5e21,  # in grams
    'moles_air': 1.736e20,
    'moles_CO2': 2.213e17,
    'GtC_emissions': 0
}

init_atmos['pCO2'] = (init_atmos['moles_CO2'] * 1e6) / init_atmos['moles_air']  # carbon ppm


tmax = 10000  # how many years to simulate (yr)
dt = 0.5  # the time step of the simulation (yr)
time = np.arange(0, tmax + dt, dt)  # the time axis for the model

emit_atmos = init_atmos.copy()  # create a copy of the original atmosphere input dictionary
emit_atmos['GtC_emissions'] = np.zeros(time.shape)  # creat an array to hold the emission scenario
emit_atmos['GtC_emissions'][(time > 500) & (time <= 700)] = 8.0

dicts = (init_lolat, init_hilat, init_deep, emit_atmos)

time, output = ocean_model(dicts, tmax, dt)
final_lolat, final_hilat, final_deep, final_atmos = output

fig, axs = plot.boxes(time, ['pCO2', 'DIC'], final_lolat, final_hilat, final_deep, final_atmos)


emission = np.linspace(500, 700, 200)
y_min = [1000, 2.21]
y_max = [2200, 2.4]

for i in range(2):
    axs[i].fill_between(emission, y_max[i], y_min[i], color='lightgray')
    axs[i].set_ylim(y_min[i], y_max[i])

axs[0].set_ylabel(r'$pCO_2$ (ppm)')
axs[1].set_ylabel(r'DIC ($mol \; m^{-3}$)')
axs[1].set_xlabel('Time (years)')

plt.savefig('QESLabReport14', dpi=600)

print((final_lolat['DIC'][-1] - final_lolat['DIC'][1]) * 12 / 1e15 * init_lolat['V'])
print((final_hilat['DIC'][-1] - final_hilat['DIC'][1]) * 12 / 1e15 * init_hilat['V'])
print((final_deep['DIC'][-1] - final_deep['DIC'][1]) * 12 / 1e15 * init_deep['V'])

plt.show()


