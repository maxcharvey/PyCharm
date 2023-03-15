import os
from cbsyst import Csys
import numpy as np
import matplotlib.pyplot as plt
from tools import plot
from tools import helpers
import sys
from QESLabReport6_2 import original_model
from QESLabReport6_3 import acidification_model
from QESLabReport6_4 import ballasting_model


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
E = Fw * SA_ocean * (1 - fSA_hilat) * Sref  # amount of salt removed from the low latitude box,  g kg-1 yr-1, ~ kg m-3 yr-1

# NOTE: Initial DIC, TA, PO4 and pCO2 values are set to steady state values from the Ocean Acidification model.

particle_velocity = 10 # m d-1
k_diss = -0.07  # d-1
n_diss = 2.0  # unitless
Omega_crit = 2.5  # unitless
calc_slope = 0.12  # f_CaCO3 / Omega

rho_org = 1100
rho_CaCO3 = 2700

def create_dicts():
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
        'tau_CO2': 2.,  # timescale of CO2 exchange, yr
        'DIC': 2.02823,  # Dissolved Inorganic Carbon concentration, mol m-3
        'TA': 2.22043,  # Total Alkalinity, mol m-3
        'tau_PO4': 3.,  # phosphate half life, yr at initial f_CaCO3
        'PO4': 8.90099e-05,  # Phosphate conc, mol m-3
        'f_CaCO3': 0.18134,  # fraction of organic matter export that produces CaCO3 at starting [CO3]
        'k_ballast': 0.0805,
        'rho_particle': 1416.27,
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
        'tau_CO2': 2.,  # timescale of CO2 exchange, yr
        'DIC': 1.99301,  # Dissolved Inorganic Carbon concentration, mol m-3
        'TA': 2.21683,  # Total Alkalinity, mol m-3
        'tau_PO4': 2.,  # phosphate half life, yr at initial f_CaCO3
        'PO4': 1.65460e-04,  # Phosphate conc, mol m-3
        'f_CaCO3': 0.30453,  # fraction of organic matter export that produces CaCO3 at starting [CO3]
        'k_ballast': 0.1609,
        'rho_particle': 1416.27,
    }
    init_lolat['V'] = init_lolat['SA'] * init_lolat['depth']  # box volume, m3

    init_deep = {
        'name': 'deep',
        'V': V_ocean - init_lolat['V'] - init_hilat['V'],  # box volume, m3
        'T': 5.483637,  # initial water temperature, Celcius
        'S': 34.47283,  # initial salinity
        'DIC': 2.32710,  # Dissolved Inorganic Carbon concentration, mol m-3
        'TA': 2.31645,  # Total Alkalinity, mol m-3
        'PO4': 2.30515e-03,  # Phosphate conc, mol m-3
    }

    init_atmos = {
        'name': 'atmos',
        'mass': 5.132e18,  # kg
        'moles_air': 1.736e20,  # moles
        'moles_CO2': 872e15 / 12,  # moles
        'GtC_emissions': 0.0  # annual emissions of CO2 into the atmosphere, GtC
    }
    init_atmos['pCO2'] = init_atmos['moles_CO2'] / init_atmos['moles_air'] * 1e6

    init_hilat['particle_sinking_time'] = init_hilat['depth'] / particle_velocity
    init_lolat['particle_sinking_time'] = init_lolat['depth'] / particle_velocity

    return [init_lolat, init_hilat, init_deep, init_atmos]

def run():
    # perform baseline model run
    # oadicts = create_dicts()
    # oatime, oadicts = acidification_model(oadicts, 2000, 0.5)
    # oalolat, oahilat, oadeep, oaatmos = oadicts
    # vars = ['DIC', 'TA', 'pCO2', 'f_CaCO3', 'Omega']
    # plot.boxes(oatime, vars, oalolat, oahilat, oadeep, oaatmos)

    bldicts = create_dicts()


    tmax = 2000  # how many years to simulate (yr)
    dt = 0.5  # the time step of the simulation (yr)
    time = np.arange(0, tmax + dt, dt)  # the time axis for the model

    emit_atmos = bldicts[3].copy()  # create a copy of the original atmosphere input dictionary
    emit_atmos['GtC_emissions'] = np.zeros(time.shape)  # creat an array to hold the emission scenario
    emit_atmos['GtC_emissions'][(time > 800) & (time <= 1000)] = 8.0

    bldicts[3] = emit_atmos

    bltime, bldicts = ballasting_model(bldicts, 2000, 0.5)
    bllolat, blhilat, bldeep, blatmos = bldicts
    vars = ['DIC', 'TA', 'pCO2', 'f_CaCO3', 'GtC_emissions']
    plot.boxes(bltime, vars, bllolat, blhilat, bldeep, blatmos)

    # for k, v in helpers.get_last_values(ohilat, ololat, odeep, oatmos).items():
        # print(k)
        # for var in vars:
           #  if var in v:
                # print(f"  {var}: {v[var]:.5f}")
    plt.show()


if __name__ == "__main__":
    run()
