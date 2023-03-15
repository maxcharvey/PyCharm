import numpy as np
import matplotlib.pyplot as plt
from cbsyst import Csys
from tools import plot
from tools import helpers

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
E = Fw * SA_ocean * (
            1 - fSA_hilat) * Sref  # amount of salt removed from the low latitude box,  g kg-1 yr-1, ~ kg m-3 yr-1

total_DIC = 38900e15 / 12  # mol C
avg_DIC = total_DIC / V_ocean

total_TA = 3.1e18  # mol TA
avg_TA = total_TA / V_ocean

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
    'DIC': 2.32226,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': avg_TA,  # Total Alkalinity, mol m-3
    'PO4': 3e15 / V_ocean,  # mol m-3
    'tau_PO4': 3,  # units of years
    'f_CaCO3': 0.2  # unitless
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
    'DIC': 2.26201,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': avg_TA,  # Total Alkalinity, mol m-3
    'PO4': 3e15 / V_ocean,  # mol m-3
    'tau_PO4': 2,  # units of years
    'f_CaCO3': 0.3  # unitless
}
init_lolat['V'] = init_lolat['SA'] * init_lolat['depth']  # box volume, m3

init_deep = {
    'name': 'deep',
    'V': V_ocean - init_lolat['V'] - init_hilat['V'],  # box volume, m3
    'T': 5.483637,  # initial water temperature, Celcius
    'S': 34.47283,  # initial salinity
    'DIC': 2.32207,  # Dissolved Inorganic Carbon concentration, mol m-3
    'TA': avg_TA,  # Total Alkalinity, mol m-3
    'PO4': 3e15 / V_ocean
}

init_atmos = {
    'name': 'atmos',
    'mass': 5.132e18,  # kg
    'moles_air': 1.736e20,  # moles
    'moles_CO2': 850e15 / 12,  # moles
    'GtC_emissions': 0.0  # annual emissions of CO2 into the atmosphere, GtC
}

init_atmos['pCO2'] = init_atmos['moles_CO2'] / init_atmos['moles_air'] * 1e6


def ocean_model(lolat, hilat, deep, atmos, tmax, dt):
    # create the time scale for the model
    time = np.arange(0, tmax + dt, dt)

    # identify which variables will change with time
    model_vars = ['T', 'S', 'DIC', 'TA', 'PO4']
    atmos_model_vars = ['moles_CO2', 'pCO2']

    # create copies of the input dictionaries so we don't modify the originals
    lolat = lolat.copy()
    hilat = hilat.copy()
    deep = deep.copy()
    atmos = atmos.copy()

    # turn all time-evolving variables into arrays containing the start values
    for box in [lolat, hilat, deep]:
        for k in model_vars:
            box[k] = np.full(time.shape, box[k])
    for k in atmos_model_vars:
        atmos[k] = np.full(time.shape, atmos[k])
    if isinstance(atmos['GtC_emissions'], (int, float)):
        atmos['GtC_emissions'] = np.full(time.shape, atmos['GtC_emissions'])

    # calculate initial surface carbon chemistry in the surface boxes using Csys, and store a few key variables - CO2, pH, pCO2 and K0
    for box in [lolat, hilat]:
        csys = Csys(
            TA=box['TA'], DIC=box['DIC'],
            T_in=box['T'], S_in=box['S'],
            unit='mmol'  # we specify mmol here because mol m-3 is the same as mmol L-1, and Csys works in L-1
        )
        box['CO2'] = csys.CO2
        box['pH'] = csys.pHtot
        box['pCO2'] = csys.pCO2
        box['K0'] = csys.Ks.K0

    # Create a dictionary to keep track of the fluxes calculated at each step
    fluxes = {}

    for i in range(1, time.size):
        last = i - 1  # index of last model step

        # calculate circulation flux, Q
        dT = lolat['T'][last] - hilat['T'][last]
        dS = lolat['S'][last] - hilat['S'][last]
        Q = Q_k * (Q_alpha * dT - Q_beta * dS)

        # calculate mixing fluxes for model variables (nothing to do here!)
        for var in model_vars:
            fluxes[f'Q_{var}_deep'] = Q * (hilat[var][last] - deep[var][last]) * dt  # amount dt-1
            fluxes[f'Q_{var}_hilat'] = Q * (lolat[var][last] - hilat[var][last]) * dt  # amount dt-1
            fluxes[f'Q_{var}_lolat'] = Q * (deep[var][last] - lolat[var][last]) * dt  # amount dt-1

            fluxes[f'vmix_{var}_hilat'] = hilat['V'] / hilat['tau_M'] * (
                    hilat[var][last] - deep[var][last]) * dt  # amount dt-1
            fluxes[f'vmix_{var}_lolat'] = lolat['V'] / lolat['tau_M'] * (
                    lolat[var][last] - deep[var][last]) * dt  # amount dt-1

        # calculate surface-specific fluxes
        for box in [hilat, lolat]:
            boxname = box['name']
            # temperature exchange with atmosphere
            fluxes[f'dT_{boxname}'] = box['V'] / box['tau_T'] * (
                    box['T_atmos'] - box['T'][last]) * dt  # V * degrees dt-1
            # CO2 exchange with atmosphere
            fluxes[f'dCO2_{boxname}'] = box['V'] / box['tau_CO2'] * (
                    box['CO2'][last] - 1e-3 * atmos['pCO2'][last] * box['K0'][last]) * dt  # mol dt-1

            # PO4 export from hilat and lolat to deep ocean
            fluxes[f'flux_PO4_{boxname}'] = box['V'] / box['tau_PO4'] * (
                box['PO4'][last]) * dt

            # DIC export from hilat/olat to the deep ocean
            fluxes[f'flux_DIC_{boxname}'] = 106 * fluxes[f'flux_PO4_{boxname}'] * (1 + box['f_CaCO3'])

            # TA exchange between hilat/lolat and the deep ocean
            fluxes[f'flux_TA_{boxname}'] = (106 * 2 * box['f_CaCO3'] - 18) * fluxes[f'flux_PO4_{boxname}']

        fluxes['dCO2_emissions'] = atmos['GtC_emissions'][last] * 1e15 / 12 * dt  # mol dt-1

        # update deep box
        for var in model_vars:
            deep[var][i] = deep[var][last] \
                           + (fluxes[f'Q_{var}_deep']
                              + fluxes[f'vmix_{var}_hilat']
                              + fluxes[f'vmix_{var}_lolat']
                              ) / deep['V']
            if var in ['DIC', 'TA', 'PO4']:
                deep[var][i] += fluxes[f'flux_{var}_hilat'] / deep['V']
                deep[var][i] += fluxes[f'flux_{var}_lolat'] / deep['V']

        # update surface boxes
        for box in [hilat, lolat]:
            boxname = box['name']
            box['S'][i] = box['S'][last] + (fluxes[f'Q_S_{boxname}'] - fluxes[f'vmix_S_{boxname}'] + box['E'] * dt) / \
                          box['V']  # PSU dt-1
            box['T'][i] = box['T'][last] + (
                    fluxes[f'Q_T_{boxname}'] - fluxes[f'vmix_T_{boxname}'] + fluxes[f'dT_{boxname}']) / box[
                              'V']  # degrees dt-1

            box['DIC'][i] = box['DIC'][last] + (
                    fluxes[f'Q_DIC_{boxname}']
                    - fluxes[f'vmix_DIC_{boxname}']
                    - fluxes[f'dCO2_{boxname}']
                    - fluxes[f'flux_DIC_{boxname}']) / box['V']  # mol m-3 dt-1

            box['TA'][i] = box['TA'][last] \
                           + (fluxes[f'Q_TA_{boxname}']
                              - fluxes[f'vmix_TA_{boxname}']
                              - fluxes[f'flux_TA_{boxname}']) / box['V']  # mol m-3 dt-1

            box['PO4'][i] = box['PO4'][last] \
                            + (fluxes[f'Q_PO4_{boxname}']
                               - fluxes[f'vmix_PO4_{boxname}']
                               - fluxes[f'flux_PO4_{boxname}']) / box['V']  # mol m-3 dt-1
            # update carbon speciation (nothing to do here!)
            csys = Csys(
                TA=box['TA'][i], DIC=box['DIC'][i],
                T_in=box['T'][i], S_in=box['S'][i],
                unit='mmol'
            )
            box['CO2'][i] = csys.CO2  # 1e3 converts back to mol m-3
            box['pCO2'][i] = csys.pCO2
            box['pH'][i] = csys.pHtot
            box['K0'][i] = csys.Ks.K0

        # update atmosphere
        atmos['moles_CO2'][i] = atmos['moles_CO2'][last] + fluxes['dCO2_hilat'] + fluxes['dCO2_lolat'] + fluxes[
            'dCO2_emissions']
        atmos['pCO2'][i] = 1e6 * atmos['moles_CO2'][i] / atmos['moles_air']

    return time, lolat, hilat, deep, atmos


def run():
    tmax = 3000  # how many years to simulate (yr)
    dt = 0.5  # the time step of the simulation (yr)
    time = np.arange(0, tmax + dt, dt)  # the time axis for the model

    emit_atmos = init_atmos.copy()  # create a copy of the original atmosphere input dictionary
    emit_atmos['GtC_emissions'] = np.zeros(time.shape)  # creat an array to hold the emission scenario
    emit_atmos['GtC_emissions'][(time > 200) & (time <= 400)] = 8.0

    X1_init_hilat = init_hilat.copy()
    X1_init_lolat = init_lolat.copy()
    X1_init_hilat['f_CaCO3'] /= 2
    X1_init_lolat['f_CaCO3'] /= 2

    X2_init_hilat = X1_init_hilat.copy()
    X2_init_lolat = X1_init_lolat.copy()
    X2_init_hilat['tau_PO4'] *= 2
    X2_init_lolat['tau_PO4'] *= 2

    # Run models
    time, lolat, hilat, deep, atmos = ocean_model(init_lolat, init_hilat, init_deep, emit_atmos, 3000, 0.5)
    X1_time, X1_lolat, X1_hilat, X1_deep, X1_atmos = ocean_model(X1_init_lolat, X1_init_hilat, init_deep, emit_atmos,
                                                                  3000, 0.5)
    X2_time, X2_lolat, X2_hilat, X2_deep, X2_atmos = ocean_model(X2_init_lolat, X2_init_hilat, init_deep, emit_atmos,
                                                                 3000, 0.5)

    # Make Plots
    fig, axs = plot.boxes(time, ['pCO2', 'GtC_emissions'], lolat, hilat, deep, atmos)

    plot.boxes(X1_time, ['pCO2', 'GtC_emissions'], X1_lolat, X1_hilat, X1_deep, X1_atmos, axs=axs, ls=':', label='low calc')
    plot.boxes(X2_time, ['pCO2', 'GtC_emissions'], X2_lolat, X2_hilat, X2_deep, X2_atmos, axs=axs, ls='--', label='low bio')
    axs[-1].plot(time, emit_atmos['GtC_emissions'], c='orange')
    axs[-1].legend(fontsize=8)

    for k, v in helpers.get_last_values(X1_hilat, X1_lolat, X1_atmos).items():
        print(k, v['pCO2'])
    for k, v in helpers.get_last_values(X2_hilat, X2_lolat, X2_atmos).items():
        print(k, v['pCO2'])

    plt.show()


if __name__ == '__main__':
    run()
