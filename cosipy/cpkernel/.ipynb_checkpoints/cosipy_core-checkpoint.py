import numpy as np
import pandas as pd

from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.cpkernel.init import init_snowpack, load_snowpack
from cosipy.cpkernel.io import IOClass
from cosipy.modules.albedo import updateAlbedo
from cosipy.modules.densification import densification
from cosipy.modules.evaluation import evaluate
from cosipy.modules.heatEquation import solveHeatEquation
from cosipy.modules.penetratingRadiation import penetrating_radiation
from cosipy.modules.percolation import percolation
from cosipy.modules.refreezing import refreezing
from cosipy.modules.roughness import updateRoughness
from cosipy.modules.surfaceTemperature import update_surface_temperature
import gc

def init_nan_array_1d(nt: int) -> np.ndarray:
    """Initialise and fill an array with NaNs.
    
    Args:
        nt: Array size (time dimension).

    Returns:
        NaN array.
    """
    if not Config.WRF_X_CSPY:
        x = np.full(nt, np.nan)
    else:
        x = None

    return x


def init_nan_array_2d(nt: int, max_layers: int) -> np.ndarray:
    """Initialise and fill an array with NaNs.
    
    Args:
        nt: Array's temporal resolution.
        max_layers: Array's spatial resolution.

    Returns:
        2D NaN array.
    """
    if not Config.WRF_X_CSPY and Config.full_field:
        x = np.full((nt, max_layers), np.nan)
    else:
        x = None

    return x

def count_timesteps(start_time, end_time, dt):
    """Calc. number of steps between two times, given a timestep
    
    Args:
        start_time: 
        end_time:
        dt: timestep in [s] (usually timestep of the input file)

    Returns:
        int    
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    nt = int((end_time - start_time).total_seconds() / dt)
    return nt


def cosipy_core(DATA, indY, indX, GRID_RESTART=None, stake_names=None, stake_data=None):
    """Cosipy core function.

    The calculations are performed on a single core.

    Args:
        DATA (xarray.Dataset): Dataset with single grid point.
        indY (int): The grid cell's Y index.
        indX (int): The grid cell's X index.
        GRID_RESTART (xarray.Dataset): Use a restart dataset instead of
            creating an initial profile. Default ``None``.
        stake_names (list): Stake names. Default ``None``.
        stake_data (pd.Dataframe): Stake measurements. Default ``None``.

    Returns:
        All calculated variables for one grid point.
    """
    
    # Declare locally for faster lookup
    albedo_method = Constants.albedo_method  
    time_tephra_event = Constants.time_tephra_event
    dt = Constants.dt
    max_layers = Constants.max_layers
    z = Constants.z
    try:
        z_u = Constants.z_u
    except:
        z_u=None
    mult_factor_RRR = Constants.mult_factor_RRR
    densification_method = Constants.densification_method
    ice_density = Constants.ice_density
    water_density = Constants.water_density
    minimum_snowfall = Constants.minimum_snowfall
    zero_temperature = Constants.zero_temperature
    lat_heat_sublimation = Constants.lat_heat_sublimation
    lat_heat_melting = Constants.lat_heat_melting
    lat_heat_vaporize = Constants.lat_heat_vaporize
    center_snow_transfer_function = Constants.center_snow_transfer_function
    spread_snow_transfer_function = Constants.spread_snow_transfer_function
    constant_density = Constants.constant_density
    albedo_fresh_snow = Constants.albedo_fresh_snow
    albedo_firn = Constants.albedo_firn
    WRF_X_CSPY = Config.WRF_X_CSPY
    output_atm = Config.output_atm.split(",") if Config.output_atm else []
    output_internal = Config.output_internal.split(",") if Config.output_internal else []
    output_full = [f"LAYER_{v.strip()}" for v in Config.output_full.split(",") if v.strip()] if Config.output_full else []
    log_int = Config.logging_interval            # timesteps per log entry

    # Replace values from constants.py if coupled
    # TODO: This only affects the current module scope instead of global.
    if WRF_X_CSPY:
        dt = int(DATA.DT.values)
        max_layers = int(DATA.max_layers.values)
        z = float(DATA.ZLVL.values)
    nt = len(DATA.time.values)  # accessing DATA is expensive
    n_logs = (nt + log_int - 1) // log_int          # number of log entries
    # MOD: Fixed methods dict (dummy assumptions)
    base_methods = {
        'RAIN':'sum','SNOWFALL':'sum',
        'LWin':'mean','LWout':'mean','H':'mean','LE':'mean','B':'mean','QRR':'mean',
        'TS':'mean','ALBEDO':'mean','Z0':'mean',
        'MB':'sum','surfMB':'sum','Q':'sum','SNOWHEIGHT':'snapshot',
        'RHO_AVG':'mean','SDF':'mean','TOTALHEIGHT':'snapshot','LAYERS':'snapshot',
        'ME':'sum','intMB':'sum','EVAPORATION':'sum','SUBLIMATION':'sum',
        'CONDENSATION':'sum','DEPOSITION':'sum','REFREEZE':'sum',
        'subM':'sum','surfM':'sum','MOL':'snapshot'
    }
    presaved = [
        "HGT",
        "MASK",
        "SLOPE",
        "ASPECT",
        "T2",
        "RH2",
        "U2",
        "PRES",
        "G",
        "RRR",
        "SNOWFALL",
        "N",
        "LWin",
    ]
    
    
    """
    Local variables bypass local array creation for WRF_X_CSPY
    TODO: Implement more elegant solution.
    """
    if not WRF_X_CSPY:
        results = {}
        for var in output_atm + output_internal + output_full:
            if var in output_full and Config.full_field:
                results[var] = np.full((n_logs, max_layers), np.nan)
            else:
                results[var] = np.full(n_logs, 0,dtype='float32')
        if Config.track_ages:
            _LAYER_CREATION_AGE = init_nan_array_2d(nt, max_layers)
            _LAYER_BURIAL_AGE = init_nan_array_2d(nt, max_layers)


    #--------------------------------------------
    # Initialize snowpack or load restart grid
    #--------------------------------------------
    if GRID_RESTART is None:
        GRID = init_snowpack(DATA)
    else:
        GRID = load_snowpack(GRID_RESTART)

    # Create the local output datasets if not coupled
    RESTART = None
    if not WRF_X_CSPY:
        IO = IOClass(DATA)
        RESTART = IO.create_local_restart_dataset()

    # hours since the last snowfall (albedo module)
    # hours_since_snowfall = 0

    #--------------------------------------------
    # Get data from file
    #--------------------------------------------
    T2 = DATA.T2.values
    RH2 = DATA.RH2.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    U2 = DATA.U2.values

    #--------------------------------------------
    # Checks for optional input variables
    #--------------------------------------------
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = DATA.RRR.values * mult_factor_RRR
    elif 'SNOWFALL' in DATA:
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = None
        RAIN = None
    else:
        SNOWF = None
        RRR = DATA.RRR.values * mult_factor_RRR

    # Use RRR rather than snowfall?
    if Config.force_use_TP:
        SNOWF = None

    LWin = np.array(nt * [None])
    N = np.array(nt * [None])
    if ('LWin' in DATA) and ('N' in DATA):
        LWin = DATA.LWin.values
        N = DATA.N.values
    elif 'LWin' in DATA:
        LWin = DATA.LWin.values
    else:
        LWin = None
        N = DATA.N.values

    # Use N rather than LWin
    if Config.force_use_N:
        LWin = None

    SLOPE = 0.
    if 'SLOPE' in DATA:
        SLOPE = DATA.SLOPE.values

    # Initial cumulative mass balance variable
    MB_cum = 0

    # Initial snow albedo and surface temperature for Bougamont et al. 2005 albedo
    surface_temperature = 270.0
    albedo_snow = albedo_fresh_snow
    if WRF_X_CSPY:
        albedo_snow = albedo_firn

    if Config.stake_evaluation:
        # Create pandas dataframe for stake evaluation
        _df = pd.DataFrame(index=stake_data.index, columns=['mb','snowheight'], dtype='float')

    if time_tephra_event:
        tte = count_timesteps(Config.time_start, time_tephra_event, dt)

    te = 0   #counter to see what lwin is triggered
    #--------------------------------------------
    # TIME LOOP
    #--------------------------------------------
    for t in np.arange(nt):
        
        # Check grid
        GRID.grid_check()

        # get seconds since start
        # timestamp = dt*t
        # if Config.WRF_X_CSPY:
            # timestamp = np.float64(DATA.CURR_SECS.values)

        # Calc fresh snow density
        if densification_method != 'constant':
            density_fresh_snow = np.maximum((109.0+6.0*(T2[t]-273.16)+26.0*np.sqrt(U2[t])), 50.0)
        else:
            density_fresh_snow = constant_density 

        # Derive snowfall [m] and rain rates [m w.e.]
        if (SNOWF is not None) and (RRR is not None):
            SNOWFALL = SNOWF[t]
            RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/water_density) * 1000.0
        elif SNOWF is not None:
            SNOWFALL = SNOWF[t]
        elif RRR is not None:
            # Else convert total precipitation [mm] to snowheight [m]; liquid/solid fraction
            SNOWFALL = (RRR[t]/1000.0)*(water_density/density_fresh_snow)*(0.5*(-np.tanh(((T2[t]-zero_temperature) - center_snow_transfer_function) * spread_snow_transfer_function) + 1.0))
            RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/water_density) * 1000.0
        else:
            raise ValueError("No SNOWFALL or RRR data provided.")

        # if snowfall is smaller than the threshold
        if SNOWFALL<minimum_snowfall:
            SNOWFALL = 0.0

        # if rainfall is smaller than the threshold
        if RAIN<(minimum_snowfall*(density_fresh_snow/water_density)*1000.0):
            RAIN = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_fresh_snow(SNOWFALL, density_fresh_snow, np.minimum(float(T2[t]),zero_temperature), 0.0, dt)
        else:
            GRID.set_fresh_snow_props_update_time(dt)

        # Guarantee that solar radiation is greater equal zero
        G[t] = max(G[t],0.0)

        #--------------------------------------------
        # Merge grid layers, if necessary
        #--------------------------------------------
        if albedo_method =='Density_derived': rho = GRID.get_avg_density(50) # it creates slightly more smooth results for albedo creation, if surface density is sample here
        sdf_pre = GRID.get_node_sdf(0)
        
        GRID.update_grid()

        sdf_post = GRID.get_node_sdf(0)
        # if sdf_pre!=sdf_post:
        #     print(DATA.time.data[t])
        #     print('sdf_pre: ',sdf_pre)
        #     print('sdf_post: ',sdf_post)
        #--------------------------------------------
        # Calculate albedo and roughness length changes if first layer is snow
        #--------------------------------------------
        if albedo_method =='from_file':
            alpha, albedo_snow = DATA.AL.values[t], DATA.AL.values[t]
            
        elif albedo_method =='Density_derived':
            al=DATA.AL.values[t]
            if al <0: al= np.nan
            alpha, albedo_snow = updateAlbedo(GRID,surface_temperature,t,al,rho = rho)
                
        else:               
            if time_tephra_event:
                alpha, albedo_snow = updateAlbedo(GRID,surface_temperature,t,albedo_snow,rho = 1.0,tt = tte)
            else:
                alpha, albedo_snow = updateAlbedo(GRID,surface_temperature,t,albedo_snow,rho = 1.0)
        #--------------------------------------------
        # Update roughness length
        #--------------------------------------------
        z0 = updateRoughness(GRID)

        #--------------------------------------------
        # Surface Energy Balance
        #--------------------------------------------
        # Calculate net shortwave radiation
        SWnet = G[t] * (1 - alpha)

        # Penetrating SW radiation and subsurface melt
        if SWnet > 0.0:
            subsurface_melt, G_penetrating = penetrating_radiation(GRID, SWnet, dt)
        else:
            subsurface_melt = 0.0
            G_penetrating = 0.0

        # Calculate residual net shortwave radiation (penetrating part removed)
        sw_radiation_net = SWnet - G_penetrating
        
        #try:
        if (LWin is not None) and (N is not None)and hasattr(N, "__getitem__"):
            te=1  #couter to see what lwin is triggered

            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                 ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                 = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                 U2[t], RAIN, SLOPE, LWin=LWin[t], N=N[t], z_u = z_u)
        elif LWin is not None:
            te=2
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                                             U2[t], RAIN, SLOPE, LWin=LWin[t], z_u = z_u)
        else:
            te=3
            # Find new surface temperature (LW is parametrized using cloud fraction)
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                                             U2[t], RAIN, SLOPE, N=N[t], z_u = z_u)
        # except Exception as error:
        #     print(error)
        #     print(t)
        #     print(indY, indX)
        #--------------------------------------------
        # Surface mass fluxes [m w.e.q.]
        #--------------------------------------------
        if surface_temperature < zero_temperature:
            sublimation = min(latent_heat_flux / (water_density * lat_heat_sublimation), 0.) * dt
            deposition = max(latent_heat_flux / (water_density * lat_heat_sublimation), 0.) * dt
            evaporation = 0.
            condensation = 0.
        else:
            sublimation = 0.
            deposition = 0.
            evaporation = min(latent_heat_flux / (water_density * lat_heat_vaporize), 0.) * dt
            condensation = max(latent_heat_flux / (water_density * lat_heat_vaporize), 0.) * dt

        #--------------------------------------------
        # Melt process - mass changes of snowpack (melting, sublimation, deposition, evaporation, condensation)
        #--------------------------------------------
        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(
            0,
            sw_radiation_net
            + lw_radiation_in
            + lw_radiation_out
            + ground_heat_flux
            + rain_heat_flux
            + sensible_heat_flux
            + latent_heat_flux
        )
        
        # Convert melt energy to m w.e.q.
        melt = melt_energy * dt / (1000 * lat_heat_melting)

        # Remove melt [m w.e.q.]
        lwc_from_melted_layers = GRID.remove_melt_weq(melt - sublimation - deposition)

        #--------------------------------------------
        # Percolation
        #--------------------------------------------
        Q  = percolation(GRID, melt + condensation + RAIN/1000.0 + lwc_from_melted_layers, dt)

        #--------------------------------------------
        # Refreezing
        #--------------------------------------------
        water_refreezed = refreezing(GRID)

        #--------------------------------------------
        # Solve the heat equation
        #--------------------------------------------
        solveHeatEquation(GRID, dt)

        #--------------------------------------------
        # Calculate new density to densification
        #--------------------------------------------
        densification(GRID, SLOPE, dt,U2[t])

        #--------------------------------------------
        # Calculate mass balance
        #--------------------------------------------
        surface_mass_balance = (
            SNOWFALL * (density_fresh_snow / water_density)
            - melt
            + sublimation
            + deposition
            + evaporation
        )
        internal_mass_balance = water_refreezed - subsurface_melt
        mass_balance = surface_mass_balance + internal_mass_balance

        # internal_mass_balance2 = melt-Q  + subsurface_melt
        # mass_balance_check = surface_mass_balance + internal_mass_balance2

        # Cumulative mass balance for stake evaluation 
        MB_cum = MB_cum + mass_balance
        if Config.track_ages:
            GRID.update_node_ages()
        
        
        # Store cumulative MB in pandas frame for validation
        if stake_names:
            if DATA.isel(time=t).time.values in stake_data.index:
                _df['mb'].loc[DATA.isel(time=t).time.values] = MB_cum 
                _df['snowheight'].loc[DATA.isel(time=t).time.values] = GRID.get_total_snowheight()
        # Save results -- standalone cosipy case
        
        if not WRF_X_CSPY:
            idx = t // log_int
            if "RAIN"       in output_atm: results["RAIN"][idx]       += RAIN
            if "SNOWFALL"   in output_atm: results["SNOWFALL"][idx]   += SNOWFALL * (density_fresh_snow / water_density)
            if "LWin"       in output_atm: results["LWin"][idx]       += lw_radiation_in
            if "LWout"      in output_atm: results["LWout"][idx]      += lw_radiation_out
            if "H"          in output_atm: results["H"][idx]          += sensible_heat_flux
            if "LE"         in output_atm: results["LE"][idx]         += latent_heat_flux
            if "B"          in output_atm: results["B"][idx]          += ground_heat_flux
            if "QRR"        in output_atm: results["QRR"][idx]        += rain_heat_flux
            if "TS"         in output_atm: results["TS"][idx]         += surface_temperature
            if "ALBEDO"     in output_atm: results["ALBEDO"][idx]     += alpha
            if "Z0"         in output_atm: results["Z0"][idx]         += z0

            # Internal outputs (sum)
            if "MB"         in output_internal: results["MB"][idx]         += mass_balance
            if "surfMB"     in output_internal: results["surfMB"][idx]     += surface_mass_balance
            if "Q"          in output_internal: results["Q"][idx]          += Q
            if "RHO_AVG"    in output_internal: results["RHO_AVG"][idx]    += GRID.get_avg_density_nowater(Constants.m_lim)
            if "SDF"        in output_internal: results["SDF"][idx]        += GRID.get_node_sdf(0)
            if "ME"         in output_internal: results["ME"][idx]         += melt_energy
            if "intMB"      in output_internal: results["intMB"][idx]      += internal_mass_balance
            if "EVAPORATION"in output_internal: results["EVAPORATION"][idx]+= evaporation
            if "SUBLIMATION"in output_internal: results["SUBLIMATION"][idx]+= sublimation
            if "CONDENSATION" in output_internal: results["CONDENSATION"][idx] += condensation
            if "DEPOSITION" in output_internal: results["DEPOSITION"][idx] += deposition
            if "REFREEZE"   in output_internal: results["REFREEZE"][idx]   += water_refreezed
            if "subM"       in output_internal: results["subM"][idx]       += subsurface_melt
            if "surfM"      in output_internal: results["surfM"][idx]      += melt
            
            if "SNOWHEIGHT" in output_internal: results["SNOWHEIGHT"][idx] = GRID.get_total_snowheight()
            if "TOTALHEIGHT"in output_internal: results["TOTALHEIGHT"][idx] = GRID.get_total_height()
            if "LAYERS"     in output_internal: results["LAYERS"][idx]     = GRID.get_number_layers()
            if MOL is None: MOL=np.nan
            if "MOL"        in output_internal: results["MOL"][idx]        = MOL

            # Full‐field (2D) outputs (sum across layers)
            if (t+1) % log_int == 0:
                if Config.full_field:
                    nlay = GRID.get_number_layers()
                    if "LAYER_HEIGHT"         in output_full: results["LAYER_HEIGHT"][idx,:nlay]         = GRID.get_height()
                    if "LAYER_RHO"            in output_full: results["LAYER_RHO"][idx,:nlay]            = GRID.get_density()
                    if "LAYER_T"              in output_full: results["LAYER_T"][idx,:nlay]              = GRID.get_temperature()
                    if "LAYER_LWC"            in output_full: results["LAYER_LWC"][idx,:nlay]            = GRID.get_liquid_water_content()
                    if "LAYER_CC"             in output_full: results["LAYER_CC"][idx,:nlay]             = GRID.get_cold_content()
                    if "LAYER_POROSITY"       in output_full: results["LAYER_POROSITY"][idx,:nlay]       = GRID.get_porosity()
                    if "LAYER_ICE_FRACTION"   in output_full: results["LAYER_ICE_FRACTION"][idx,:nlay]   = GRID.get_ice_fraction()
                    if "LAYER_IRREDUCIBLE_WATER" in output_full: results["LAYER_IRREDUCIBLE_WATER"][idx,:nlay] = GRID.get_irreducible_water_content()
                    if "LAYER_REFREEZE"       in output_full: results["LAYER_REFREEZE"][idx,:nlay]       = GRID.get_refreeze()
                    if Config.track_ages:
                        if "LAYER_BURIAL_AGE" in output_full:    results["LAYER_BURIAL_AGE"][idx,:nlay]   = GRID.get_burial_age()
                        if "LAYER_CREATION_AGE" in output_full:  results["LAYER_CREATION_AGE"][idx,:nlay] = GRID.get_creation_age()
                for var in output_internal+output_atm:
                    if not var in presaved:
                        method=base_methods[var]
                        if method == 'mean':
                            results[var][idx] = results[var][idx] /log_int
        
# Save results -- WRF_X_CSPY case
        else:
            _SNOWHEIGHT = GRID.get_total_snowheight()
            _RHO_AVG = GRID.get_avg_density(Constants.m_lim)
            _SDF = GRID.get_node_sdf(0)
            _TOTALHEIGHT = GRID.get_total_height()
            _NLAYERS = GRID.get_number_layers()
            _new_snow_height, _new_snow_timestamp, _old_snow_timestamp = GRID.get_fresh_snow_props()

            _LAYER_HEIGHT = np.array(max_layers * [np.nan])
            _LAYER_RHO = np.array(max_layers * [np.nan])
            _LAYER_T = np.array(max_layers * [np.nan])
            _LAYER_LWC = np.array(max_layers * [np.nan])
            _LAYER_ICE_FRACTION = np.array(max_layers * [np.nan])
            if GRID.get_number_layers()>max_layers:
                raise ValueError('Maximum number of layers reached')
            _LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height()
            _LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density()
            _LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature()
            _LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
            _LAYER_ICE_FRACTION[0:GRID.get_number_layers()] = GRID.get_ice_fraction()

    gc.collect()
    if Config.stake_evaluation:
        # Evaluate stakes
        _stat = evaluate(stake_names, stake_data, _df)
    else:
        _stat = None
        _df = None
    
    _new_snow_height, _new_snow_timestamp, _old_snow_timestamp = GRID.get_fresh_snow_props()
    # Restart
    RESTART.NLAYERS.values[:] = GRID.get_number_layers()
    RESTART.NEWSNOWHEIGHT.values[:] = _new_snow_height
    RESTART.NEWSNOWTIMESTAMP.values[:] = _new_snow_timestamp
    RESTART.OLDSNOWTIMESTAMP.values[:] = _old_snow_timestamp
    RESTART.LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height()
    RESTART.LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density()
    RESTART.LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature()
    RESTART.LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
    RESTART.LAYER_IF[0:GRID.get_number_layers()] = GRID.get_ice_fraction()

    #print(f"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa te ist:", te) 

    
    # Return a filtered dictionary instead of a fixed tuple'
    if not WRF_X_CSPY:
        return indY, indX, RESTART, results, stake_names, _stat, _df
    else:
        return None, None, None, results, None, None, None

