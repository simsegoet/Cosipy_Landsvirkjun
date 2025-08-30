import numpy as np

from cosipy.constants import Constants

# only required for njitted functions
albedo_method = Constants.albedo_method
#albedo_mod_snow_aging = Constants.albedo_mod_snow_aging
snow_ice_threshold = Constants.snow_ice_threshold
#albedo_firn = Constants.albedo_firn
#albedo_fresh_snow = Constants.albedo_fresh_snow
#albedo_ice = Constants.albedo_ice
#albedo_mod_snow_depth = Constants.albedo_mod_snow_depth
dt = Constants.dt
zero_temperature = Constants.zero_temperature
#t_star_cutoff = Constants.t_star_cutoff
#t_star_dry = Constants.t_star_dry
#t_star_wet = Constants.t_star_wet
#t_star_K = Constants.t_star_K
albedo_dust = Constants.albedo_dust

tephra_params = {
    'albedo_mod_snow_aging': Constants.tephra_albedo_mod_snow_aging,
    'albedo_firn': Constants.tephra_albedo_firn,
    'albedo_fresh_snow': Constants.tephra_albedo_fresh_snow,
    'albedo_ice': Constants.tephra_albedo_ice,
    'albedo_mod_snow_depth': Constants.tephra_albedo_mod_snow_depth,
    't_star_cutoff': Constants.tephra_t_star_cutoff,
    't_star_dry': Constants.tephra_t_star_dry,
    't_star_wet': Constants.tephra_t_star_wet,
    't_star_K': Constants.tephra_t_star_K,
}

default_params = {
    'albedo_mod_snow_aging': Constants.albedo_mod_snow_aging,
    'albedo_firn': Constants.albedo_firn,
    'albedo_fresh_snow': Constants.albedo_fresh_snow,
    'albedo_ice': Constants.albedo_ice,
    'albedo_mod_snow_depth': Constants.albedo_mod_snow_depth,
    't_star_cutoff': Constants.t_star_cutoff,
    't_star_dry': Constants.t_star_dry,
    't_star_wet': Constants.t_star_wet,
    't_star_K': Constants.t_star_K,
}


def updateAlbedo(
    GRID, surface_temperature: float, t: int,albedo_snow: float, rho: float = None, tt:int = None
) -> tuple:
    """Update the surface albedo.

    Implemented albedo methods:

        - **Oerlemans98**: Oerlemans & Knap (1998)
        - **Bougamont05**: Bougamont et al. (2005)
        - **Density_derived**: Jonas Liebsch

    Args:
        GRID (Grid): Glacier data structure.
        surface_temperature: Surface temperature.
        albedo_snow: Initial snow albedo.

    Returns:
        tuple[float,float]: Updated surface albedo and snow albedo.

    Raises:
        NotImplementedError: Albedo method is not allowed.
    """
    params = tephra_params if (tt is not None and t > tt) else default_params
    albedo_allowed = ["Oerlemans98", "Bougamont05", "Density_derived"]
    if albedo_method == "Oerlemans98":
        alphaMod = method_Oerlemans(GRID, params)
    elif albedo_method == "Bougamont05":
        alphaMod, albedo_snow = method_Bougamont(GRID, surface_temperature, albedo_snow, params)
    elif albedo_method == "Density_derived":
        alphaMod = method_Density_derived(GRID, albedo_snow, rho)
    elif albedo_method == "from_file":
        alphaMod = method_from_file(GRID, albedo_snow, params)
    else:
        error_message = (
            f'Albedo method = "{albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(albedo_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return alphaMod, albedo_snow


def get_surface_properties(GRID, albedo_mod_snow_aging) -> tuple:
    """Get snowpack properties.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        tuple[float, float, float]: Height and timestamp of fresh snow,
        and the hours elapsed since the last snowfall.
    """

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = fresh_snow_timestamp / 3600.0

    """If fresh snow disappears faster than the snow ageing scale then
    set the hours_since_snowfall to the old values of the underlying
    snowpack."""
    if (hours_since_snowfall < (albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0
    ):
        GRID.set_fresh_snow_props_to_old_props()
        (
            fresh_snow_height,
            fresh_snow_timestamp,
            _,
        ) = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = fresh_snow_timestamp / 3600.0

    return fresh_snow_height, fresh_snow_timestamp, hours_since_snowfall





def get_simple_albedo(elapsed_time: float, albedo_firn, albedo_fresh_snow, albedo_mod_snow_aging) -> float:
    """Get surface albedo neglecting snowpack depth.

    From Oerlemans & Knap (1998).

    Args:
        elapsed_time: Hours elapsed since last snowfall.

    Returns:
        Surface albedo without accounting for snowpack depth.
    """

    albedo = albedo_firn + (albedo_fresh_snow - albedo_firn) * np.exp(
        (-elapsed_time) / (albedo_mod_snow_aging * 24.0)
    )

    return albedo


def get_albedo_with_decay(snow_albedo: float, snow_height: float, albedo_ice, albedo_mod_snow_depth) -> float:
    """Apply surface albedo decay due to the snow depth.

    Taken from Oerlemans & Knap (1998).

    Args:
        snow_albedo: Initial snow albedo.
        snow_height: Height of snowpack.

    Returns:
        Surface albedo with snow depth decay.
    """
    albedo = snow_albedo + (albedo_ice - snow_albedo) * np.exp(
        (-1.0 * snow_height) / (albedo_mod_snow_depth / 100.0)
    )

    return albedo


def method_Oerlemans(GRID, params) -> float:
    """Get surface albedo using method from Oerlemans & Knap (1998).

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Surface albedo.
    """
    _, _, hours_since_snowfall = get_surface_properties(GRID, params['albedo_mod_snow_aging'])

    # Check if snow or ice
    if GRID.get_node_density(0) <= snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()  # np.sum(GRID.get_height()[0:idx])

        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = get_simple_albedo(hours_since_snowfall, params['albedo_firn'], params['albedo_fresh_snow'], params['albedo_mod_snow_aging'])
        alphaMod = get_albedo_with_decay(alphaSnow, h, params['albedo_ice'], params['albedo_mod_snow_depth'])

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = params['albedo_ice']

    return alphaMod


def method_Bougamont(GRID, surface_temperature: float, albedo_snow: float, params):
    """Get surface and snow albedos using method from Bougamont (2005).

    Args:
        GRID (Grid): Glacier data structure.
        surface_temperature: Surface temperature.
        albedo_snow: Initial snow albedo.

    Returns:
        tuple[float, float]: Updated surface and snow albedos.

    TODO: Account for disappearance of uppermost fresh snow layer is
    difficult due to non-constant decay rate. Unsure how to implement.
    """
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, _, hours_since_snowfall = get_surface_properties(GRID, params['albedo_mod_snow_aging'])

    # Convert integration time from seconds to days:
    dt_days =  86400.0 /dt

    # Check if snow or ice:
    if GRID.get_node_density(0) <= snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()

        if surface_temperature >= zero_temperature:
            # Snow albedo decay timescale (t*) on a melting snow surface:
            t_star = params['t_star_wet']
        else:
            # Snow albedo decay timescale (t*) on a dry snow surface:
            if surface_temperature < params['t_star_cutoff']:
                t_star = (
                    params['t_star_dry'] + (zero_temperature - params['t_star_cutoff']) * params['t_star_K']
                )
            else:
                t_star = (
                    params['t_star_dry']
                    + (zero_temperature - surface_temperature) * params['t_star_K']
                )

        """Effect of snow albedo decay due to the temporal metamorphosis of
        snow (Bougamont et al. 2005 - based off Oerlemans & Knap 1998):
        Exponential function discretised to account for variable surface
        temperature-dependant decay timescales."""

        # slightly faster than one-liner
        t_star_days = float(t_star) * float(dt_days)
        delta_albedo = albedo_snow - params['albedo_firn']
        albedo_snow -= delta_albedo / t_star_days
        

        # Reset if snowfall in current timestep
        if hours_since_snowfall == 0:
            albedo_snow = params['albedo_fresh_snow']

        # Effect of surface albedo decay due to the snow depth (Oerlemans & Knap 1998):
        alphaMod = get_albedo_with_decay(albedo_snow, h, params['albedo_ice'], params['albedo_mod_snow_depth'])

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = params['albedo_ice']

    # Ensure output value is of the float data type.
    alphaMod = float(alphaMod)  # faster than isinstance

    return alphaMod, albedo_snow

def rho2al(rho):
    return -5.56220168e-12*rho**4 +  1.27091544e-08*rho**3 -9.48318922e-06*rho**2 +  1.76457100e-03*rho + 7.67773797e-01

def method_Density_derived(GRID, albedo_obs, rho_surf)-> float:
    #rho_surf = GRID.get_avg_density(m_lim)
    sdf = GRID.get_node_sdf(0)
    albedo_clean_snow = rho2al(rho_surf)
    
    if  not np.isnan(albedo_obs) and albedo_obs != 0:
        new_sdf = (albedo_clean_snow-albedo_obs)/(albedo_clean_snow - albedo_dust)
        if new_sdf>=0 and new_sdf <=1:
            GRID.set_sdf_from_surf(new_sdf)
            sdf = new_sdf
        elif new_sdf>=-0.1 and new_sdf <0:
            GRID.set_sdf_from_surf(0)
            sdf = new_sdf

        
    return albedo_clean_snow*(1-sdf)+albedo_dust*(sdf)

def method_from_file(GRID, albedo_from_file, params):
    if not np.isnan(albedo_from_file):
        return float(albedo_from_file)
    else:
        return method_Oerlemans(GRID, params)
        


### idea; albedo decay like (Brock et al. 2000)? or?
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
