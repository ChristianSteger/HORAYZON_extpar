# Description: Miscellaneous functions
#
# Author: Christian R. Steger (Christian.Steger@meteoswiss.ch), April 2025

import numpy as np
from pyproj import Transformer

# -----------------------------------------------------------------------------

def observer_perspective(lon: np.ndarray, lat: np.ndarray,
                         elevation: np.ndarray, lon_obs: float, lat_obs: float,
                         elevation_obs:float) -> tuple:
    """Transform points to 'observer perspective'. Latitude/longitude -> ECEF
    -> ENU -> spherical coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array with longitude of points [rad]
    lat : ndarray of double
        Array with latitude of points [rad]
    elevation : ndarray of float
        Array with elevation of points [m]
    lon_obs : double
        Longitude of 'observer' [rad]
    lat_obs : double
        Latitude of 'observer' [rad]
    elevation_obs : float
        Elevation of 'observer' [m]

    Returns
    -------
    phi : ndarray of double
        Array with azimuth angles [deg]
    theta : ndarray of double
        Array with elevation angles [deg]
    radius : ndarray of double
        Array with radii [m]"""

    txt = "+proj=pipeline +step +proj=cart +ellps=sphere +R=6371229.0" \
          + " +step +proj=topocentric +ellps=sphere +R=6371229.0" \
          + " +lon_0=" + str(np.rad2deg(lon_obs)) \
          + " +lat_0=" + str(np.rad2deg(lat_obs)) \
          + " +h_0=" + str(elevation_obs)
    t = Transformer.from_pipeline(txt)
    x, y, z = t.transform(np.rad2deg(lon), np.rad2deg(lat), elevation)
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.rad2deg(np.arccos(z / radius))  # zenith angle [90.0, 0.0 deg]
    theta = 90.0 - theta  # elevation angle [0.0, 90 deg]
    phi = np.rad2deg(np.arctan2(x, y))  # azimuth angle [-180.0, +180.0 deg]
    phi[phi < 0.0] += 360.0  # [0.0, 360.0 deg]

    return phi, theta, radius

# -----------------------------------------------------------------------------
