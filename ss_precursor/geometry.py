"""Spherical ray geometry; bins are angular cells, not equal-area cells."""
import numpy as np


def geographic_bin(latitude, longitude, size=5):
    if not np.isfinite(latitude) or not -90 <= latitude <= 90 or not np.isfinite(longitude):
        raise ValueError("Invalid coordinates")
    if size <= 0 or not np.isclose(180/size, round(180/size)):
        raise ValueError("Bin size must divide 180 degrees")
    lon = (longitude+180) % 360-180
    row = min(int(np.floor((latitude+90)/size)), round(180/size)-1)
    col = int(np.floor((lon+180)/size))
    return row, col


def point_on_ray(latitude, longitude, azimuth, arc_degrees):
    """Great-circle destination, consistent with spherical TauP distances."""
    lat, lon, az, arc = np.deg2rad([latitude, longitude, azimuth, arc_degrees])
    lat2 = np.arcsin(np.clip(np.sin(lat)*np.cos(arc)+np.cos(lat)*np.sin(arc)*np.cos(az), -1, 1))
    lon2 = lon+np.arctan2(np.sin(az)*np.sin(arc)*np.cos(lat), np.cos(arc)-np.sin(lat)*np.sin(lat2))
    return float(np.rad2deg(lat2)), float((np.rad2deg(lon2)+180) % 360-180)


def spherical_geometry(evlat, evlon, stlat, stlon):
    from obspy.geodetics import locations2degrees
    def bearing(a, b, c, d):
        a, b, c, d = np.deg2rad([a, b, c, d])
        return float(np.rad2deg(np.arctan2(np.sin(d-b)*np.cos(c),
                     np.cos(a)*np.sin(c)-np.sin(a)*np.cos(c)*np.cos(d-b))) % 360)
    return float(locations2degrees(evlat, evlon, stlat, stlon)), bearing(evlat, evlon, stlat, stlon), bearing(stlat, stlon, evlat, evlon)
