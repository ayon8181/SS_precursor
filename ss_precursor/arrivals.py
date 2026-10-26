"""Select a unique underside-reflection ray; never select the first branch silently."""
import numpy as np
from .geometry import point_on_ray


def unique_reflection(model, depth, distance, phase, boundary, evlat, evlon, azimuth):
    arrivals = model.get_ray_paths(source_depth_in_km=depth, distance_in_degree=distance, phase_list=[phase])
    candidates = []
    for a in arrivals:
        p = a.path
        indices = np.where(np.isclose(p['depth'][1:-1], boundary, atol=1e-5)
                           & (p['depth'][:-2] > p['depth'][1:-1])
                           & (p['depth'][2:] > p['depth'][1:-1]))[0]+1
        if len(indices) == 1 and abs(np.rad2deg(p['dist'][-1])-distance) < 0.01:
            candidates.append((a, int(indices[0])))
    if len(candidates) != 1:
        raise ValueError(f"{phase}: expected unique minor-arc underside branch, found {len(candidates)} of {len(arrivals)} arrivals")
    a, i = candidates[0]
    lat, lon = point_on_ray(evlat, evlon, azimuth, np.rad2deg(a.path['dist'][i]))
    return {"phase": phase, "time_s": float(a.time), "ray_parameter_s_degree": float(a.ray_param_sec_degree),
            "reflection_latitude": lat, "reflection_longitude": lon, "boundary_km": float(boundary)}


def predictions(model, origin, distance, azimuth, interfaces, interfering):
    args = (model, origin.depth/1000, distance)
    output = {"SS": unique_reflection(*args, "SS", 0, origin.latitude, origin.longitude, azimuth)}
    discontinuities = model.model.s_mod.v_mod.get_discontinuity_depths()
    for h in interfaces:
        if not np.any(np.isclose(discontinuities, h, atol=1e-5)):
            raise ValueError(f"Requested interface {h} km absent from reference model")
        name = f"S^{h:g}S"
        output[name] = unique_reflection(*args, name, h, origin.latitude, origin.longitude, azimuth)
    interference = []
    if interfering:
        for a in model.get_travel_times(origin.depth/1000, distance, phase_list=interfering):
            interference.append({"phase": a.name, "time_s": float(a.time),
                                 "ray_parameter_s_degree": float(a.ray_param_sec_degree)})
    return output, interference
