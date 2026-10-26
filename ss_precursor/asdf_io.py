"""Read-only ASDF access and explicit event/origin association."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def open_asdf(path):
    from pyasdf import ASDFDataSet
    ds = ASDFDataSet(str(path), mode="r", mpi=False)
    try:
        yield ds
    finally:
        # Explicit close also works with older pyasdf context-manager versions.
        ds._close()


def select_event(ds, event_id=None, origin_policy="preferred"):
    events = [e for e in ds.events if event_id is None or str(e.resource_id) == event_id]
    if len(events) != 1:
        raise ValueError("Select one event explicitly with its full resource ID")
    event = events[0]
    if origin_policy == "preferred":
        origin = event.preferred_origin()
    else:
        origins = [o for o in event.origins if str(o.resource_id) == origin_policy]
        origin = origins[0] if len(origins) == 1 else None
    if origin is None or any(getattr(origin, k) is None for k in ("time", "latitude", "longitude", "depth")):
        raise ValueError("Selected origin missing or incomplete; configure an explicit origin ID")
    return event, origin


def associated_stream(group, tag, event_id, event_count):
    stream = group[tag]
    selected = stream.__class__()
    for tr in stream:
        ids = [str(v) for v in getattr(getattr(tr.stats, "asdf", {}), "event_ids", [])]
        if len(ids) > 1:
            raise ValueError("Waveform has ambiguous multiple-event association")
        if ids and event_id not in ids:
            continue
        if not ids and event_count != 1:
            raise ValueError("Waveform lacks event association in multi-event file")
        selected += tr
    if not selected:
        raise ValueError("No waveform associated with selected event")
    return selected


def inspect_file(path, station_limit=None):
    with open_asdf(path) as ds:
        events = []
        for e in ds.events:
            events.append({"id": str(e.resource_id), "preferred_origin_id": str(e.preferred_origin_id),
                "preferred_focal_mechanism_id": str(e.preferred_focal_mechanism_id),
                "origins": [{"id": str(o.resource_id), "type": str(o.origin_type),
                    "time": str(o.time), "latitude": o.latitude, "longitude": o.longitude,
                    "depth_km": None if o.depth is None else o.depth/1000} for o in e.origins],
                "focal_mechanisms": [f.__str__() for f in e.focal_mechanisms]})
        names = ds.waveforms.list()
        rows = []
        for name in names[:station_limit]:
            row = {"station": name}
            try:
                group = ds.waveforms[name]
                row.update(coordinates=group.coordinates, tags=group.get_waveform_tags(), traces=[])
                for tag in row["tags"]:
                    for tr in group[tag]:
                        row["traces"].append({"tag": tag, "id": tr.id,
                            "start": str(tr.stats.starttime), "delta_s": tr.stats.delta,
                            "duration_s": (tr.stats.npts-1)*tr.stats.delta, "npts": tr.stats.npts,
                            "units": getattr(tr.stats, "units", None),
                            "event_ids": [str(v) for v in getattr(tr.stats.asdf, "event_ids", [])],
                            "start_from_origins_s": {str(o.resource_id): float(tr.stats.starttime-o.time)
                                for e in ds.events for o in e.origins if o.time is not None}})
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return {"path": str(path.resolve()), "events": events, "station_count": len(names), "station_ids": names,
                "stations_inspected": len(rows), "stations": rows}
