"""TLC taxi-zone shapefile -> app/data/zones.geojson (WGS84, simplified).

The shapefile is in NY State Plane (feet). Zones that come as several
records (e.g. 56, 103) are merged into one feature per LocationID, and each
gets a point inside it (``lon``/``lat``) for the map's demand columns.

    .venv/bin/python scripts/build_zone_geojson.py data/zones/shapes/taxi_zones/taxi_zones.shp
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

SIMPLIFY_FEET = 40
OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "zones.geojson"


def _round(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 5), round(coords[1], 5)]
    return [_round(c) for c in coords]


def main(shp_path: str):
    reader = shapefile.Reader(shp_path)
    fields = [f[0] for f in reader.fields[1:]]
    to_wgs84 = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    def reproject(xy):
        return np.column_stack(to_wgs84.transform(xy[:, 0], xy[:, 1]))

    parts, props = defaultdict(list), {}
    for rec in reader.shapeRecords():
        attrs = dict(zip(fields, rec.record))
        zone_id = int(attrs["LocationID"])
        parts[zone_id].append(shape(rec.shape.__geo_interface__))
        props[zone_id] = {"zone_id": zone_id, "zone": attrs["zone"], "borough": attrs["borough"]}

    features = []
    for zone_id in sorted(parts):
        geom = unary_union(parts[zone_id]).simplify(SIMPLIFY_FEET, preserve_topology=True)
        geom = shapely.transform(geom, reproject)
        center = geom.representative_point()  # inside the zone, unlike a centroid
        geom = mapping(geom)
        geom["coordinates"] = _round(geom["coordinates"])
        properties = {**props[zone_id], "lon": round(center.x, 5), "lat": round(center.y, 5)}
        features.append({"type": "Feature", "properties": properties, "geometry": geom})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    print(f"{len(features)} zones -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
