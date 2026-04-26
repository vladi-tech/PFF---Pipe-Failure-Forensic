#!/usr/bin/env python3
"""
prepare_road_buffer.py

Creates a road-buffer AOI for later SNAP subsetting.

This script:
1. Reads config.yaml.
2. Reads the main AOI polygon.
3. Reads the road network GeoJSON.
4. Clips roads to the AOI.
5. Reprojects roads to a metric CRS.
6. Buffers roads by the configured distance, usually 5 m.
7. Dissolves the buffer into one polygon/multipolygon.
8. Simplifies the buffer to make SNAP WKT easier to handle.
9. Writes:
   - clipped roads GeoJSON
   - road buffer GeoJSON
   - road buffer GeoPackage
   - road buffer WKT

Install:
    pip install geopandas shapely pyproj fiona PyYAML

Run:
    python prepare_road_buffer.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import yaml
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def die(message: str, exit_code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        die(f"YAML file is empty or invalid: {path}")

    return data


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config

    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]

    return cur


def read_aoi_as_gdf(path: str | Path) -> gpd.GeoDataFrame:
    """
    Reads AOI GeoJSON as a GeoDataFrame.

    Accepts:
    - FeatureCollection
    - Feature
    - Polygon
    - MultiPolygon
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geojson_type = data.get("type")

    if geojson_type == "FeatureCollection":
        gdf = gpd.read_file(path)

    elif geojson_type == "Feature":
        geom = shape(data["geometry"])
        props = data.get("properties", {}) or {}
        gdf = gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

    elif geojson_type in {"Polygon", "MultiPolygon"}:
        geom = shape(data)
        gdf = gpd.GeoDataFrame([{"name": "aoi"}], geometry=[geom], crs="EPSG:4326")

    else:
        die("AOI must be a Polygon, MultiPolygon, Feature, or FeatureCollection.")

    if gdf.empty:
        die("AOI file contains no features.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")

    gdf["geometry"] = gdf.geometry.apply(fix_geometry)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    if gdf.empty:
        die("AOI geometry is empty or invalid after repair.")

    return gdf


def fix_geometry(geom: BaseGeometry) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def normalize_roads_gdf(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads.empty:
        die("Road network file contains no features.")

    if roads.crs is None:
        print("WARNING: Roads file has no CRS. Assuming EPSG:4326.")
        roads = roads.set_crs("EPSG:4326")

    roads = roads.to_crs("EPSG:4326")
    roads["geometry"] = roads.geometry.apply(fix_geometry)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()

    if roads.empty:
        die("Road network contains no valid geometries after repair.")

    allowed_types = {
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }

    roads = roads[roads.geometry.geom_type.isin(allowed_types)].copy()

    if roads.empty:
        die(
            "Road network has no LineString, MultiLineString, Polygon, or MultiPolygon geometries."
        )

    return roads


def cap_style_to_int(value: str) -> int:
    """
    Shapely buffer cap styles:
    1 = round
    2 = flat
    3 = square
    """
    value = (value or "round").lower().strip()

    mapping = {
        "round": 1,
        "flat": 2,
        "square": 3,
    }

    if value not in mapping:
        die("road_buffer.cap_style must be one of: round, flat, square")

    return mapping[value]


def join_style_to_int(value: str) -> int:
    """
    Shapely buffer join styles:
    1 = round
    2 = mitre
    3 = bevel
    """
    value = (value or "round").lower().strip()

    mapping = {
        "round": 1,
        "mitre": 2,
        "miter": 2,
        "bevel": 3,
    }

    if value not in mapping:
        die("road_buffer.join_style must be one of: round, mitre, bevel")

    return mapping[value]


def write_wkt(path: str | Path, geom: BaseGeometry) -> None:
    ensure_dir(Path(path).parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(geom.wkt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )

    args = parser.parse_args()
    config = read_yaml(args.config)

    aoi_file = get_nested(config, "project", "aoi_file", default="aoi.geojson")
    roads_file = get_nested(config, "road_buffer", "roads_file", default="roads.geojson")

    output_clipped_roads_geojson = get_nested(
        config,
        "road_buffer",
        "output_clipped_roads_geojson",
        default="outputs/clipped_roads.geojson",
    )
    output_buffer_geojson = get_nested(
        config,
        "road_buffer",
        "output_buffer_geojson",
        default="outputs/road_buffer_5m.geojson",
    )
    output_buffer_gpkg = get_nested(
        config,
        "road_buffer",
        "output_buffer_gpkg",
        default="outputs/road_buffer_5m.gpkg",
    )
    output_buffer_wkt = get_nested(
        config,
        "road_buffer",
        "output_buffer_wkt",
        default="outputs/road_buffer_5m.wkt",
    )

    metric_crs = get_nested(
        config,
        "road_buffer",
        "metric_crs",
        default="EPSG:32635",
    )
    buffer_distance_m = float(
        get_nested(config, "road_buffer", "buffer_distance_m", default=5)
    )
    simplify_tolerance_m = float(
        get_nested(config, "road_buffer", "simplify_tolerance_m", default=0.5)
    )
    cap_style = cap_style_to_int(
        get_nested(config, "road_buffer", "cap_style", default="round")
    )
    join_style = join_style_to_int(
        get_nested(config, "road_buffer", "join_style", default="round")
    )

    if buffer_distance_m <= 0:
        die("road_buffer.buffer_distance_m must be greater than zero.")

    if simplify_tolerance_m < 0:
        die("road_buffer.simplify_tolerance_m cannot be negative.")

    if not Path(aoi_file).exists():
        die(f"AOI file not found: {aoi_file}")

    if not Path(roads_file).exists():
        die(f"Road network file not found: {roads_file}")

    print("Reading AOI...")
    aoi = read_aoi_as_gdf(aoi_file)

    print("Reading road network...")
    roads = gpd.read_file(roads_file)
    roads = normalize_roads_gdf(roads)

    print("Clipping roads to AOI...")
    aoi_union = unary_union(aoi.geometry)
    aoi_gdf = gpd.GeoDataFrame([{"name": "aoi"}], geometry=[aoi_union], crs="EPSG:4326")

    try:
        clipped_roads = gpd.clip(roads, aoi_gdf)
    except Exception:
        # Fallback for environments where spatial index or clip behavior is problematic.
        roads_intersecting = roads[roads.intersects(aoi_union)].copy()
        roads_intersecting["geometry"] = roads_intersecting.geometry.intersection(aoi_union)
        clipped_roads = roads_intersecting

    clipped_roads = clipped_roads[
        clipped_roads.geometry.notna() & ~clipped_roads.geometry.is_empty
    ].copy()

    if clipped_roads.empty:
        die("No road features intersect the AOI. Check roads.geojson and aoi.geojson.")

    ensure_dir(Path(output_clipped_roads_geojson).parent)
    clipped_roads.to_file(output_clipped_roads_geojson, driver="GeoJSON")

    print(f"Clipped road features: {len(clipped_roads)}")
    print(f"Wrote clipped roads: {output_clipped_roads_geojson}")

    print(f"Reprojecting clipped roads to metric CRS: {metric_crs}")
    clipped_roads_metric = clipped_roads.to_crs(metric_crs)

    print(f"Creating {buffer_distance_m} m road buffer...")
    buffered_geometries = clipped_roads_metric.geometry.buffer(
        buffer_distance_m,
        cap_style=cap_style,
        join_style=join_style,
    )

    buffer_union = unary_union(buffered_geometries)

    if buffer_union.is_empty:
        die("Road buffer is empty.")

    if not buffer_union.is_valid:
        buffer_union = buffer_union.buffer(0)

    if simplify_tolerance_m > 0:
        print(f"Simplifying buffer with tolerance {simplify_tolerance_m} m...")
        buffer_union = buffer_union.simplify(
            simplify_tolerance_m,
            preserve_topology=True,
        )

        if not buffer_union.is_valid:
            buffer_union = buffer_union.buffer(0)

    if buffer_union.is_empty:
        die("Road buffer became empty after simplification.")

    buffer_metric_gdf = gpd.GeoDataFrame(
        [
            {
                "name": "road_buffer",
                "buffer_m": buffer_distance_m,
                "simplify_m": simplify_tolerance_m,
                "metric_crs": metric_crs,
            }
        ],
        geometry=[buffer_union],
        crs=metric_crs,
    )

    buffer_4326_gdf = buffer_metric_gdf.to_crs("EPSG:4326")
    buffer_4326_geom = buffer_4326_gdf.geometry.iloc[0]

    ensure_dir(Path(output_buffer_geojson).parent)
    buffer_4326_gdf.to_file(output_buffer_geojson, driver="GeoJSON")

    ensure_dir(Path(output_buffer_gpkg).parent)
    buffer_4326_gdf.to_file(output_buffer_gpkg, layer="road_buffer_5m", driver="GPKG")

    write_wkt(output_buffer_wkt, buffer_4326_geom)

    print("Done.")
    print(f"Road buffer GeoJSON: {output_buffer_geojson}")
    print(f"Road buffer GeoPackage: {output_buffer_gpkg}")
    print(f"Road buffer WKT: {output_buffer_wkt}")
    print()
    print("This WKT will be used by SNAP GPT for Sentinel product subsetting.")


if __name__ == "__main__":
    main()