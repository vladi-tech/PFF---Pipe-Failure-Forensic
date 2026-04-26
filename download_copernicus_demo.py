#!/usr/bin/env python3
"""
download_copernicus_demo.py

Minimal SNAP-friendly Copernicus Data Space downloader.

This script:
1. Reads config.yaml.
2. Reads aoi.geojson.
3. Queries Copernicus Data Space OData.
4. Downloads original .zip products.
5. Writes:
   - outputs/catalogue_results.csv
   - outputs/download_statistics_by_date.csv
   - outputs/download_log.csv
   - outputs/footprints.geojson

SNAP compatibility:
- Products are downloaded as original .zip files.
- The script does not unzip, clip, convert, or make COGs.
- Road-buffer subsetting will be handled later by SNAP GPT.

Install:
    pip install requests shapely PyYAML tqdm python-dotenv

Run:
    python download_copernicus_demo.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


CATALOGUE_BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_BASE_URL = "https://zipper.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

DEFAULT_TIMEOUT = 120
CHUNK_SIZE = 1024 * 1024


@dataclass
class Product:
    product_id: str
    name: str
    collection: str
    product_type: Optional[str]
    pixel_size_m: Optional[str]
    acquisition_start: Optional[str]
    acquisition_end: Optional[str]
    acquisition_date: Optional[str]
    publication_date: Optional[str]
    cloud_cover: Optional[float]
    size_bytes: Optional[int]
    size_mb: Optional[float]
    footprint_geojson: Optional[Dict[str, Any]]
    download_url: str
    status: str = "catalogued"
    local_path: Optional[str] = None
    error_message: Optional[str] = None


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


def read_aoi_geometry(path: str | Path) -> BaseGeometry:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geojson_type = data.get("type")

    if geojson_type == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            die("AOI FeatureCollection has no features.")
        geom = shape(features[0]["geometry"])

    elif geojson_type == "Feature":
        geom = shape(data["geometry"])

    elif geojson_type in {"Polygon", "MultiPolygon"}:
        geom = shape(data)

    else:
        die("AOI must be a Polygon, MultiPolygon, Feature, or FeatureCollection.")

    if geom.is_empty:
        die("AOI geometry is empty.")

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty or not geom.is_valid:
        die("AOI geometry is invalid and could not be fixed.")

    return geom


def geometry_to_odata_wkt(geom: BaseGeometry) -> str:
    return geom.wkt


def iso_start(date_text: str) -> str:
    return f"{date_text}T00:00:00.000Z"


def iso_end(date_text: str) -> str:
    return f"{date_text}T23:59:59.999Z"


def parse_datetime_to_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return value[:10] if len(value) >= 10 else None


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config

    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]

    return cur


def get_access_token(username: str, password: str) -> str:
    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(
        TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=DEFAULT_TIMEOUT,
    )

    if not response.ok:
        die(
            "Could not obtain Copernicus access token. "
            f"HTTP {response.status_code}: {response.text[:500]}"
        )

    token = response.json().get("access_token")

    if not token:
        die("Token response did not contain access_token.")

    return token


def attribute_filter_string(attribute_name: str, value: str) -> str:
    escaped_name = attribute_name.replace("'", "''")
    escaped_value = value.replace("'", "''")

    return (
        "Attributes/OData.CSC.StringAttribute/any(att:"
        f"att/Name eq '{escaped_name}' and "
        f"att/OData.CSC.StringAttribute/Value eq '{escaped_value}')"
    )


def attribute_filter_double_le(attribute_name: str, value: float) -> str:
    escaped_name = attribute_name.replace("'", "''")

    return (
        "Attributes/OData.CSC.DoubleAttribute/any(att:"
        f"att/Name eq '{escaped_name}' and "
        f"att/OData.CSC.DoubleAttribute/Value le {float(value)})"
    )


def build_odata_filter(
    collection_name: str,
    start_date: str,
    end_date: str,
    aoi_wkt: str,
    product_type: Optional[str] = None,
    max_cloud_cover: Optional[float] = None,
) -> str:
    clauses = [
        f"Collection/Name eq '{collection_name}'",
        f"ContentDate/Start gt {iso_start(start_date)}",
        f"ContentDate/Start lt {iso_end(end_date)}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
    ]

    if product_type:
        clauses.append(attribute_filter_string("productType", product_type))

    if max_cloud_cover is not None:
        clauses.append(attribute_filter_double_le("cloudCover", max_cloud_cover))

    return " and ".join(clauses)


def query_odata_products(
    collection_name: str,
    start_date: str,
    end_date: str,
    aoi_wkt: str,
    product_type: Optional[str],
    max_cloud_cover: Optional[float],
    top: int,
    print_queries: bool = False,
) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []

    filter_text = build_odata_filter(
        collection_name=collection_name,
        start_date=start_date,
        end_date=end_date,
        aoi_wkt=aoi_wkt,
        product_type=product_type,
        max_cloud_cover=max_cloud_cover,
    )

    if print_queries:
        print()
        print("QUERY FILTER:")
        print(filter_text)
        print()

    params = {
        "$filter": filter_text,
        "$orderby": "ContentDate/Start asc",
        "$top": str(top),
        "$expand": "Attributes",
    }

    url = f"{CATALOGUE_BASE_URL}/Products"
    first_request = True

    while url:
        if first_request:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            first_request = False
        else:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)

        if not response.ok:
            print(
                f"WARNING: Query failed for collection={collection_name}, "
                f"product_type={product_type}. "
                f"HTTP {response.status_code}: {response.text[:700]}",
                file=sys.stderr,
            )
            break

        data = response.json()
        batch = data.get("value", [])

        if not isinstance(batch, list):
            print(
                f"WARNING: Unexpected response shape for collection={collection_name}.",
                file=sys.stderr,
            )
            break

        products.extend(batch)

        if len(products) >= top:
            products = products[:top]
            break

        url = data.get("@odata.nextLink")

    return products


def get_attribute(product: Dict[str, Any], name: str) -> Any:
    attributes = product.get("Attributes", []) or []

    for attr in attributes:
        if not isinstance(attr, dict):
            continue

        if attr.get("Name") != name:
            continue

        if "Value" in attr:
            return attr["Value"]

        for key, value in attr.items():
            if key.lower().endswith("value"):
                return value

    return None


def get_product_size_bytes(product: Dict[str, Any]) -> Optional[int]:
    content_length = product.get("ContentLength")

    if isinstance(content_length, int):
        return content_length

    size_attr = get_attribute(product, "size")

    if size_attr is None:
        return None

    if isinstance(size_attr, (int, float)):
        return int(size_attr)

    if isinstance(size_attr, str):
        text = size_attr.strip().upper()

        try:
            if text.endswith("GB"):
                return int(float(text[:-2].strip()) * 1024**3)
            if text.endswith("MB"):
                return int(float(text[:-2].strip()) * 1024**2)
            if text.endswith("KB"):
                return int(float(text[:-2].strip()) * 1024)
            if text.endswith("B"):
                return int(float(text[:-1].strip()))
            return int(float(text))
        except Exception:
            return None

    return None


def extract_footprint(product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    geo = product.get("GeoFootprint")

    if isinstance(geo, dict):
        return geo

    footprint = product.get("Footprint")

    if isinstance(footprint, dict):
        return footprint

    return None


def extract_product_type(product: Dict[str, Any]) -> Optional[str]:
    product_type = get_attribute(product, "productType")

    if product_type:
        return str(product_type)

    name = product.get("Name") or ""

    if name.startswith("S1") and len(name) >= 14:
        # Example:
        # S1A_IW_GRDH_1SDV_...
        # This fallback is approximate but useful for CSV reporting.
        return name[4:14].strip("_")

    if "_MSIL1C_" in name:
        return "S2MSI1C"

    if "_MSIL2A_" in name:
        return "S2MSI2A"

    return None


def infer_pixel_size_m(
    collection: str,
    product_type: Optional[str],
    product_name: Optional[str],
    config: Dict[str, Any],
) -> Optional[str]:
    """
    Returns a human-readable nominal pixel size.

    Sentinel-2 full products contain multiple band resolutions:
    10 m, 20 m, and 60 m.

    Sentinel-1 is product-dependent. For demo reporting:
    - IW GRDH is usually treated as 10 m nominal spacing.
    - EW GRDM is often 40 m.
    - SLC/RAW/OCN do not have one simple map pixel size before processing.
    """
    collection_upper = (collection or "").upper()
    product_type_upper = (product_type or "").upper()
    name_upper = (product_name or "").upper()

    if collection_upper == "SENTINEL-2":
        return "10/20/60"

    if collection_upper == "SENTINEL-1":
        combined = f"{product_type_upper} {name_upper}"

        if "GRDH" in combined:
            return "10"

        if "GRDM" in combined:
            return "40"

        if "SLC" in combined:
            return "variable"

        if "RAW" in combined:
            return "not_applicable"

        if "OCN" in combined:
            return "not_applicable"

        return None

    if collection_upper in {"LANDSAT-8", "LANDSAT-9"}:
        return "15/30/100"

    if collection_upper in {"TERRA", "AQUA", "TERRAAQUA"}:
        return "250/500/1000"

    if collection_upper == "COP-DEM":
        return "30/90"

    return None


def normalize_product(
    raw: Dict[str, Any],
    fallback_collection: str,
    config: Dict[str, Any],
) -> Optional[Product]:
    product_id = raw.get("Id")

    if not product_id:
        return None

    name = raw.get("Name") or raw.get("Title") or product_id

    collection = (
        get_nested(raw, "Collection", "Name")
        or raw.get("CollectionName")
        or fallback_collection
    )

    content_date = raw.get("ContentDate") or {}
    acquisition_start = content_date.get("Start")
    acquisition_end = content_date.get("End")

    publication_date = raw.get("PublicationDate")
    product_type = extract_product_type(raw)

    cloud_cover = get_attribute(raw, "cloudCover")

    try:
        cloud_cover = float(cloud_cover) if cloud_cover is not None else None
    except Exception:
        cloud_cover = None

    size_bytes = get_product_size_bytes(raw)
    size_mb = round(size_bytes / 1024 / 1024, 3) if size_bytes else None

    pixel_size_m = infer_pixel_size_m(
        collection=str(collection),
        product_type=product_type,
        product_name=str(name),
        config=config,
    )

    return Product(
        product_id=product_id,
        name=str(name),
        collection=str(collection),
        product_type=product_type,
        pixel_size_m=pixel_size_m,
        acquisition_start=acquisition_start,
        acquisition_end=acquisition_end,
        acquisition_date=parse_datetime_to_date(acquisition_start),
        publication_date=publication_date,
        cloud_cover=cloud_cover,
        size_bytes=size_bytes,
        size_mb=size_mb,
        footprint_geojson=extract_footprint(raw),
        download_url=f"{DOWNLOAD_BASE_URL}/Products({product_id})/$value",
    )


def selected_product_types(collection_config: Dict[str, Any]) -> List[str]:
    product_levels = collection_config.get("product_levels", {}) or {}
    result: List[str] = []

    for level_config in product_levels.values():
        if not isinstance(level_config, dict):
            continue

        if level_config.get("enabled") is not True:
            continue

        product_type = level_config.get("copernicus_product_type")

        if product_type:
            result.append(str(product_type))

    return result


def selected_complementary_collections(config: Dict[str, Any]) -> List[str]:
    complementary = get_nested(config, "collections", "complementary_data", default={}) or {}

    if complementary.get("enabled") is not True:
        return []

    selected: List[str] = []

    for key, item in complementary.items():
        if key in {"enabled", "display_name"}:
            continue

        if not isinstance(item, dict):
            continue

        if item.get("enabled") is not True:
            continue

        collection_name = item.get("copernicus_collection_name") or item.get("display_name")

        if collection_name:
            selected.append(str(collection_name))

    return selected


def collect_catalogue(config: Dict[str, Any], aoi_wkt: str) -> List[Product]:
    project = config.get("project", {}) or {}

    start = project.get("start_date")
    end = project.get("end_date")

    if not start or not end:
        die("config.yaml must contain project.start_date and project.end_date.")

    max_total = int(get_nested(config, "download", "max_total_products", default=12))
    query_top = max(max_total * 5, 100)

    use_product_type_filters = bool(
        get_nested(config, "search", "use_product_type_filters", default=False)
    )
    print_queries = bool(get_nested(config, "search", "print_queries", default=False))

    all_products: List[Product] = []

    sentinel_1 = get_nested(config, "collections", "sentinel_1", default={}) or {}

    if sentinel_1.get("enabled") is True:
        collection_name = sentinel_1.get("copernicus_collection_name", "SENTINEL-1")

        if use_product_type_filters:
            product_types = selected_product_types(sentinel_1)
        else:
            product_types = [None]

        for product_type in product_types:
            raw_products = query_odata_products(
                collection_name=collection_name,
                start_date=start,
                end_date=end,
                aoi_wkt=aoi_wkt,
                product_type=product_type,
                max_cloud_cover=None,
                top=query_top,
                print_queries=print_queries,
            )

            print(f"{collection_name} products found for product_type={product_type}: {len(raw_products)}")

            for raw in raw_products:
                product = normalize_product(
                    raw=raw,
                    fallback_collection=collection_name,
                    config=config,
                )
                if product is not None:
                    all_products.append(product)

    sentinel_2 = get_nested(config, "collections", "sentinel_2", default={}) or {}

    if sentinel_2.get("enabled") is True:
        collection_name = sentinel_2.get("copernicus_collection_name", "SENTINEL-2")

        if use_product_type_filters:
            product_types = selected_product_types(sentinel_2)
        else:
            product_types = [None]

        cloud_cover_enabled = bool(
            get_nested(config, "collections", "sentinel_2", "cloud_cover", "enabled", default=False)
        )

        max_cloud_cover = None

        if cloud_cover_enabled:
            max_cloud_cover = get_nested(
                config,
                "collections",
                "sentinel_2",
                "cloud_cover",
                "max_percent",
                default=None,
            )
            max_cloud_cover = float(max_cloud_cover) if max_cloud_cover is not None else None

        for product_type in product_types:
            raw_products = query_odata_products(
                collection_name=collection_name,
                start_date=start,
                end_date=end,
                aoi_wkt=aoi_wkt,
                product_type=product_type,
                max_cloud_cover=max_cloud_cover,
                top=query_top,
                print_queries=print_queries,
            )

            print(f"{collection_name} products found for product_type={product_type}: {len(raw_products)}")

            for raw in raw_products:
                product = normalize_product(
                    raw=raw,
                    fallback_collection=collection_name,
                    config=config,
                )
                if product is not None:
                    all_products.append(product)

    complementary_collections = selected_complementary_collections(config)

    for collection_name in complementary_collections:
        raw_products = query_odata_products(
            collection_name=collection_name,
            start_date=start,
            end_date=end,
            aoi_wkt=aoi_wkt,
            product_type=None,
            max_cloud_cover=None,
            top=query_top,
            print_queries=print_queries,
        )

        print(f"{collection_name} products found: {len(raw_products)}")

        for raw in raw_products:
            product = normalize_product(
                raw=raw,
                fallback_collection=collection_name,
                config=config,
            )
            if product is not None:
                all_products.append(product)

    deduped: Dict[str, Product] = {}

    for product in all_products:
        deduped[product.product_id] = product

    products = list(deduped.values())
    products.sort(key=lambda p: (p.acquisition_start or "", p.collection, p.name))

    return products


def select_minimal_products(config: Dict[str, Any], products: List[Product]) -> List[Product]:
    max_total = int(get_nested(config, "download", "max_total_products", default=12))
    max_per_collection = int(
        get_nested(config, "download", "max_products_per_collection", default=4)
    )
    max_per_day = int(get_nested(config, "download", "max_products_per_day", default=1))

    selected: List[Product] = []
    by_collection: Dict[str, int] = defaultdict(int)
    by_collection_day: Dict[Tuple[str, str], int] = defaultdict(int)

    for product in products:
        day = product.acquisition_date or "unknown"
        collection = product.collection
        collection_day_key = (collection, day)

        if len(selected) >= max_total:
            break

        if by_collection[collection] >= max_per_collection:
            continue

        if by_collection_day[collection_day_key] >= max_per_day:
            continue

        selected.append(product)
        by_collection[collection] += 1
        by_collection_day[collection_day_key] += 1

    return selected


def safe_filename(name: str) -> str:
    bad_characters = '<>:"/\\|?*'

    for character in bad_characters:
        name = name.replace(character, "_")

    name = name.strip()

    if not name:
        name = "unnamed_product"

    return name[:220]


def download_product(
    product: Product,
    output_root: str | Path,
    token: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Product:
    collection_dir = ensure_dir(Path(output_root) / safe_filename(product.collection))
    day_dir = ensure_dir(collection_dir / (product.acquisition_date or "unknown-date"))

    filename = safe_filename(product.name)

    if not filename.lower().endswith(".zip"):
        filename += ".zip"

    local_path = day_dir / filename
    product.local_path = str(local_path)

    if dry_run:
        product.status = "dry_run"
        return product

    if local_path.exists() and not overwrite:
        product.status = "already_exists"
        product.size_bytes = local_path.stat().st_size
        product.size_mb = round(product.size_bytes / 1024 / 1024, 3)
        return product

    headers = {"Authorization": f"Bearer {token}"}

    try:
        with requests.get(
            product.download_url,
            headers=headers,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=True,
        ) as response:
            if not response.ok:
                product.status = "failed"
                product.error_message = f"HTTP {response.status_code}: {response.text[:500]}"
                return product

            total = int(response.headers.get("Content-Length", 0))
            temporary_path = local_path.with_suffix(local_path.suffix + ".part")

            with open(temporary_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=product.name[:45],
            ) as progress:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    progress.update(len(chunk))

            temporary_path.replace(local_path)

            product.status = "downloaded"
            product.size_bytes = local_path.stat().st_size
            product.size_mb = round(product.size_bytes / 1024 / 1024, 3)

            return product

    except Exception as exc:
        product.status = "failed"
        product.error_message = str(exc)
        return product


def write_catalogue_csv(path: str | Path, products: List[Product]) -> None:
    ensure_dir(Path(path).parent)

    fields = [
        "product_id",
        "product_name",
        "collection",
        "product_type",
        "pixel_size_m",
        "acquisition_start",
        "acquisition_end",
        "acquisition_date",
        "publication_date",
        "cloud_cover",
        "size_mb",
        "status",
        "local_path",
        "download_url",
        "error_message",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for product in products:
            writer.writerow(
                {
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "collection": product.collection,
                    "product_type": product.product_type,
                    "pixel_size_m": product.pixel_size_m,
                    "acquisition_start": product.acquisition_start,
                    "acquisition_end": product.acquisition_end,
                    "acquisition_date": product.acquisition_date,
                    "publication_date": product.publication_date,
                    "cloud_cover": product.cloud_cover,
                    "size_mb": product.size_mb,
                    "status": product.status,
                    "local_path": product.local_path,
                    "download_url": product.download_url,
                    "error_message": product.error_message,
                }
            )


def write_download_log_csv(path: str | Path, products: List[Product]) -> None:
    ensure_dir(Path(path).parent)

    fields = [
        "timestamp_utc",
        "product_id",
        "product_name",
        "collection",
        "product_type",
        "pixel_size_m",
        "acquisition_date",
        "status",
        "local_path",
        "size_mb",
        "error_message",
    ]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for product in products:
            writer.writerow(
                {
                    "timestamp_utc": now,
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "collection": product.collection,
                    "product_type": product.product_type,
                    "pixel_size_m": product.pixel_size_m,
                    "acquisition_date": product.acquisition_date,
                    "status": product.status,
                    "local_path": product.local_path,
                    "size_mb": product.size_mb,
                    "error_message": product.error_message,
                }
            )


def write_statistics_csv(
    path: str | Path,
    all_products: List[Product],
    selected_products: List[Product],
) -> None:
    ensure_dir(Path(path).parent)

    available = defaultdict(int)
    downloaded = defaultdict(int)
    total_size = defaultdict(float)
    product_names = defaultdict(list)
    local_paths = defaultdict(list)
    pixel_sizes = defaultdict(set)

    for product in all_products:
        key = (product.acquisition_date or "unknown", product.collection, product.product_type or "")
        available[key] += 1

        if product.pixel_size_m:
            pixel_sizes[key].add(str(product.pixel_size_m))

    for product in selected_products:
        key = (product.acquisition_date or "unknown", product.collection, product.product_type or "")

        if product.status in {"downloaded", "already_exists", "dry_run"}:
            downloaded[key] += 1

            if product.size_mb:
                total_size[key] += product.size_mb

            product_names[key].append(product.name)

            if product.local_path:
                local_paths[key].append(product.local_path)

            if product.pixel_size_m:
                pixel_sizes[key].add(str(product.pixel_size_m))

    fields = [
        "date",
        "collection",
        "product_type",
        "pixel_size_m",
        "available_products",
        "downloaded_products",
        "total_size_mb",
        "product_names",
        "local_paths",
    ]

    all_keys = sorted(set(available.keys()) | set(downloaded.keys()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for day, collection, product_type in all_keys:
            key = (day, collection, product_type)

            writer.writerow(
                {
                    "date": day,
                    "collection": collection,
                    "product_type": product_type,
                    "pixel_size_m": "; ".join(sorted(pixel_sizes[key])),
                    "available_products": available[key],
                    "downloaded_products": downloaded[key],
                    "total_size_mb": round(total_size[key], 3),
                    "product_names": "; ".join(product_names[key]),
                    "local_paths": "; ".join(local_paths[key]),
                }
            )


def write_footprints_geojson(
    path: str | Path,
    products: List[Product],
    aoi_geom: Optional[BaseGeometry] = None,
) -> None:
    ensure_dir(Path(path).parent)

    features: List[Dict[str, Any]] = []

    if aoi_geom is not None:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "aoi",
                    "name": "AOI",
                },
                "geometry": mapping(aoi_geom),
            }
        )

    for product in products:
        if not product.footprint_geojson:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "product_footprint",
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "collection": product.collection,
                    "product_type": product.product_type,
                    "pixel_size_m": product.pixel_size_m,
                    "acquisition_date": product.acquisition_date,
                    "cloud_cover": product.cloud_cover,
                    "status": product.status,
                    "local_path": product.local_path,
                },
                "geometry": product.footprint_geojson,
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "name": "copernicus_demo_footprints",
        "features": features,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


def print_summary(all_products: List[Product], selected_products: List[Product]) -> None:
    print()
    print("Catalogue summary")
    print("-----------------")

    by_collection_available = defaultdict(int)
    by_collection_selected = defaultdict(int)

    for product in all_products:
        by_collection_available[product.collection] += 1

    for product in selected_products:
        by_collection_selected[product.collection] += 1

    collections = sorted(
        set(by_collection_available.keys()) | set(by_collection_selected.keys())
    )

    if not collections:
        print("No products found.")
        print()
        return

    for collection in collections:
        print(
            f"{collection}: "
            f"available={by_collection_available[collection]}, "
            f"selected={by_collection_selected[collection]}"
        )

    print()


def update_catalogue_status(
    all_products: List[Product],
    selected_products: List[Product],
) -> List[Product]:
    selected_by_id = {product.product_id: product for product in selected_products}
    updated: List[Product] = []

    for product in all_products:
        if product.product_id in selected_by_id:
            updated.append(selected_by_id[product.product_id])
        else:
            updated.append(product)

    return updated


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("COPERNICUS_USERNAME"),
        help="Copernicus Data Space username. Can also use COPERNICUS_USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("COPERNICUS_PASSWORD"),
        help="Copernicus Data Space password. Can also use COPERNICUS_PASSWORD.",
    )

    args = parser.parse_args()

    config = read_yaml(args.config)

    aoi_file = get_nested(config, "project", "aoi_file", default="aoi.geojson")
    aoi_geom = read_aoi_geometry(aoi_file)
    aoi_wkt = geometry_to_odata_wkt(aoi_geom)

    output_dir = get_nested(config, "download", "output_dir", default="data/raw")
    dry_run = bool(get_nested(config, "download", "dry_run", default=False))
    overwrite_existing = bool(
        get_nested(config, "download", "overwrite_existing", default=False)
    )

    catalogue_csv = get_nested(
        config,
        "outputs",
        "catalogue_csv",
        default="outputs/catalogue_results.csv",
    )
    statistics_csv = get_nested(
        config,
        "outputs",
        "statistics_csv",
        default="outputs/download_statistics_by_date.csv",
    )
    download_log_csv = get_nested(
        config,
        "outputs",
        "download_log_csv",
        default="outputs/download_log.csv",
    )
    footprints_geojson = get_nested(
        config,
        "outputs",
        "footprints_geojson",
        default="outputs/footprints.geojson",
    )

    print("Querying Copernicus catalogue...")
    all_products = collect_catalogue(config, aoi_wkt)
    selected_products = select_minimal_products(config, all_products)

    print_summary(all_products, selected_products)

    write_catalogue_csv(catalogue_csv, all_products)
    write_footprints_geojson(footprints_geojson, all_products, aoi_geom=aoi_geom)

    if not selected_products:
        print("No products selected. CSV files were still written.")
        write_statistics_csv(statistics_csv, all_products, selected_products)
        write_download_log_csv(download_log_csv, selected_products)
        return

    if dry_run:
        print("Dry run enabled. No files will be downloaded.")

        for product in selected_products:
            filename = safe_filename(product.name)

            if not filename.lower().endswith(".zip"):
                filename += ".zip"

            product.local_path = str(
                Path(output_dir)
                / safe_filename(product.collection)
                / (product.acquisition_date or "unknown-date")
                / filename
            )
            product.status = "dry_run"

    else:
        if not args.username or not args.password:
            die(
                "Missing credentials. Set COPERNICUS_USERNAME and "
                "COPERNICUS_PASSWORD, or pass --username and --password."
            )

        print("Getting Copernicus access token...")
        token = get_access_token(args.username, args.password)

        print("Downloading selected products as original .zip files...")
        for product in selected_products:
            download_product(
                product=product,
                output_root=output_dir,
                token=token,
                overwrite=overwrite_existing,
                dry_run=False,
            )
            time.sleep(0.2)

    all_products = update_catalogue_status(all_products, selected_products)

    write_catalogue_csv(catalogue_csv, all_products)
    write_download_log_csv(download_log_csv, selected_products)
    write_statistics_csv(statistics_csv, all_products, selected_products)
    write_footprints_geojson(footprints_geojson, all_products, aoi_geom=aoi_geom)

    print("Done.")
    print(f"Catalogue CSV:   {catalogue_csv}")
    print(f"Statistics CSV:  {statistics_csv}")
    print(f"Download log:    {download_log_csv}")
    print(f"Footprints file: {footprints_geojson}")
    print(f"Raw data folder: {output_dir}")
    print()
    print("For SNAP, open the downloaded .zip files directly from data/raw.")


if __name__ == "__main__":
    main()