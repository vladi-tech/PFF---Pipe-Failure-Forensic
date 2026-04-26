#!/usr/bin/env python3
"""
process_sentinel1_snap.py

Automates Sentinel-1 GRD processing in SNAP GPT.

Processing chain:
1. Read original Sentinel-1 GRD .SAFE.zip
2. Apply Orbit File
3. Thermal Noise Removal
4. Remove GRD Border Noise
5. Calibration to Sigma0_VV
6. Terrain Correction
7. Convert Sigma0_VV to dB
8. Subset to road-buffer WKT bounding region
9. Write SNAP BEAM-DIMAP .dim product

Important:
- This creates SNAP-native .dim/.data outputs.
- The Subset operator crops to the geometry region but does not necessarily mask
  every pixel outside the road buffer. Final road-buffer-only GeoTIFF masking
  will be handled in a later file.

Run:
    python process_sentinel1_snap.py --config config.yaml

Test one product:
    python process_sentinel1_snap.py --config config.yaml --limit 1
"""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


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


def read_text_file(path: str | Path, description: str) -> str:
    p = Path(path)

    if not p.exists():
        die(f"{description} not found: {path}")

    text = p.read_text(encoding="utf-8").strip()

    if not text:
        die(f"{description} is empty: {path}")

    return text


def read_catalogue(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)

    if not p.exists():
        die(f"Catalogue CSV not found: {path}")

    with open(p, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        die(f"Catalogue CSV has no rows: {path}")

    return rows


def safe_filename(name: str) -> str:
    bad_characters = '<>:"/\\|?*'

    for character in bad_characters:
        name = name.replace(character, "_")

    name = name.strip()

    if not name:
        name = "unnamed_product"

    return name[:160]


def is_valid_s1_input(row: Dict[str, str], accepted_product_types: List[str]) -> bool:
    collection = (row.get("collection") or "").strip().upper()
    product_type = (row.get("product_type") or "").strip()
    status = (row.get("status") or "").strip().lower()
    local_path = (row.get("local_path") or "").strip()

    if collection != "SENTINEL-1":
        return False

    if product_type not in accepted_product_types:
        return False

    if status not in {"downloaded", "already_exists"}:
        return False

    if not local_path:
        return False

    if not Path(local_path).exists():
        return False

    return True


def output_path_for_product(row: Dict[str, str], output_root: str | Path) -> Path:
    acquisition_date = safe_filename(row.get("acquisition_date") or "unknown-date")
    product_name = safe_filename(row.get("product_name") or row.get("product_id") or "s1_product")

    return Path(output_root) / acquisition_date / f"{product_name}_Orb_Cal_TC_VV_db.dim"


def create_graph_xml(
    template_graph_file: str | Path,
    generated_graph_file: str | Path,
    input_product: str | Path,
    output_product: str | Path,
    wkt: str,
    dem_name: str,
    pixel_spacing_m: float,
    map_projection: str,
    format_name: str,
) -> None:
    template_path = Path(template_graph_file)

    if not template_path.exists():
        die(f"Sentinel-1 graph template not found: {template_graph_file}")

    template = template_path.read_text(encoding="utf-8")

    replacements = {
        "${input}": html.escape(str(Path(input_product).resolve()), quote=False),
        "${output}": html.escape(str(Path(output_product).resolve()), quote=False),
        "${wkt}": html.escape(wkt, quote=False),
        "${demName}": html.escape(dem_name, quote=False),
        "${pixelSpacingInMeter}": str(pixel_spacing_m),
        "${mapProjection}": html.escape(map_projection, quote=False),
        "${formatName}": html.escape(format_name, quote=False),
    }

    graph_text = template

    for token, value in replacements.items():
        graph_text = graph_text.replace(token, value)

    generated_path = Path(generated_graph_file)
    ensure_dir(generated_path.parent)
    generated_path.write_text(graph_text, encoding="utf-8")


def run_gpt(gpt_executable: str, graph_file: str | Path) -> subprocess.CompletedProcess[str]:
    command = [
        gpt_executable,
        str(graph_file),
    ]

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def write_processing_log(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)

    fields = [
        "timestamp_utc",
        "processor",
        "product_id",
        "product_name",
        "collection",
        "product_type",
        "pixel_size_m",
        "acquisition_date",
        "raw_local_path",
        "processed_local_path",
        "generated_graph_file",
        "processing_status",
        "return_code",
        "error_message",
    ]

    existing_rows: List[Dict[str, Any]] = []

    p = Path(path)
    if p.exists():
        with open(p, "r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of Sentinel-1 products to process.",
    )

    args = parser.parse_args()
    config = read_yaml(args.config)

    enabled = bool(get_nested(config, "snap_processing", "sentinel_1", "enabled", default=True))

    if not enabled:
        print("snap_processing.sentinel_1.enabled is false. Nothing to do.")
        return

    gpt_executable = get_nested(config, "snap_processing", "gpt_executable", default="gpt")
    catalogue_csv = get_nested(
        config,
        "snap_processing",
        "input_catalogue_csv",
        default="outputs/catalogue_results.csv",
    )
    buffer_wkt_file = get_nested(
        config,
        "snap_processing",
        "input_buffer_wkt",
        default="outputs/road_buffer_5m.wkt",
    )
    generated_graphs_dir = get_nested(
        config,
        "snap_processing",
        "generated_graphs_dir",
        default="outputs/snap_processing_graphs",
    )
    processing_log_csv = get_nested(
        config,
        "snap_processing",
        "processing_log_csv",
        default="outputs/snap_processing_log.csv",
    )

    graph_template = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "graph_template",
        default="s1_orbit_calibrate_tc_graph.xml",
    )
    output_dir = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "output_dir",
        default="data/processed_snap/SENTINEL-1",
    )
    accepted_product_types = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "accepted_product_types",
        default=["IW_GRDH_1S"],
    )
    output_format = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "output_format",
        default="BEAM-DIMAP",
    )
    dem_name = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "dem_name",
        default="SRTM 1Sec HGT",
    )
    map_projection = get_nested(
        config,
        "snap_processing",
        "sentinel_1",
        "map_projection",
        default="AUTO:42001",
    )
    pixel_spacing_m = float(
        get_nested(
            config,
            "snap_processing",
            "sentinel_1",
            "pixel_spacing_m",
            default=10,
        )
    )
    overwrite_existing = bool(
        get_nested(config, "snap_processing", "overwrite_existing", default=False)
    )

    if not isinstance(accepted_product_types, list):
        die("snap_processing.sentinel_1.accepted_product_types must be a list.")

    accepted_product_types = [str(x) for x in accepted_product_types]

    if not Path(graph_template).exists():
        die(f"Sentinel-1 graph template not found: {graph_template}")

    wkt = read_text_file(buffer_wkt_file, "Road-buffer WKT")
    catalogue_rows = read_catalogue(catalogue_csv)

    candidates = [
        row for row in catalogue_rows
        if is_valid_s1_input(row, accepted_product_types)
    ]

    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No downloaded Sentinel-1 GRD products found for processing.")
        print("Expected catalogue rows with:")
        print("  collection = SENTINEL-1")
        print(f"  product_type in {accepted_product_types}")
        print("  status = downloaded or already_exists")
        print("  local_path pointing to an existing .zip")
        return

    print(f"Sentinel-1 products to process: {len(candidates)}")
    print(f"Using SNAP GPT: {gpt_executable}")
    print(f"Using graph template: {graph_template}")
    print(f"Using WKT: {buffer_wkt_file}")
    print(f"Output folder: {output_dir}")
    print()

    log_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(candidates, start=1):
        raw_path = Path(row["local_path"])
        processed_path = output_path_for_product(row, output_dir)
        ensure_dir(processed_path.parent)

        product_id = safe_filename(row.get("product_id") or f"s1_product_{index}")
        generated_graph_file = (
            Path(generated_graphs_dir)
            / "SENTINEL-1"
            / f"{index:04d}_{product_id}_s1_processing_graph.xml"
        )

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        base_log = {
            "timestamp_utc": timestamp,
            "processor": "sentinel_1_orbit_calibration_terrain_correction",
            "product_id": row.get("product_id"),
            "product_name": row.get("product_name"),
            "collection": row.get("collection"),
            "product_type": row.get("product_type"),
            "pixel_size_m": row.get("pixel_size_m"),
            "acquisition_date": row.get("acquisition_date"),
            "raw_local_path": str(raw_path),
            "processed_local_path": str(processed_path),
            "generated_graph_file": str(generated_graph_file),
        }

        if processed_path.exists() and not overwrite_existing:
            print(f"[{index}/{len(candidates)}] Exists, skipping: {processed_path}")
            log_rows.append(
                {
                    **base_log,
                    "processing_status": "already_exists",
                    "return_code": 0,
                    "error_message": "",
                }
            )
            continue

        print(f"[{index}/{len(candidates)}] Creating Sentinel-1 SNAP graph:")
        print(f"  graph:  {generated_graph_file}")
        print(f"  input:  {raw_path}")
        print(f"  output: {processed_path}")

        create_graph_xml(
            template_graph_file=graph_template,
            generated_graph_file=generated_graph_file,
            input_product=raw_path,
            output_product=processed_path,
            wkt=wkt,
            dem_name=dem_name,
            pixel_spacing_m=pixel_spacing_m,
            map_projection=map_projection,
            format_name=output_format,
        )

        result = run_gpt(
            gpt_executable=gpt_executable,
            graph_file=generated_graph_file,
        )

        if result.returncode == 0:
            print("  status: success")
            log_rows.append(
                {
                    **base_log,
                    "processing_status": "success",
                    "return_code": result.returncode,
                    "error_message": "",
                }
            )
        else:
            error_text = (result.stderr or result.stdout or "").strip()
            print("  status: failed")
            print(f"  return code: {result.returncode}")

            if error_text:
                print(f"  error: {error_text[:1500]}")

            log_rows.append(
                {
                    **base_log,
                    "processing_status": "failed",
                    "return_code": result.returncode,
                    "error_message": error_text[:3000],
                }
            )

        print()

    write_processing_log(processing_log_csv, log_rows)

    print("Done.")
    print(f"Processing log CSV: {processing_log_csv}")
    print(f"Processed Sentinel-1 folder: {output_dir}")


if __name__ == "__main__":
    main()