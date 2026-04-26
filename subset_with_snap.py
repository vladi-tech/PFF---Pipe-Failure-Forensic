#!/usr/bin/env python3
"""
subset_with_snap.py

Runs SNAP GPT to subset downloaded Copernicus/Sentinel products using
the road-buffer WKT created by prepare_road_buffer.py.

This version avoids the Windows command-line length limit by generating
one temporary SNAP graph XML per product. The WKT is embedded inside the
XML graph instead of being passed through -Pwkt.

Inputs:
- config.yaml
- outputs/catalogue_results.csv
- outputs/road_buffer_5m.wkt
- snap_subset_road_buffer.xml
- downloaded .zip products from data/raw

Outputs:
- SNAP subset products in data/subset_snap
- outputs/snap_subset_log.csv
- outputs/snap_temp_graphs/*.xml

Run:
    python subset_with_snap.py --config config.yaml --limit 1
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


def read_wkt(path: str | Path) -> str:
    p = Path(path)

    if not p.exists():
        die(f"WKT file not found: {path}")

    text = p.read_text(encoding="utf-8").strip()

    if not text:
        die(f"WKT file is empty: {path}")

    return text


def read_catalogue(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)

    if not p.exists():
        die(f"Catalogue CSV not found: {path}")

    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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

    return name[:150]


def is_downloaded_product(row: Dict[str, str]) -> bool:
    status = (row.get("status") or "").strip().lower()
    local_path = (row.get("local_path") or "").strip()

    if status not in {"downloaded", "already_exists"}:
        return False

    if not local_path:
        return False

    if not Path(local_path).exists():
        return False

    return True


def output_path_for_product(
    row: Dict[str, str],
    output_root: str | Path,
) -> Path:
    collection = safe_filename(row.get("collection") or "UNKNOWN_COLLECTION")
    acquisition_date = safe_filename(row.get("acquisition_date") or "unknown-date")
    product_name = safe_filename(row.get("product_name") or row.get("product_id") or "product")

    return Path(output_root) / collection / acquisition_date / f"{product_name}_road_buffer_subset.dim"


def create_product_graph_xml(
    template_graph_file: str | Path,
    generated_graph_file: str | Path,
    input_product: str | Path,
    output_product: str | Path,
    wkt: str,
) -> None:
    """
    Creates a concrete SNAP graph XML for one product.

    This replaces:
    - ${input}
    - ${output}
    - ${wkt}

    The values are XML-escaped because they are written into XML text nodes.
    """
    template_path = Path(template_graph_file)

    if not template_path.exists():
        die(f"SNAP graph template not found: {template_graph_file}")

    template = template_path.read_text(encoding="utf-8")

    input_text = html.escape(str(Path(input_product).resolve()), quote=False)
    output_text = html.escape(str(Path(output_product).resolve()), quote=False)
    wkt_text = html.escape(wkt, quote=False)

    graph_text = (
        template
        .replace("${input}", input_text)
        .replace("${output}", output_text)
        .replace("${wkt}", wkt_text)
    )

    generated_path = Path(generated_graph_file)
    ensure_dir(generated_path.parent)
    generated_path.write_text(graph_text, encoding="utf-8")


def run_snap_graph(
    gpt_executable: str,
    graph_file: str | Path,
) -> subprocess.CompletedProcess[str]:
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


def write_subset_log(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)

    fields = [
        "timestamp_utc",
        "product_id",
        "product_name",
        "collection",
        "product_type",
        "pixel_size_m",
        "acquisition_date",
        "raw_local_path",
        "subset_local_path",
        "generated_graph_file",
        "subset_status",
        "return_code",
        "error_message",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

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
        help="Optional maximum number of products to subset.",
    )
    args = parser.parse_args()

    config = read_yaml(args.config)

    enabled = bool(get_nested(config, "snap_subset", "enabled", default=True))

    if not enabled:
        print("snap_subset.enabled is false. Nothing to do.")
        return

    gpt_executable = get_nested(config, "snap_subset", "gpt_executable", default="gpt")
    template_graph_file = get_nested(
        config,
        "snap_subset",
        "graph_file",
        default="snap_subset_road_buffer.xml",
    )
    catalogue_csv = get_nested(
        config,
        "snap_subset",
        "input_catalogue_csv",
        default="outputs/catalogue_results.csv",
    )
    buffer_wkt_file = get_nested(
        config,
        "snap_subset",
        "input_buffer_wkt",
        default="outputs/road_buffer_5m.wkt",
    )
    output_dir = get_nested(
        config,
        "snap_subset",
        "output_dir",
        default="data/subset_snap",
    )
    subset_log_csv = get_nested(
        config,
        "snap_subset",
        "subset_log_csv",
        default="outputs/snap_subset_log.csv",
    )
    overwrite_existing = bool(
        get_nested(config, "snap_subset", "overwrite_existing", default=False)
    )

    generated_graphs_dir = get_nested(
        config,
        "snap_subset",
        "generated_graphs_dir",
        default="outputs/snap_temp_graphs",
    )

    if not Path(template_graph_file).exists():
        die(f"SNAP graph template file not found: {template_graph_file}")

    wkt = read_wkt(buffer_wkt_file)
    catalogue_rows = read_catalogue(catalogue_csv)

    candidates = [row for row in catalogue_rows if is_downloaded_product(row)]

    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No downloaded products found in catalogue CSV.")
        print("Expected rows with status downloaded or already_exists and a valid local_path.")
        return

    print(f"Products to subset: {len(candidates)}")
    print(f"Using SNAP GPT: {gpt_executable}")
    print(f"Using graph template: {template_graph_file}")
    print(f"Using WKT: {buffer_wkt_file}")
    print(f"Generated graphs folder: {generated_graphs_dir}")
    print()

    log_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(candidates, start=1):
        raw_path = Path(row["local_path"])
        subset_path = output_path_for_product(row, output_dir)
        ensure_dir(subset_path.parent)

        product_id = safe_filename(row.get("product_id") or f"product_{index}")
        generated_graph_file = (
            Path(generated_graphs_dir)
            / f"{index:04d}_{product_id}_subset_graph.xml"
        )

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        base_log = {
            "timestamp_utc": timestamp,
            "product_id": row.get("product_id"),
            "product_name": row.get("product_name"),
            "collection": row.get("collection"),
            "product_type": row.get("product_type"),
            "pixel_size_m": row.get("pixel_size_m"),
            "acquisition_date": row.get("acquisition_date"),
            "raw_local_path": str(raw_path),
            "subset_local_path": str(subset_path),
            "generated_graph_file": str(generated_graph_file),
        }

        if subset_path.exists() and not overwrite_existing:
            print(f"[{index}/{len(candidates)}] Exists, skipping: {subset_path}")
            log_rows.append(
                {
                    **base_log,
                    "subset_status": "already_exists",
                    "return_code": 0,
                    "error_message": "",
                }
            )
            continue

        print(f"[{index}/{len(candidates)}] Creating SNAP graph:")
        print(f"  graph:  {generated_graph_file}")

        create_product_graph_xml(
            template_graph_file=template_graph_file,
            generated_graph_file=generated_graph_file,
            input_product=raw_path,
            output_product=subset_path,
            wkt=wkt,
        )

        print("Running SNAP GPT:")
        print(f"  input:  {raw_path}")
        print(f"  output: {subset_path}")

        result = run_snap_graph(
            gpt_executable=gpt_executable,
            graph_file=generated_graph_file,
        )

        if result.returncode == 0:
            print("  status: success")
            log_rows.append(
                {
                    **base_log,
                    "subset_status": "success",
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
                    "subset_status": "failed",
                    "return_code": result.returncode,
                    "error_message": error_text[:3000],
                }
            )

        print()

    write_subset_log(subset_log_csv, log_rows)

    print("Done.")
    print(f"Subset log CSV: {subset_log_csv}")
    print(f"Subset output folder: {output_dir}")
    print(f"Generated graph folder: {generated_graphs_dir}")


if __name__ == "__main__":
    main()