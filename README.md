# PFF---Pipe-Failure-Forensic
Cassini Hackaton PFF Repo

# Copernicus Demo Pipeline

This project downloads Copernicus/Sentinel data for an AOI, prepares a road-buffer AOI, subsets products in SNAP, and prepares the basis for SNAP-based Sentinel-1 and Sentinel-2 processing.

The current workflow is designed for a demo. It prioritizes clarity, reproducibility, and SNAP compatibility over production-scale optimization.

## What the pipeline does

The pipeline currently supports:

1. Downloading original Copernicus/Sentinel `.SAFE.zip` products.
2. Writing catalogue, download, and date-statistics CSV files.
3. Clipping a road network to the AOI.
4. Creating a 5 m road buffer around the clipped roads.
5. Subsetting downloaded Sentinel products with SNAP GPT.
6. Preparing Sentinel-1 GRD processing with orbit correction, calibration, terrain correction, and VV backscatter output.
7. Preparing Sentinel-2 index processing for NDWI and NDRE.

The important design choice is that raw Sentinel products are kept as original `.zip` files so they remain suitable for SNAP.

## Folder structure

```text
copernicus_demo/
  aoi.geojson
  roads.geojson
  config.yaml

  download_copernicus_demo.py
  prepare_road_buffer.py
  subset_with_snap.py
  process_sentinel1_snap.py

  snap_subset_road_buffer.xml
  s1_orbit_calibrate_tc_graph.xml
  s2_indices_graph.xml

  requirements.txt
  README.md

  data/
    raw/
    subset_snap/
    processed_snap/
    masked_geotiff/

  outputs/
    catalogue_results.csv
    download_statistics_by_date.csv
    download_log.csv
    footprints.geojson

    clipped_roads.geojson
    road_buffer_5m.geojson
    road_buffer_5m.gpkg
    road_buffer_5m.wkt

    snap_subset_log.csv
    snap_processing_log.csv
    snap_temp_graphs/
    snap_processing_graphs/
