"""
03_osm_road_network.py — Build inter-district travel distance matrix
=====================================================================
Uses OpenStreetMap road data to compute network distances between
producing and consuming districts in South Sumatra.

SETUP:
    pip install osmnx networkx geopandas shapely

DATA NEEDED:
    Download South Sumatra OSM extract from:
    - Geofabrik: https://download.geofabrik.de/asia/indonesia/sumatra.html
    - Or GEO2day: https://geo2day.com/asia/indonesia/south_sumatra.html
    Place the .pbf file in data/raw/

USAGE:
    python 03_osm_road_network.py

OUTPUT:
    data/processed/district_distance_matrix.csv
    data/processed/district_centroids.geojson
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Average road speed assumptions (km/h) by road type
# Conservative estimates for South Sumatra conditions
SPEED_DEFAULTS = {
    "motorway": 80, "motorway_link": 60,
    "trunk": 60, "trunk_link": 40,
    "primary": 50, "primary_link": 35,
    "secondary": 40, "secondary_link": 30,
    "tertiary": 30, "tertiary_link": 25,
    "residential": 20, "unclassified": 20,
    "living_street": 15, "service": 15,
}

# South Sumatra kabupaten/kota and their approximate centroids
# (used if OSMnx geocoding is slow or unavailable)
DISTRICT_CENTROIDS = {
    "Palembang":           (-2.976, 104.775),
    "Ogan Komering Ilir":  (-3.200, 105.400),
    "Ogan Komering Ulu":   (-4.050, 104.050),
    "Muara Enim":          (-3.700, 103.750),
    "Lahat":               (-3.800, 103.550),
    "Musi Rawas":          (-3.100, 102.900),
    "Musi Banyuasin":      (-2.700, 104.200),
    "Banyuasin":           (-2.500, 104.800),
    "OKU Selatan":         (-4.400, 104.100),
    "OKU Timur":           (-3.750, 104.500),
    "Ogan Ilir":           (-3.250, 104.650),
    "Empat Lawang":        (-3.950, 103.250),
    "Penukal Abab Lematang Ilir": (-3.350, 103.800),
    "Musi Rawas Utara":    (-2.800, 102.700),
    "Lubuklinggau":        (-3.300, 102.867),
    "Prabumulih":          (-3.433, 104.233),
    "Pagar Alam":          (-4.017, 103.267),
}

# Classify districts as producing vs consuming
# (horticultural production centers are highland kabupaten)
PRODUCING_DISTRICTS = [
    "OKU Selatan", "Lahat", "Pagar Alam", "Empat Lawang",
    "Ogan Komering Ulu", "OKU Timur"
]
CONSUMING_DISTRICTS = [
    "Palembang", "Lubuklinggau", "Prabumulih",
    "Ogan Komering Ilir", "Musi Banyuasin", "Banyuasin",
    "Ogan Ilir", "Muara Enim", "Musi Rawas",
    "Penukal Abab Lematang Ilir", "Musi Rawas Utara"
]


def method_a_osmnx():
    """
    Use OSMnx to download and analyze the road network.
    Requires internet access. This is the cleanest approach.
    """
    try:
        import osmnx as ox
        import networkx as nx
    except ImportError:
        log.error("OSMnx not installed. Run: pip install osmnx")
        return None

    log.info("Method A: Downloading road network via OSMnx...")
    log.info("(This may take 5-10 minutes for the full province)")

    try:
        # Download driveable road network for South Sumatra
        G = ox.graph_from_place(
            "Sumatera Selatan, Indonesia",
            network_type="drive",
            simplify=True
        )
        log.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Add travel time to edges
        G = ox.add_edge_speeds(G, fallback=30)  # 30 km/h default
        G = ox.add_edge_travel_times(G)

        # Compute shortest path travel times between district centroids
        results = []
        for orig_name, (orig_lat, orig_lon) in DISTRICT_CENTROIDS.items():
            orig_node = ox.nearest_nodes(G, orig_lon, orig_lat)
            for dest_name, (dest_lat, dest_lon) in DISTRICT_CENTROIDS.items():
                if orig_name == dest_name:
                    continue
                dest_node = ox.nearest_nodes(G, dest_lon, dest_lat)
                try:
                    # Travel time in seconds
                    tt = nx.shortest_path_length(G, orig_node, dest_node, weight="travel_time")
                    # Distance in meters
                    dist = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
                    results.append({
                        "origin": orig_name,
                        "destination": dest_name,
                        "distance_km": round(dist / 1000, 1),
                        "travel_time_hrs": round(tt / 3600, 2),
                    })
                except nx.NetworkXNoPath:
                    results.append({
                        "origin": orig_name,
                        "destination": dest_name,
                        "distance_km": np.nan,
                        "travel_time_hrs": np.nan,
                    })
                    log.warning(f"  No path: {orig_name} -> {dest_name}")

        return pd.DataFrame(results)

    except Exception as e:
        log.error(f"OSMnx method failed: {e}")
        return None


def method_b_euclidean():
    """
    Fallback: Compute straight-line (Euclidean) distances between
    district centroids, then apply a road sinuosity factor.
    Less accurate but always works without internet.
    """
    log.info("Method B: Computing Euclidean distances with sinuosity correction...")

    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    # Road sinuosity factor: actual road distance is typically 1.3-1.5x straight-line
    # in hilly South Sumatra terrain (highland producing areas)
    SINUOSITY = 1.4
    AVG_SPEED_KMH = 35  # conservative for mixed road conditions

    results = []
    for orig_name, (orig_lat, orig_lon) in DISTRICT_CENTROIDS.items():
        for dest_name, (dest_lat, dest_lon) in DISTRICT_CENTROIDS.items():
            if orig_name == dest_name:
                continue
            straight_km = haversine_km(orig_lat, orig_lon, dest_lat, dest_lon)
            road_km = round(straight_km * SINUOSITY, 1)
            travel_hrs = round(road_km / AVG_SPEED_KMH, 2)
            results.append({
                "origin": orig_name,
                "destination": dest_name,
                "distance_km": road_km,
                "travel_time_hrs": travel_hrs,
            })

    return pd.DataFrame(results)


def build_producer_consumer_matrix(distance_df):
    """
    From the full distance matrix, extract the minimum distance
    from each consuming district to any producing district.
    This is the key feature for the model: how far is the nearest
    supply source from each market?
    """
    log.info("Building producer-consumer distance features...")

    # Filter to producing -> consuming pairs only
    pc = distance_df[
        (distance_df["origin"].isin(PRODUCING_DISTRICTS)) &
        (distance_df["destination"].isin(CONSUMING_DISTRICTS))
    ].copy()

    # For each consuming district, find the nearest producing district
    nearest = (pc.groupby("destination")
        .agg(
            nearest_producer=("origin", lambda x: x.iloc[pc.loc[x.index, "distance_km"].idxmin() - x.index[0]] if len(x) > 0 else "unknown"),
            min_distance_km=("distance_km", "min"),
            min_travel_hrs=("travel_time_hrs", "min"),
            avg_distance_km=("distance_km", "mean"),
        )
        .reset_index()
        .rename(columns={"destination": "district"})
    )

    # Also add producing districts with distance 0 (they produce locally)
    for d in PRODUCING_DISTRICTS:
        if d not in nearest["district"].values:
            nearest = pd.concat([nearest, pd.DataFrame([{
                "district": d,
                "nearest_producer": d,
                "min_distance_km": 0,
                "min_travel_hrs": 0,
                "avg_distance_km": 0,
            }])], ignore_index=True)

    return nearest


def main():
    log.info("=" * 60)
    log.info("OSM Road Network Processing — South Sumatra")
    log.info("=" * 60)

    # Try OSMnx first, fall back to Euclidean
    distance_df = method_a_osmnx()
    if distance_df is None:
        distance_df = method_b_euclidean()

    # Save full matrix
    full_path = PROC_DIR / "district_distance_matrix.csv"
    distance_df.to_csv(full_path, index=False)
    log.info(f"Full distance matrix saved: {full_path} ({len(distance_df)} pairs)")

    # Build the producer-consumer feature table
    pc_features = build_producer_consumer_matrix(distance_df)
    pc_path = PROC_DIR / "district_supply_distance.csv"
    pc_features.to_csv(pc_path, index=False)
    log.info(f"Producer-consumer features saved: {pc_path}")
    log.info(f"\n{pc_features.to_string()}")

    # Save centroids as GeoJSON for mapping
    centroids_data = []
    for name, (lat, lon) in DISTRICT_CENTROIDS.items():
        centroids_data.append({
            "district": name,
            "lat": lat,
            "lon": lon,
            "is_producer": name in PRODUCING_DISTRICTS,
        })
    centroids_df = pd.DataFrame(centroids_data)
    centroids_gdf = gpd.GeoDataFrame(
        centroids_df,
        geometry=gpd.points_from_xy(centroids_df.lon, centroids_df.lat),
        crs="EPSG:4326"
    )
    geo_path = PROC_DIR / "district_centroids.geojson"
    centroids_gdf.to_file(geo_path, driver="GeoJSON")
    log.info(f"Centroids saved: {geo_path}")

    log.info("\nDone! Distance features are ready for the modeling pipeline.")


if __name__ == "__main__":
    main()
