from pathlib import Path
import json
import math
import sys
import pandas as pd
import numpy as np

MU_EARTH = 398600.4418       # km^3 / s^2
R_EARTH = 6378.137           # km

def read_table(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".json", ".geojson"]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # common wrappers
            for key in ["data", "results", "features"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.json_normalize(data)

    if suffix in [".txt", ".tle", ".3le"]:
        # keep only basic manifest info for TLE-like files
        return None

    return None


def classify_file(path: Path, df):
    name = path.name.lower()

    if df is None:
        if path.suffix.lower() in [".tle", ".3le", ".txt"]:
            return "tle_or_text"
        return "unknown"

    cols = {c.upper() for c in df.columns}

    if "NORAD_CAT_ID" in cols and "EPOCH" in cols and "MEAN_MOTION" in cols:
        if "gp_history" in name or "history" in name:
            return "gp_history"
        return "gp"

    if "NORAD_CAT_ID" in cols and ("OBJECT_TYPE" in cols or "OPS_STATUS_CODE" in cols):
        if "satcat" in name:
            return "satcat"
        return "metadata_or_satcat"

    if any(c in cols for c in ["F10.7_OBSERVED", "F10.7_ADJUSTED", "AP_AVG", "KP_SUM"]):
        return "space_weather"

    if any(c in cols for c in ["MJD", "X", "Y", "UT1-UTC", "LOD"]):
        return "eop"

    return "table"


def normalize_norad(df):
    cols = {c.upper(): c for c in df.columns}
    if "NORAD_CAT_ID" in cols:
        df[cols["NORAD_CAT_ID"]] = pd.to_numeric(df[cols["NORAD_CAT_ID"]], errors="coerce").astype("Int64")
    return df


def add_orbit_geometry(df):
    cols = {c.upper(): c for c in df.columns}

    if "MEAN_MOTION" not in cols:
        return df

    mm_col = cols["MEAN_MOTION"]
    e_col = cols.get("ECCENTRICITY")

    mm = pd.to_numeric(df[mm_col], errors="coerce")  # rev/day
    n_rad_s = mm * 2.0 * math.pi / 86400.0

    a_km = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

    if e_col is not None:
        ecc = pd.to_numeric(df[e_col], errors="coerce")
    else:
        ecc = 0.0

    df["SEMIMAJOR_AXIS_KM_EST"] = a_km
    df["PERIGEE_KM_EST"] = a_km * (1.0 - ecc) - R_EARTH
    df["APOGEE_KM_EST"] = a_km * (1.0 + ecc) - R_EARTH
    df["ALTITUDE_MEAN_KM_EST"] = a_km - R_EARTH

    return df


def summarize_df(path, df, ftype):
    if df is None:
        return {
            "file": str(path),
            "type": ftype,
            "rows": None,
            "cols": None,
            "norad_count": None,
            "epoch_min": None,
            "epoch_max": None,
            "columns": ""
        }

    cols_upper = {c.upper(): c for c in df.columns}

    epoch_min = epoch_max = None
    if "EPOCH" in cols_upper:
        ep = pd.to_datetime(df[cols_upper["EPOCH"]], errors="coerce", utc=True)
        if ep.notna().any():
            epoch_min = str(ep.min())
            epoch_max = str(ep.max())

    norad_count = None
    if "NORAD_CAT_ID" in cols_upper:
        norad_count = pd.to_numeric(df[cols_upper["NORAD_CAT_ID"]], errors="coerce").nunique()

    return {
        "file": str(path),
        "type": ftype,
        "rows": len(df),
        "cols": len(df.columns),
        "norad_count": norad_count,
        "epoch_min": epoch_min,
        "epoch_max": epoch_max,
        "columns": ", ".join(map(str, df.columns[:40]))
    }


def main(root_dir):
    root = Path(root_dir).expanduser().resolve()
    out = root / "ssa_data_audit_outputs"
    out.mkdir(exist_ok=True)

    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]

    manifest_rows = []
    summary_rows = []

    gp_frames = []
    satcat_frames = []
    space_weather_frames = []
    eop_frames = []

    for path in files:
        try:
            df = read_table(path)
            if df is not None:
                df = normalize_norad(df)
            ftype = classify_file(path, df)

            manifest_rows.append({
                "file": str(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size_mb": path.stat().st_size / 1e6,
                "type": ftype
            })

            summary_rows.append(summarize_df(path, df, ftype))

            if df is not None:
                df["_SOURCE_FILE"] = path.name

                if ftype in ["gp", "gp_history"]:
                    df = add_orbit_geometry(df)
                    gp_frames.append(df)

                elif ftype in ["satcat", "metadata_or_satcat"]:
                    satcat_frames.append(df)

                elif ftype == "space_weather":
                    space_weather_frames.append(df)

                elif ftype == "eop":
                    eop_frames.append(df)

        except Exception as e:
            manifest_rows.append({
                "file": str(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size_mb": path.stat().st_size / 1e6,
                "type": f"error: {e}"
            })

    manifest = pd.DataFrame(manifest_rows)
    summaries = pd.DataFrame(summary_rows)

    manifest.to_csv(out / "manifest.csv", index=False)
    summaries.to_csv(out / "file_summaries.csv", index=False)

    print("\n=== FILE TYPE COUNTS ===")
    print(manifest["type"].value_counts(dropna=False))

    print("\n=== FILE SUMMARIES WRITTEN ===")
    print(out / "manifest.csv")
    print(out / "file_summaries.csv")

    if gp_frames:
        gp = pd.concat(gp_frames, ignore_index=True, sort=False)
        gp_cols_upper = {c.upper(): c for c in gp.columns}

        if "EPOCH" in gp_cols_upper:
            gp["_EPOCH_DT"] = pd.to_datetime(gp[gp_cols_upper["EPOCH"]], errors="coerce", utc=True)

        if "OBJECT_NAME" in gp_cols_upper:
            obj_name_col = gp_cols_upper["OBJECT_NAME"]
        else:
            obj_name_col = None

        gp.to_csv(out / "all_gp_like_records.csv", index=False)

        object_summary = gp.groupby(gp_cols_upper["NORAD_CAT_ID"], dropna=True).agg(
            rows=(gp_cols_upper["NORAD_CAT_ID"], "size"),
            epoch_min=("_EPOCH_DT", "min") if "_EPOCH_DT" in gp.columns else (gp_cols_upper["NORAD_CAT_ID"], "size"),
            epoch_max=("_EPOCH_DT", "max") if "_EPOCH_DT" in gp.columns else (gp_cols_upper["NORAD_CAT_ID"], "size"),
            mean_alt_km=("ALTITUDE_MEAN_KM_EST", "median") if "ALTITUDE_MEAN_KM_EST" in gp.columns else (gp_cols_upper["NORAD_CAT_ID"], "size"),
            perigee_km=("PERIGEE_KM_EST", "median") if "PERIGEE_KM_EST" in gp.columns else (gp_cols_upper["NORAD_CAT_ID"], "size"),
            apogee_km=("APOGEE_KM_EST", "median") if "APOGEE_KM_EST" in gp.columns else (gp_cols_upper["NORAD_CAT_ID"], "size"),
        ).reset_index()

        if obj_name_col is not None:
            names = gp.groupby(gp_cols_upper["NORAD_CAT_ID"])[obj_name_col].first().reset_index()
            object_summary = object_summary.merge(names, on=gp_cols_upper["NORAD_CAT_ID"], how="left")

        object_summary.to_csv(out / "object_summary.csv", index=False)

        # Candidate controlled fleet: Iridium
        if obj_name_col is not None:
            iridium = gp[gp[obj_name_col].astype(str).str.contains("IRIDIUM", case=False, na=False)].copy()
            iridium.to_csv(out / "candidate_controlled_iridium_records.csv", index=False)

            latest_iridium = iridium.sort_values("_EPOCH_DT").groupby(gp_cols_upper["NORAD_CAT_ID"]).tail(1)
            latest_iridium.to_csv(out / "candidate_controlled_iridium_latest.csv", index=False)

            print("\n=== IRIDIUM CANDIDATES ===")
            print(f"Unique Iridium-like objects: {latest_iridium[gp_cols_upper['NORAD_CAT_ID']].nunique()}")
            show_cols = [c for c in [gp_cols_upper["NORAD_CAT_ID"], obj_name_col, "ALTITUDE_MEAN_KM_EST", "PERIGEE_KM_EST", "APOGEE_KM_EST"] if c in latest_iridium.columns]
            print(latest_iridium[show_cols].head(20).to_string(index=False))

        # Background LEO shell
        if "PERIGEE_KM_EST" in gp.columns and "APOGEE_KM_EST" in gp.columns:
            latest = gp.sort_values("_EPOCH_DT").groupby(gp_cols_upper["NORAD_CAT_ID"]).tail(1).copy()
            leo_bg = latest[
                (latest["PERIGEE_KM_EST"] >= 500.0) &
                (latest["APOGEE_KM_EST"] <= 1200.0)
            ].copy()

            leo_bg.to_csv(out / "candidate_background_leo_500_1200_latest.csv", index=False)

            print("\n=== BACKGROUND LEO 500–1200 km CANDIDATES ===")
            print(f"Unique objects: {leo_bg[gp_cols_upper['NORAD_CAT_ID']].nunique()}")
            if obj_name_col is not None:
                print(leo_bg[[gp_cols_upper["NORAD_CAT_ID"], obj_name_col, "PERIGEE_KM_EST", "APOGEE_KM_EST"]].head(20).to_string(index=False))

    if satcat_frames:
        satcat = pd.concat(satcat_frames, ignore_index=True, sort=False)
        satcat.to_csv(out / "all_satcat_like_records.csv", index=False)
        print("\n=== SATCAT-LIKE RECORDS ===")
        print(f"Rows: {len(satcat)}")
        print(out / "all_satcat_like_records.csv")

    if space_weather_frames:
        sw = pd.concat(space_weather_frames, ignore_index=True, sort=False)
        sw.to_csv(out / "all_space_weather_records.csv", index=False)
        print("\n=== SPACE WEATHER RECORDS ===")
        print(f"Rows: {len(sw)}")
        print(out / "all_space_weather_records.csv")

    if eop_frames:
        eop = pd.concat(eop_frames, ignore_index=True, sort=False)
        eop.to_csv(out / "all_eop_records.csv", index=False)
        print("\n=== EOP RECORDS ===")
        print(f"Rows: {len(eop)}")
        print(out / "all_eop_records.csv")

    recommendation = """
Recommended next step:

1. Use ISS only for propagation/debugging.
2. Use 10 Iridium satellites as the first controlled centralized fleet.
3. Add 100 nearest LEO background objects from the 500–1200 km shell.
4. Build the Gymnasium environment around SGP4 propagation.
5. Start with an observation-tasking action space before adding maneuver control.

Minimum environment state per controlled satellite:
    r_eci, v_eci, altitude, inclination, eccentricity,
    nearest-object distances, nearest-object relative speeds,
    object-type indicators, time-to-closest-approach proxy.

Recommended first action space:
    Discrete(N_controlled_satellites)
meaning: choose which satellite/object pair receives tracking priority at each step.

Recommended first reward:
    + reward for observing high-risk close approaches
    - penalty for missed close approaches
    - small penalty for excessive retasking
    - optional fuel penalty later if maneuvers are added.
"""
    (out / "recommendation.txt").write_text(recommendation)

    print("\n=== RECOMMENDATION WRITTEN ===")
    print(out / "recommendation.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python ssa_data_audit.py /path/to/data/folder")

    main(sys.argv[1])