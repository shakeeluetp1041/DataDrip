# app_cascade.py

import streamlit as st
import pandas as pd
import joblib

# 0) Page config must be the first Streamlit call:
st.set_page_config(page_title="Pump Status Predictor", layout="centered")

# 1) Import your custom preprocessors (ensure preprocessors.py is next to this file)
from preprocessors import (
    LowerCaseStrings,
    StringConverter,
    YearExtractor,
    IQRCapper,
    ConstructionYearTransformer,
    ObjectToNumericConverter,
    AgeCalculator,
    FrequencyEncoder,
    RegionCodeCombiner,
    ColumnDropper,
    AgePipeline,
    GeoContextImputer
)

# 2) Load your saved pipeline (preprocessing + classifier)
@st.cache_resource
def load_full_pipeline(path: str = "full_pipeline.joblib"):
    return joblib.load(path)

pipeline = load_full_pipeline("full_pipeline.joblib")

# 3) (Optional) Try to load a slice of the training set so we can populate dropdowns:
@st.cache_data
def load_feature_metadata(csv_path: str = "../data/Training_Set_Values.csv"):
    return pd.read_csv(csv_path)

feature_df = None
try:
    feature_df = load_feature_metadata()
except FileNotFoundError:
    feature_df = None  # If the CSV isn't found, we'll fall back to hard-coded options.

# 4) Load the geo-lookup table for cascading dropdowns:
@st.cache_data
def load_geo_lookup(path: str = "../data/geo_lookup.csv"):
    return pd.read_csv(path)

geo_lookup = load_geo_lookup()

# 5) Build region → basin lookup (for dynamic basin dropdown)
if feature_df is not None:
    feature_df["region_lower"] = feature_df["region"].astype(str).str.lower()
    REGION_TO_BASINS = (
        feature_df
        .groupby("region_lower")["basin"]
        .apply(lambda col: sorted(col.dropna().unique().tolist()))
        .to_dict()
    )
else:
    # Hard-coded fallback if feature_df wasn’t loaded
    REGION_TO_BASINS = {
        "dodoma": ["lake victoria", "rufiji", "pangani", "wami / ruvu"],
        "iringa": ["rufiji", "pangani"],
        "kagera": ["lake victoria", "ruvuma / southern coast"],
        # …add other regions as needed…
    }

# 6) Build permit/public_meeting options (used below)
if feature_df is not None:
    permit_opts = sorted(feature_df["permit"].dropna().unique().tolist())
    public_meeting_opts = sorted(feature_df["public_meeting"].dropna().unique().tolist())
else:
    permit_opts = [0, 1]
    public_meeting_opts = [0, 1]

# 7) Build all other categorical-options lists once (from feature_df or fallback)
if feature_df is not None:
    wpt_name_opts              = sorted(feature_df["wpt_name"].dropna().unique().tolist())
    scheme_management_opts     = sorted(feature_df["scheme_management"].dropna().unique().tolist())
    funder_opts                = sorted(feature_df["funder"].dropna().unique().tolist())
    installer_opts             = sorted(feature_df["installer"].dropna().unique().tolist())
    scheme_name_opts           = sorted(feature_df["scheme_name"].dropna().unique().tolist())
    management_opts            = sorted(feature_df["management"].dropna().unique().tolist())

    water_quality_opts         = sorted(feature_df["water_quality"].dropna().unique().tolist())
    quantity_opts              = sorted(feature_df["quantity"].dropna().unique().tolist())

    extraction_type_opts       = sorted(feature_df["extraction_type"].dropna().unique().tolist())
    extraction_type_class_opts = sorted(feature_df["extraction_type_class"].dropna().unique().tolist())

    management_group_opts      = sorted(feature_df["management_group"].dropna().unique().tolist())
    payment_opts               = sorted(feature_df["payment"].dropna().unique().tolist())
    payment_type_opts          = sorted(feature_df["payment_type"].dropna().unique().tolist())
    source_class_opts          = sorted(feature_df["source_class"].dropna().unique().tolist())
    waterpoint_type_opts       = sorted(feature_df["waterpoint_type"].dropna().unique().tolist())
else:
    wpt_name_opts              = ["none"]
    scheme_management_opts     = ["vwc", "water authority", "other"]
    funder_opts                = ["government", "private", "other"]
    installer_opts             = ["company_a", "company_b", "other"]
    scheme_name_opts           = ["none", "scheme_a", "scheme_b"]
    management_opts            = ["vwc", "water authority", "other"]

    water_quality_opts         = ["milky", "soft", "salty", "unknown", "other"]
    quantity_opts              = ["enough", "insufficient", "seasonal", "unknown"]

    extraction_type_opts       = ["handpump", "motorpump", "rope pump", "other"]
    extraction_type_class_opts = ["handpump", "motorpump", "submersible", "other"]

    management_group_opts      = ["private", "parastatal", "user-group", "other"]
    payment_opts               = ["pay annually", "pay monthly", "pay per bucket", "pay when scheme fails", "unknown"]
    payment_type_opts          = ["annually", "monthly", "per bucket", "on failure", "other", "unknown"]
    source_class_opts          = ["improved spring", "dam", "surface", "other", "unknown"]
    waterpoint_type_opts       = ["communal standpipe", "communal standpipe multiple", "hand pump", "improved spring", "other"]

# 8) Load centroids for auto-filling lat/lon:
@st.cache_data
def load_centroids(csv_path: str = "../data/Training_Set_Values.csv") -> pd.DataFrame:
    """
    Groups the training CSV by (region, district_code, ward) and computes
    mean latitude/longitude for each group. Returns a DataFrame
    with columns ['region','district_code','ward','latitude_centroid','longitude_centroid'].
    """
    df_train = pd.read_csv(csv_path)
    df_train["region"] = df_train["region"].astype(str).str.lower()
    df_valid_geo = df_train.dropna(subset=["latitude", "longitude"]).copy()
    cent = (
        df_valid_geo
        .groupby(["region", "district_code", "ward"], dropna=False)[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    cent = cent.rename(
        columns={"latitude": "latitude_centroid", "longitude": "longitude_centroid"}
    )
    return cent

centroids = load_centroids()


# -----------------------------------------------------------------------------
# Cascading helper: Region → District → Ward → Subvillage
# -----------------------------------------------------------------------------
def user_input_cascading_geo(geo_lookup: pd.DataFrame) -> dict[str, object]:
    st.sidebar.header("🗺 Geographic Inputs (cascading)")

    # 1) Region (lowercased)
    regions = sorted(geo_lookup["region"].dropna().str.lower().unique().tolist())
    region = st.sidebar.selectbox("Region", regions, key="c_region")

    # 2) District Code (only those in selected region)
    district_opts = (
        geo_lookup[geo_lookup["region"].str.lower() == region]["district_code"]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )
    district_opts = sorted(district_opts)
    district_code = st.sidebar.selectbox("District Code", district_opts, key="c_district")

    # 3) Ward (only those matching region & district)
    ward_opts = (
        geo_lookup[
            (geo_lookup["region"].str.lower() == region)
            & (geo_lookup["district_code"] == district_code)
        ]["ward"]
        .dropna()
        .unique()
        .tolist()
    )
    ward_opts = sorted(ward_opts)
    ward = st.sidebar.selectbox("Ward", ward_opts, key="c_ward")

    # 4) Subvillage (matching region, district, ward)
    subv_opts = (
        geo_lookup[
            (geo_lookup["region"].str.lower() == region)
            & (geo_lookup["district_code"] == district_code)
            & (geo_lookup["ward"] == ward)
        ]["subvillage"]
        .dropna()
        .unique()
        .tolist()
    )
    subv_opts = sorted(subv_opts)
    if not subv_opts:
        subv_opts = ["None"]
    subvillage = st.sidebar.selectbox("Subvillage", subv_opts, key="c_subvillage")
    if subvillage == "None":
        subvillage = pd.NA

    return {
        "region":        region,
        "district_code": district_code,
        "ward":          ward,
        "subvillage":    subvillage,
    }


# -----------------------------------------------------------------------------
# Main feature-input function: renders all widgets and returns one-row DataFrame
# -----------------------------------------------------------------------------
def user_input_features() -> pd.DataFrame:
    # 1) Numeric inputs (population, amount_tsh, gps_height)
    st.sidebar.header("Numeric Inputs")
    population = st.sidebar.number_input(
        "Population Served", min_value=0, max_value=50000, value=1000, step=50, key="population"
    )
    amount_tsh = st.sidebar.number_input(
        "Total Static Head (TSH)", min_value=0, max_value=500, value=50, step=5, key="amount_tsh"
    )
    gps_height = st.sidebar.number_input(
        "GPS Height (m)", min_value=-50, max_value=5000, value=100, step=10, key="gps_height"
    )

    # 2) Permit / Public Meeting
    st.sidebar.header("Permit & Public Meeting")
    permit = st.sidebar.selectbox("Permit? (0 = No, 1 = Yes)", permit_opts, key="permit")
    public_meeting = st.sidebar.selectbox(
        "Public Meeting? (0 = No, 1 = Yes)", public_meeting_opts, key="public_meeting"
    )

    # 3) Construction Year & Date Recorded (for AgeCalculator)
    st.sidebar.header("Construction Year & Date Recorded")
    construction_year = st.sidebar.number_input(
        "Construction Year", min_value=1900, max_value=2025, value=2005, step=1, key="construction_year"
    )
    date_recorded = st.sidebar.date_input(
        "Date Recorded", value=pd.to_datetime("2020-01-01"), key="date_recorded"
    )

    # 4) Frequency-encoded columns
    st.sidebar.header("WPT & Scheme Details")
    wpt_name = st.sidebar.selectbox("Waterpoint Name", wpt_name_opts, key="wpt_name")
    scheme_management = st.sidebar.selectbox(
        "Scheme Management", scheme_management_opts, key="scheme_management"
    )
    funder = st.sidebar.selectbox("Funder", funder_opts, key="funder")
    installer = st.sidebar.selectbox("Installer", installer_opts, key="installer")
    scheme_name = st.sidebar.selectbox("Scheme Name", scheme_name_opts, key="scheme_name")
    management = st.sidebar.selectbox("Management (Frequency-encoded)", management_opts, key="management")

    # 5) Other categorical columns (water_quality, quantity)
    st.sidebar.header("Other Categorical Inputs")
    water_quality = st.sidebar.selectbox("Water Quality", water_quality_opts, key="water_quality")
    quantity = st.sidebar.selectbox("Quantity", quantity_opts, key="quantity")

    # 6) Extraction Details
    st.sidebar.header("Extraction Details")
    extraction_type = st.sidebar.selectbox("Extraction Type", extraction_type_opts, key="extraction_type")
    extraction_type_class = st.sidebar.selectbox(
        "Extraction Type Class", extraction_type_class_opts, key="extraction_type_class"
    )

    # 7) Management / Payment / Source Class / Waterpoint Type
    st.sidebar.header("Management & Payment & Source Class")
    management_group = st.sidebar.selectbox("Management Group", management_group_opts, key="management_group")
    payment = st.sidebar.selectbox("Payment (raw)", payment_opts, key="payment")
    payment_type = st.sidebar.selectbox("Payment Type", payment_type_opts, key="payment_type")
    source_class = st.sidebar.selectbox("Source Class", source_class_opts, key="source_class")
    waterpoint_type = st.sidebar.selectbox("Waterpoint Type", waterpoint_type_opts, key="waterpoint_type")

    # ────────────────────────────────────────────────────────────────────
    # 8) Cascade geographic inputs:
    geo_vals = user_input_cascading_geo(geo_lookup)
    region = geo_vals["region"]              # lowercase string
    district_code = geo_vals["district_code"]
    ward = geo_vals["ward"]
    subvillage = geo_vals["subvillage"]      # pd.NA if “None”

    # 9) Derive region_code / lga from geo_lookup
    try:
        region_code = int(
            geo_lookup[geo_lookup["region"].str.lower() == region]["region_code"]
            .dropna()
            .iloc[0]
        )
    except (IndexError, KeyError):
        region_code = pd.NA

    try:
        lga = (
            geo_lookup[
                (geo_lookup["region"].str.lower() == region)
                & (geo_lookup["district_code"] == district_code)
                & (geo_lookup["ward"] == ward)
                & (geo_lookup["subvillage"] == subvillage)
            ]["lga"]
            .dropna()
            .iloc[0]
        )
    except (IndexError, KeyError):
        lga = pd.NA

    # ────────────────────────────────────────────────────────────────────
    # 10) AUTO-FILL LAT/LON FROM CENTROID
    try:
        cent_row = centroids[
            (centroids["region"] == region)
            & (centroids["district_code"] == district_code)
            & (centroids["ward"] == ward)
        ].iloc[0]
        latitude_centroid = float(cent_row["latitude_centroid"])
        longitude_centroid = float(cent_row["longitude_centroid"])
    except (IndexError, KeyError):
        latitude_centroid = pd.NA
        longitude_centroid = pd.NA

    st.sidebar.header("Auto-filled Latitude & Longitude")
    if pd.notna(latitude_centroid) and pd.notna(longitude_centroid):
        st.sidebar.markdown(
            f"ℹ️ Based on ({region}, {district_code}, {ward}):\n"
            f"- latitude = **{latitude_centroid:.4f}**\n"
            f"- longitude = **{longitude_centroid:.4f}**"
        )
    else:
        st.sidebar.markdown("⚠️ No centroid found; please enter latitude/longitude manually.")

    latitude = st.sidebar.number_input(
        "Latitude (°)",
        min_value=-20.0,
        max_value=20.0,
        value=latitude_centroid if pd.notna(latitude_centroid) else 0.0,
        format="%.5f",
        key="latitude"
    )
    longitude = st.sidebar.number_input(
        "Longitude (°)",
        min_value=20.0,
        max_value=40.0,
        value=longitude_centroid if pd.notna(longitude_centroid) else 0.0,
        format="%.5f",
        key="longitude"
    )

    # ────────────────────────────────────────────────────────────────────
    # 11) DYNAMIC “Basin” dropdown, filtered by region
    try:
        dynamic_basin_list = REGION_TO_BASINS[region]
    except KeyError:
        dynamic_basin_list = sorted(feature_df["basin"].dropna().unique().tolist()) if feature_df is not None else []
    basin = st.sidebar.selectbox("Basin", dynamic_basin_list, key="basin")

    # ────────────────────────────────────────────────────────────────────
    # 12) Assemble everything into a one-row DataFrame and return it
    data = {
        # A) Dropped placeholders (all pd.NA)
        "id":                    pd.NA,
        "num_private":           pd.NA,
        "recorded_by":           pd.NA,
        "extraction_type_group": pd.NA,
        "quality_group":         pd.NA,
        "quantity_group":        pd.NA,
        "source":                pd.NA,
        "waterpoint_type_group": pd.NA,

        # B) Region & codes
        "region":        region,
        "region_code":   region_code,
        "district_code": int(district_code),
        "ward":          ward,
        "subvillage":    subvillage,
        "lga":           lga,

        # C) Geolocation
        "latitude":      latitude,
        "longitude":     longitude,

        # D) Numeric
        "population":    population,
        "amount_tsh":    amount_tsh,
        "gps_height":    gps_height,

        # E) Permit / Public Meeting
        "permit": int(permit),
        "public_meeting": int(public_meeting),

        # F) Construction & Age
        "construction_year": construction_year,
        "date_recorded":      pd.to_datetime(date_recorded).strftime("%Y-%m-%d"),

        # G) Frequency-encoded
        "wpt_name":          wpt_name,
        "scheme_management": scheme_management,
        "funder":            funder,
        "installer":         installer,
        "scheme_name":       scheme_name,
        "management":        management,

        # H) LowerCaseStrings raw
        "basin":          basin,
        "source_type":    source_type_opts[0] if feature_df is None else source_type_opts[0],  # or pick first value
        "water_quality":  water_quality,
        "quantity":       quantity,

        # I) Extraction details
        "extraction_type":        extraction_type,
        "extraction_type_class":  extraction_type_class,

        # J) Management / Payment / Source Class / Waterpoint Type
        "management_group": management_group,
        "payment":          payment,
        "payment_type":     payment_type,
        "source_class":     source_class,
        "waterpoint_type":  waterpoint_type,
    }

    return pd.DataFrame(data, index=[0])


# ─────────────────────────────────────────────────────────────────────────────
# RUNNING THE APP: render sidebar, then display predict button, results, map
# ─────────────────────────────────────────────────────────────────────────────
input_df_small = user_input_features()

if st.sidebar.button("🔍 Predict"):
    # 1) Build the full_input DataFrame from input_df_small
    #    (since user_input_features already returned a full row, we can reuse it:)
    input_df = input_df_small.copy()

    # 2) Predict with your pipeline
    y_pred = pipeline.predict(input_df)[0]
    y_proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None

    # 3) Map numeric label → text
    status_map      = {0: "Non-Functional", 1: "Needs Repair", 2: "Functional"}
    predicted_label = status_map.get(y_pred, str(y_pred))

    st.subheader("📝 Prediction Result")
    st.write(f"**Predicted Pump Status:** :green[{predicted_label}]")

    if y_proba is not None:
        proba_df = pd.DataFrame({
            "Status": ["Non-Functional", "Needs Repair", "Functional"],
            "Probability": y_proba
        }).set_index("Status")
        st.bar_chart(proba_df)

    # 4) Show location on a map
    st.subheader("📍 Location on Map")
    map_df = pd.DataFrame({"lat": [input_df.loc[0, "latitude"]], "lon": [input_df.loc[0, "longitude"]]})
    st.map(map_df, zoom=6)

else:
    st.info("🌟 Adjust parameters and click **Predict** when ready.")
