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
    feature_df = None  # If the CSV isn't found, we'll fall back to hard‐coded options.


# 4) Load the geo‐lookup table for cascading dropdowns:
@st.cache_data
def load_geo_lookup(path: str = "../data/geo_lookup.csv"):
    return pd.read_csv(path)

geo_lookup = load_geo_lookup()

@st.cache_data
def load_centroids(csv_path: str = "../data/Training_Set_Values.csv") -> pd.DataFrame:
    """
    Read the full training CSV, group by (region, district_code, ward),
    and compute mean latitude/longitude for each group.  Returns a DataFrame
    with columns ['region','district_code','ward','latitude_centroid','longitude_centroid'].
    """
    df_train = pd.read_csv(csv_path)
    # Ensure region string is lowercase to match cascading (if you forced lowercasing)
    df_train["region"] = df_train["region"].astype(str).str.lower()
    # Drop any rows where either latitude or longitude is missing/zero:
    df_valid_geo = df_train.dropna(subset=["latitude", "longitude"]).copy()
    # Now group by region, district_code, ward
    cent = (
        df_valid_geo
        .groupby(["region", "district_code", "ward"], dropna=False)[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    # Rename columns to avoid clashing with user_input_features’ “latitude” / “longitude”
    cent = cent.rename(
        columns={"latitude": "latitude_centroid", "longitude": "longitude_centroid"}
    )
    return cent

# Call it once so it’s cached
centroids = load_centroids()


# -----------------------------------------------------------------------------
# Then adjust your cascading helper so that when the user picks (region, district_code,
# ward), you can look up a centroid later.  Nothing changes inside user_input_cascading_geo itself.
# (It still just returns region/district_code/ward/subvillage.)
# -----------------------------------------------------------------------------

def user_input_cascading_geo(geo_lookup: pd.DataFrame) -> dict[str, object]:
    """
    Renders four cascading dropdowns: Region → District Code → Ward → Subvillage.
    If no subvillages exist, shows “None” and returns subvillage=pd.NA.
    """
    st.sidebar.header("🗺 Geographic Inputs (cascading)")

    # 1) Region
    regions = sorted(geo_lookup["region"].dropna().str.lower().unique().tolist())
    region = st.sidebar.selectbox("Region", regions, key="c_region")

    # 2) District Code (only those in the selected region)
    district_opts = (
        geo_lookup[geo_lookup["region"].str.lower() == region]["district_code"]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )
    district_opts = sorted(district_opts)
    district_code = st.sidebar.selectbox("District Code", district_opts, key="c_district")

    # 3) Ward (only where region & district_code match)
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

    # 4) Subvillage (only where region, district_code, ward match)
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
# Finally, inside user_input_features(), replace the old latitude/longitude widget
# section with this “auto‐fill from centroid” logic.  In other words:
#   – Remove the two st.sidebar.number_input calls for latitude/longitude
#   – Insert the block below immediately after you read region/district/ward/subvillage
# -----------------------------------------------------------------------------

def user_input_features() -> pd.DataFrame:
    """
    Renders all sidebar widgets for the raw columns your pipeline expects.
    Returns a one‐row DataFrame with exactly the keys the pipeline needs.
    """
    # ────────────────────────────────────────────────────────────────────
    # 1) Build dropdown options (from feature_df, or fallback)
    # ────────────────────────────────────────────────────────────────────
    if feature_df is not None:
        # Numeric codes:
        region_code_opts           = sorted(feature_df["region_code"].dropna().unique().astype(int).tolist())
        # (district_code, ward, subvillage come via geo_lookup → cascading)

        # Categorical/raw‐string columns:
        wpt_name_opts              = feature_df["wpt_name"].dropna().unique().tolist()
        scheme_management_opts     = feature_df["scheme_management"].dropna().unique().tolist()
        funder_opts                = feature_df["funder"].dropna().unique().tolist()
        installer_opts             = feature_df["installer"].dropna().unique().tolist()
        scheme_name_opts           = feature_df["scheme_name"].dropna().unique().tolist()
        management_opts            = feature_df["management"].dropna().unique().tolist()

        basin_opts                 = feature_df["basin"].dropna().unique().tolist()
        source_type_opts           = feature_df["source_type"].dropna().unique().tolist()
        water_quality_opts         = feature_df["water_quality"].dropna().unique().tolist()
        quantity_opts              = feature_df["quantity"].dropna().unique().tolist()

        extraction_type_opts       = feature_df["extraction_type"].dropna().unique().tolist()
        extraction_type_class_opts = feature_df["extraction_type_class"].dropna().unique().tolist()

        management_group_opts      = feature_df["management_group"].dropna().unique().tolist()
        payment_opts               = feature_df["payment"].dropna().unique().tolist()
        payment_type_opts          = feature_df["payment_type"].dropna().unique().tolist()
        source_class_opts          = feature_df["source_class"].dropna().unique().tolist()
        waterpoint_type_opts       = feature_df["waterpoint_type"].dropna().unique().tolist()

        permit_opts                = feature_df["permit"].dropna().unique().tolist()
        public_meeting_opts        = feature_df["public_meeting"].dropna().unique().tolist()
    else:
        # FALLBACK: Hard‐code reasonable lists
        region_code_opts           = list(range(1, 31))
        wpt_name_opts              = ["none"]
        scheme_management_opts     = ["vwc", "water authority", "other"]
        funder_opts                = ["government", "private", "other"]
        installer_opts             = ["company_a", "company_b", "other"]
        scheme_name_opts           = ["none", "scheme_a", "scheme_b"]
        management_opts            = ["vwc", "water authority", "other"]

        basin_opts                 = ["lake victoria", "rufiji", "pangani", "wami / ruvu", "other"]
        source_type_opts           = ["spring", "shallow well", "rainwater harvesting", "other"]
        water_quality_opts         = ["milky", "soft", "salty", "unknown", "other"]
        quantity_opts              = ["enough", "insufficient", "seasonal", "unknown"]

        extraction_type_opts       = ["handpump", "motorpump", "rope pump", "other"]
        extraction_type_class_opts = ["handpump", "motorpump", "submersible", "other"]

        management_group_opts      = ["private", "parastatal", "user-group", "other"]
        payment_opts               = ["pay annually", "pay monthly", "pay per bucket", "pay when scheme fails", "unknown"]
        payment_type_opts          = ["annually", "monthly", "per bucket", "on failure", "other", "unknown"]
        source_class_opts          = ["improved spring", "dam", "surface", "other", "unknown"]
        waterpoint_type_opts       = ["communal standpipe", "communal standpipe multiple", "hand pump", "improved spring", "other"]

        permit_opts                = [0, 1]      # 0 = no permit, 1 = yes permit
        public_meeting_opts        = [0, 1]      # 0 = no meeting, 1 = yes meeting

    # ────────────────────────────────────────────────────────────────────
    # 2) Build sidebar widgets for all non‐geographic columns
    # ────────────────────────────────────────────────────────────────────

    # A) Placeholders for columns we drop (no widget; set to pd.NA later)
    st.sidebar.header("Columns Dropped / Placeholders")
    st.sidebar.write("These will be set to pd.NA: id, num_private, recorded_by,")
    st.sidebar.write("extraction_type_group, quality_group, quantity_group, source, waterpoint_type_group")

    # B) Nothing here for region/district/ward/subvillage—handled in step 3 below.

    # C) POPULATION, AMOUNT_TSH, GPS_HEIGHT
    st.sidebar.header("Numeric Inputs")
    population = st.sidebar.number_input(
        "Population Served",
        min_value=0,
        max_value=50000,
        value=1000,
        step=50,
        key="population"
    )
    amount_tsh = st.sidebar.number_input(
        "Total Static Head (TSH)",
        min_value=0,
        max_value=500,
        value=50,
        step=5,
        key="amount_tsh"
    )
    gps_height = st.sidebar.number_input(
        "GPS Height (m)",
        min_value=-50,
        max_value=5000,
        value=100,
        step=10,
        key="gps_height"
    )

    # D) Permit / Public Meeting (0 or 1)
    st.sidebar.header("Permit & Public Meeting")
    permit = st.sidebar.selectbox("Permit? (0 = No, 1 = Yes)", permit_opts, key="permit")
    public_meeting = st.sidebar.selectbox("Public Meeting? (0 = No, 1 = Yes)", public_meeting_opts, key="public_meeting")

    # E) Construction Year & Date Recorded (for AgeCalculator)
    st.sidebar.header("Construction Year & Date Recorded")
    construction_year = st.sidebar.number_input(
        "Construction Year",
        min_value=1900,
        max_value=2025,
        value=2005,
        step=1,
        key="construction_year"
    )
    date_recorded = st.sidebar.date_input(
        "Date Recorded",
        value=pd.to_datetime("2020-01-01"),
        key="date_recorded"
    )

    # F) Frequency‐encoded columns
    st.sidebar.header("WPT & Scheme Details")
    wpt_name          = st.sidebar.selectbox("Waterpoint Name", wpt_name_opts, key="wpt_name")
    scheme_management = st.sidebar.selectbox("Scheme Management", scheme_management_opts, key="scheme_management")
    funder            = st.sidebar.selectbox("Funder", funder_opts, key="funder")
    installer         = st.sidebar.selectbox("Installer", installer_opts, key="installer")
    scheme_name       = st.sidebar.selectbox("Scheme Name", scheme_name_opts, key="scheme_name")
    management        = st.sidebar.selectbox("Management (Freq‐encoded)", management_opts, key="management")

    # G) LowerCaseStrings raw columns
    st.sidebar.header("Other Categorical Inputs")

    # (A) After you call user_input_cascading_geo() and have `region`:
    region_selected = region  # already lowercased by cascading‐helper

    # (B) Look up only those basins that occurred in training for this region:
    try:
        dynamic_basin_list = REGION_TO_BASINS[region_selected]
    except KeyError:
        # If this region wasn’t in REGION_TO_BASINS, fall back to all basins:
        dynamic_basin_list = basin_opts.copy()  # basin_opts was your full‐fallback

    # (C) Render the dropdown using that filtered list:
    basin = st.sidebar.selectbox("Basin", dynamic_basin_list, key="basin")

    source_type   = st.sidebar.selectbox("Source Type", source_type_opts, key="source_type")
    water_quality = st.sidebar.selectbox("Water Quality", water_quality_opts, key="water_quality")
    quantity      = st.sidebar.selectbox("Quantity", quantity_opts, key="quantity")

    # H) Extraction details
    st.sidebar.header("Extraction Details")
    extraction_type       = st.sidebar.selectbox("Extraction Type", extraction_type_opts, key="extraction_type")
    extraction_type_class = st.sidebar.selectbox("Extraction Type Class", extraction_type_class_opts, key="extraction_type_class")

    # I) Management Group / Payment / Payment Type / Source Class / Waterpoint Type
    st.sidebar.header("Management & Payment & Source Class")
    management_group = st.sidebar.selectbox("Management Group", management_group_opts, key="management_group")
    payment          = st.sidebar.selectbox("Payment (raw)", payment_opts, key="payment")
    payment_type     = st.sidebar.selectbox("Payment Type", payment_type_opts, key="payment_type")
    source_class     = st.sidebar.selectbox("Source Class", source_class_opts, key="source_class")
    waterpoint_type  = st.sidebar.selectbox("Waterpoint Type", waterpoint_type_opts, key="waterpoint_type")

    # ────────────────────────────────────────────────────────────────────
    # 3) Cascade geographic inputs (region, district_code, ward, subvillage)
    # ────────────────────────────────────────────────────────────────────
    geo_vals = user_input_cascading_geo(geo_lookup)
    region        = geo_vals["region"]
    district_code = geo_vals["district_code"]
    ward          = geo_vals["ward"]
    subvillage    = geo_vals["subvillage"]  # pd.NA if “None”

    # 4) Compute region_code from the lookup (one region → one code)
    try:
        region_code = int(
            geo_lookup[geo_lookup["region"].str.lower() == region]["region_code"].dropna().iloc[0]
        )
    except (IndexError, KeyError):
        region_code = pd.NA

    # 5) Infer LGA from the same lookup (one‐to‐one for (region, district_code, ward, subvillage))
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
    # 6) AUTO‐FILL LAT/LON FROM CENTROID (see centroids DataFrame above)
    # ────────────────────────────────────────────────────────────────────
    try:
        cent_row = centroids[
            (centroids["region"] == region)
            & (centroids["district_code"] == district_code)
            & (centroids["ward"] == ward)
        ].iloc[0]
        latitude_centroid  = float(cent_row["latitude_centroid"])
        longitude_centroid = float(cent_row["longitude_centroid"])
    except (IndexError, KeyError):
        latitude_centroid  = pd.NA
        longitude_centroid = pd.NA

    # Show the auto‐fill values and allow override
    st.sidebar.header("Auto‐filled Latitude & Longitude")
    if pd.notna(latitude_centroid) and pd.notna(longitude_centroid):
        st.sidebar.markdown(
            f"ℹ️ Based on ({region}, {district_code}, {ward}):\n"
            f"- Auto latitude = **{latitude_centroid:.4f}**\n"
            f"- Auto longitude = **{longitude_centroid:.4f}**"
        )
    else:
        st.sidebar.markdown("⚠️ No centroid found; please enter latitude/longitude below.")

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
    # 7) Assemble everything into a one‐row DataFrame and return it
    # ────────────────────────────────────────────────────────────────────
    data = {
        # A) Dropped placeholders
        "id":                     pd.NA,
        "num_private":            pd.NA,
        "recorded_by":            pd.NA,
        "extraction_type_group":  pd.NA,
        "quality_group":          pd.NA,
        "quantity_group":         pd.NA,
        "source":                 pd.NA,
        "waterpoint_type_group":  pd.NA,

        # B) Region & codes
        "region":                 region,
        "region_code":            region_code,
        "district_code":          int(district_code),
        "ward":                   ward,
        "subvillage":             subvillage,
        "lga":                    lga,

        # C) Geolocation
        "latitude":               latitude,
        "longitude":              longitude,

        # D) Numeric features
        "population":             population,
        "amount_tsh":             amount_tsh,
        "gps_height":             gps_height,

        # E) Permit / Public Meeting
        "permit":                 int(permit),
        "public_meeting":         int(public_meeting),

        # F) Construction & Age
        "construction_year":      construction_year,
        "date_recorded":          pd.to_datetime(date_recorded).strftime("%Y-%m-%d"),

        # G) Frequency‐encoded columns
        "wpt_name":               wpt_name,
        "scheme_management":      scheme_management,
        "funder":                 funder,
        "installer":              installer,
        "scheme_name":            scheme_name,
        "management":             management,

        # H) LowerCaseStrings raw columns
        "basin":                  basin,
        "source_type":            source_type,
        "water_quality":          water_quality,
        "quantity":               quantity,

        # I) Extraction details
        "extraction_type":        extraction_type,
        "extraction_type_class":  extraction_type_class,

        # J) Management / Payment / Source Class / Waterpoint Type
        "management_group":       management_group,
        "payment":                payment,
        "payment_type":           payment_type,
        "source_class":           source_class,
        "waterpoint_type":        waterpoint_type,
    }

    return pd.DataFrame(data, index=[0])

# Main script
st.title("💧 Pump Status Predictor")

input_df_small = user_input_features()

if st.sidebar.button("🔍 Predict"):
    # 1) Extract every value from input_df_small
    region                = input_df_small.loc[0, "region"]
    region_code           = input_df_small.loc[0, "region_code"]
    district_code         = input_df_small.loc[0, "district_code"]
    ward                  = input_df_small.loc[0, "ward"]
    subvillage            = input_df_small.loc[0, "subvillage"]
    lga                   = input_df_small.loc[0, "lga"]

    latitude              = input_df_small.loc[0, "latitude"]
    longitude             = input_df_small.loc[0, "longitude"]

    population            = input_df_small.loc[0, "population"]
    amount_tsh            = input_df_small.loc[0, "amount_tsh"]
    gps_height            = input_df_small.loc[0, "gps_height"]

    permit                = input_df_small.loc[0, "permit"]
    public_meeting        = input_df_small.loc[0, "public_meeting"]

    construction_year     = input_df_small.loc[0, "construction_year"]
    date_recorded         = input_df_small.loc[0, "date_recorded"]

    wpt_name              = input_df_small.loc[0, "wpt_name"]
    scheme_management     = input_df_small.loc[0, "scheme_management"]
    funder                = input_df_small.loc[0, "funder"]
    installer             = input_df_small.loc[0, "installer"]
    scheme_name           = input_df_small.loc[0, "scheme_name"]
    management            = input_df_small.loc[0, "management"]

    basin                 = input_df_small.loc[0, "basin"]
    source_type           = input_df_small.loc[0, "source_type"]
    water_quality         = input_df_small.loc[0, "water_quality"]
    quantity              = input_df_small.loc[0, "quantity"]

    extraction_type       = input_df_small.loc[0, "extraction_type"]
    extraction_type_class = input_df_small.loc[0, "extraction_type_class"]

    management_group      = input_df_small.loc[0, "management_group"]
    payment               = input_df_small.loc[0, "payment"]
    payment_type          = input_df_small.loc[0, "payment_type"]
    source_class          = input_df_small.loc[0, "source_class"]
    waterpoint_type       = input_df_small.loc[0, "waterpoint_type"]

    # 2) Build the full_input dict containing every raw column
    full_input = {
        # A) Columns dropped by ColumnDropper (placeholders)
        "id":                     pd.NA,
        "num_private":            pd.NA,
        "recorded_by":            pd.NA,
        "extraction_type_group":  pd.NA,
        "quality_group":          pd.NA,
        "quantity_group":         pd.NA,
        "source":                 pd.NA,
        "waterpoint_type_group":  pd.NA,

        # B) Region & codes
        "region":                 region,
        "region_code":            int(region_code),
        "district_code":          int(district_code),
        "ward":                   ward,
        "subvillage":             subvillage,
        "lga":                    lga,

        # C) Geolocation
        "latitude":               latitude,
        "longitude":              longitude,

        # D) Numeric features
        "population":             population,
        "amount_tsh":             amount_tsh,
        "gps_height":             gps_height,

        # E) Permit / Public Meeting
        "permit":                 int(permit),
        "public_meeting":         int(public_meeting),

        # F) Construction & Age
        "construction_year":      construction_year,
        "date_recorded":          date_recorded,

        # G) Frequency‐encoded columns
        "wpt_name":               wpt_name,
        "scheme_management":      scheme_management,
        "funder":                 funder,
        "installer":              installer,
        "scheme_name":            scheme_name,
        "management":             management,

        # H) LowerCaseStrings raw columns
        "basin":                  basin,
        "source_type":            source_type,
        "water_quality":          water_quality,
        "quantity":               quantity,

        # I) Extraction details
        "extraction_type":        extraction_type,
        "extraction_type_class":  extraction_type_class,

        # J) Management / Payment / Source Class / Waterpoint Type
        "management_group":       management_group,
        "payment":                payment,
        "payment_type":           payment_type,
        "source_class":           source_class,
        "waterpoint_type":        waterpoint_type,
    }

    # 3) Convert to a one‐row DataFrame
    input_df = pd.DataFrame(full_input, index=[0])

    # 4) Predict with your pipeline
    y_pred = pipeline.predict(input_df)[0]
    y_proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None

    # 5) Map numeric label → text
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

    # 6) Show location on a map
    st.subheader("📍 Location on Map")
    map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_df, zoom=6)

else:
    st.info("🌟 Adjust parameters and click **Predict** when ready.")
