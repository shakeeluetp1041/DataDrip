# app.py

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


def user_input_features():
    """
    Render one widget for each *raw* column that the pipeline expects.
    Returns a one‐row DataFrame with all of these columns:
      - id, num_private, recorded_by, extraction_type_group,
        quality_group, quantity_group, source, waterpoint_type_group, region
      - region_code, district_code, region_with_code (we will compute this)
      - subvillage, ward, lga, latitude, longitude, population, amount_tsh,
        permit, public_meeting, gps_height, construction_year, wpt_name,
        scheme_management, funder, installer, scheme_name, management
      - date_recorded
      - basin, source_type, water_quality, quantity, extraction_type,
        extraction_type_class, management_group, payment, payment_type,
        source_class, waterpoint_type
    """
    # ────────────────────────────────────────────────────────────────────
    # 1) Build lists of dropdown options (from feature_df, or fallback)
    # ────────────────────────────────────────────────────────────────────
    if feature_df is not None:
        # Numeric codes:
        region_code_opts          = sorted(feature_df["region_code"].dropna().unique().astype(int).tolist())
        district_code_opts        = sorted(feature_df["district_code"].dropna().unique().astype(int).tolist())

        # Categorical/raw‐string columns:
        region_opts               = feature_df["region"].dropna().unique().tolist()
        subvillage_opts           = feature_df["subvillage"].dropna().unique().tolist()
        ward_opts                 = feature_df["ward"].dropna().unique().tolist()
        lga_opts                  = feature_df["lga"].dropna().unique().tolist()

        wpt_name_opts             = feature_df["wpt_name"].dropna().unique().tolist()
        scheme_management_opts    = feature_df["scheme_management"].dropna().unique().tolist()
        funder_opts               = feature_df["funder"].dropna().unique().tolist()
        installer_opts            = feature_df["installer"].dropna().unique().tolist()
        scheme_name_opts          = feature_df["scheme_name"].dropna().unique().tolist()
        management_opts           = feature_df["management"].dropna().unique().tolist()

        basin_opts                = feature_df["basin"].dropna().unique().tolist()
        source_type_opts          = feature_df["source_type"].dropna().unique().tolist()
        water_quality_opts        = feature_df["water_quality"].dropna().unique().tolist()
        quantity_opts             = feature_df["quantity"].dropna().unique().tolist()

        extraction_type_opts      = feature_df["extraction_type"].dropna().unique().tolist()
        extraction_type_class_opts= feature_df["extraction_type_class"].dropna().unique().tolist()

        management_group_opts     = feature_df["management_group"].dropna().unique().tolist()
        payment_opts              = feature_df["payment"].dropna().unique().tolist()
        payment_type_opts         = feature_df["payment_type"].dropna().unique().tolist()
        source_class_opts         = feature_df["source_class"].dropna().unique().tolist()
        waterpoint_type_opts      = feature_df["waterpoint_type"].dropna().unique().tolist()

        permit_opts               = feature_df["permit"].dropna().unique().tolist()
        public_meeting_opts       = feature_df["public_meeting"].dropna().unique().tolist()
    else:
        # FALLBACK: Hard‐code reasonable lists if feature_df is not available.
        region_code_opts           = list(range(1, 31))
        district_code_opts         = list(range(1, 100))  # maybe up to 99 districts
        region_opts                = ["dodoma", "iringa", "kagera", "kigoma", "kilimanjaro",
                                      "lindi", "manyara", "mara", "mbeya", "morogoro", "mtwara",
                                      "mwanza", "pwani", "rukwa", "ruvuma", "shinyanga",
                                      "singida", "tabora", "tanga"]
        subvillage_opts            = ["subvillage_a", "subvillage_b", "other"]
        ward_opts                  = ["ward_a", "ward_b", "other"]
        lga_opts                   = ["lga_a", "lga_b", "other"]

        wpt_name_opts              = ["wpt1", "wpt2", "none"]
        scheme_management_opts     = ["vwc", "water authority", "other"]
        funder_opts                = ["government", "private", "other"]
        installer_opts             = ["company_a", "company_b", "other"]
        scheme_name_opts           = ["scheme_a", "scheme_b", "none"]
        management_opts            = ["vwc", "water authority", "other"]

        basin_opts                 = ["lake victoria", "rufiji", "pangwa", "wami / ruvu", "other"]
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
    # 2) Build sidebar widgets for *every* raw column
    # ────────────────────────────────────────────────────────────────────

    # A) Placeholders for columns we drop (no real widget, just store them as pd.NA later)
    #    id, num_private, recorded_by, extraction_type_group,
    #    quality_group, quantity_group, source, waterpoint_type_group
    #    region itself is used downstream to build region_with_code,
    #      so we DO need a widget for "region" (next line).
    st.sidebar.header("Columns Dropped / Placeholders")
    st.sidebar.write("These columns will be set to pd.NA by default:")
    st.sidebar.write("• id, num_private, recorded_by, extraction_type_group")
    st.sidebar.write("• quality_group, quantity_group, source, waterpoint_type_group")

    # B) region (so RegionCodeCombiner can build region_with_code) & region_code
    st.sidebar.header("Region Inputs")
    region = st.sidebar.selectbox("Region", region_opts, key="region")
    region_code = st.sidebar.selectbox("Region Code", region_code_opts, key="region_code")

    # C) district_code (one‐hot‐encoded)
    st.sidebar.header("District Input")
    district_code = st.sidebar.selectbox("District Code", district_code_opts, key="district_code")

    # D) Subvillage / Ward / LGA (freq‐encoded)
    st.sidebar.header("Subvillage / Ward / LGA")
    subvillage = st.sidebar.selectbox("Subvillage", subvillage_opts, key="subvillage")
    ward = st.sidebar.selectbox("Ward", ward_opts, key="ward")
    lga = st.sidebar.selectbox("LGA", lga_opts, key="lga")

    # E) Latitude / Longitude (numeric)
    st.sidebar.header("Geolocation (Numeric)")
    latitude = st.sidebar.number_input(
        "Latitude (°)", min_value=-12.0, max_value=5.0, value=-6.8, step=0.0001, format="%.5f", key="latitude"
    )
    longitude = st.sidebar.number_input(
        "Longitude (°)", min_value=29.0, max_value=41.0, value=34.7, step=0.0001, format="%.5f", key="longitude"
    )

    # F) Population, amount_tsh, gps_height (numeric)
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

    # G) Permit / Public Meeting (binary 0/1)
    st.sidebar.header("Permit & Public Meeting")
    permit = st.sidebar.selectbox("Permit? (0 = No, 1 = Yes)", permit_opts, key="permit")
    public_meeting = st.sidebar.selectbox("Public Meeting? (0 = No, 1 = Yes)", public_meeting_opts, key="public_meeting")

    # H) Construction year (numeric) & date_recorded (for AgeCalculator)
    st.sidebar.header("Construction Year & Date Recorded")
    construction_year = st.sidebar.number_input(
        "Construction Year", min_value=1900, max_value=2025, value=2005, step=1, key="construction_year"
    )
    date_recorded = st.sidebar.date_input(
        "Date Recorded", value=pd.to_datetime("2020-01-01"), key="date_recorded"
    )

    # I) WPT name / Scheme management / Funder / Installer / Scheme name (freq-encoded)
    st.sidebar.header("WPT & Scheme Details (Frequency‐Encoded)")
    wpt_name = st.sidebar.selectbox("Waterpoint Name", wpt_name_opts, key="wpt_name")
    scheme_management = st.sidebar.selectbox("Scheme Management", scheme_management_opts, key="scheme_management")
    funder = st.sidebar.selectbox("Funder", funder_opts, key="funder")
    installer = st.sidebar.selectbox("Installer", installer_opts, key="installer")
    scheme_name = st.sidebar.selectbox("Scheme Name", scheme_name_opts, key="scheme_name")
    management = st.sidebar.selectbox("Management (Freq‐encoded)", management_opts, key="management")

    # J) Basin / Source Type / Water Quality / Quantity
    st.sidebar.header("Other LowerCaseStrings Columns")
    basin = st.sidebar.selectbox("Basin", basin_opts, key="basin")
    source_type = st.sidebar.selectbox("Source Type", source_type_opts, key="source_type")
    water_quality = st.sidebar.selectbox("Water Quality", water_quality_opts, key="water_quality")
    quantity = st.sidebar.selectbox("Quantity", quantity_opts, key="quantity")

    # K) Extraction Type / Extraction Type Class
    st.sidebar.header("Extraction Details")
    extraction_type = st.sidebar.selectbox("Extraction Type", extraction_type_opts, key="extraction_type")
    extraction_type_class = st.sidebar.selectbox("Extraction Type Class", extraction_type_class_opts, key="extraction_type_class")

    # L) Management Group / Payment / Payment Type / Source Class / Waterpoint Type
    st.sidebar.header("Management & Payment & Source Class")
    management_group = st.sidebar.selectbox("Management Group", management_group_opts, key="management_group")
    payment = st.sidebar.selectbox("Payment (raw)", payment_opts, key="payment")
    payment_type = st.sidebar.selectbox("Payment Type", payment_type_opts, key="payment_type")
    source_class = st.sidebar.selectbox("Source Class", source_class_opts, key="source_class")
    waterpoint_type = st.sidebar.selectbox("Waterpoint Type", waterpoint_type_opts, key="waterpoint_type")

    # ────────────────────────────────────────────────────────────────────
    # 4) Assemble everything into a one‐row DataFrame and return it
    # ────────────────────────────────────────────────────────────────────
    data = {
        # A) Dropped placeholders (still needed as keys but set to NA later)
        "id":                     pd.NA,
        "num_private":            pd.NA,
        "recorded_by":            pd.NA,
        "extraction_type_group":  pd.NA,
        "quality_group":          pd.NA,
        "quantity_group":         pd.NA,
        "source":                 pd.NA,
        "waterpoint_type_group":  pd.NA,
        "region":                 region,            # used by RegionCodeCombiner

        # B) Region Code (numeric)
        "region_code":            int(region_code),

        # C) District + Geo fields
        "district_code":          int(district_code),
        "subvillage":             subvillage,
        "ward":                   ward,
        "lga":                    lga,
        "latitude":               latitude,
        "longitude":              longitude,

        # D) Numerics
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

input_df_small = user_input_features()

if st.sidebar.button("🔍 Predict"):
    # 1) Pull each user‐selected value out of input_df_small
    region                = input_df_small.loc[0, "region"]
    region_code           = input_df_small.loc[0, "region_code"]
    district_code         = input_df_small.loc[0, "district_code"]

    subvillage            = input_df_small.loc[0, "subvillage"]
    ward                  = input_df_small.loc[0, "ward"]
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

    # 2) Build the full_input dict with *every* raw column (no more pd.NA!)
    full_input = {
        # ────────────────────────────────────────────────────────────
        # A) Columns dropped by ColumnDropper (placeholders)
        # ────────────────────────────────────────────────────────────
        "id":                     pd.NA,
        "num_private":            pd.NA,
        "recorded_by":            pd.NA,
        "extraction_type_group":  pd.NA,
        "quality_group":          pd.NA,
        "quantity_group":         pd.NA,
        "source":                 pd.NA,
        "waterpoint_type_group":  pd.NA,
        "region":                 region,           # used by RegionCodeCombiner

        # ────────────────────────────────────────────────────────────
        # B) RegionCodeCombiner needs region + region_code
        # ────────────────────────────────────────────────────────────
        "region_code":            int(region_code),

        # ────────────────────────────────────────────────────────────
        # C) GeoContextImputer / frequency‐encoded raw columns
        # ────────────────────────────────────────────────────────────
        "district_code":          int(district_code),
        "subvillage":             subvillage,
        "ward":                   ward,
        "lga":                    lga,
        "latitude":               latitude,
        "longitude":              longitude,
        "population":             population,
        "amount_tsh":             amount_tsh,
        "permit":                 int(permit),
        "public_meeting":         int(public_meeting),
        "gps_height":             gps_height,
        "construction_year":      construction_year,
        "wpt_name":               wpt_name,
        "scheme_management":      scheme_management,
        "funder":                 funder,
        "installer":              installer,
        "scheme_name":            scheme_name,
        "management":             management,

        # ────────────────────────────────────────────────────────────
        # D) AgeCalculator columns
        # ────────────────────────────────────────────────────────────
        "date_recorded":          date_recorded,

        # ────────────────────────────────────────────────────────────
        # E) LowerCaseStrings raw columns
        # ────────────────────────────────────────────────────────────
        "basin":                  basin,
        "source_type":            source_type,
        "water_quality":          water_quality,
        "quantity":               quantity,

        # ────────────────────────────────────────────────────────────
        # F) Extraction‐related raw columns
        # ────────────────────────────────────────────────────────────
        "extraction_type":        extraction_type,
        "extraction_type_class":  extraction_type_class,

        # ────────────────────────────────────────────────────────────
        # G) Management, Payment, Source Class, Waterpoint Type
        # ────────────────────────────────────────────────────────────
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
