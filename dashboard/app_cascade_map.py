# app_clickmap.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from math import radians, cos, sin, sqrt, atan2
import folium
from streamlit_folium import st_folium

# 0) Page config
st.set_page_config(page_title="Pump Status Predictor (Click-Map)", layout="wide")

# 1) Load preprocessors & pipeline
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
    GeoContextImputer,
)

@st.cache_resource
def load_full_pipeline(path: str = "full_pipeline.joblib"):
    return joblib.load(path)

pipeline = load_full_pipeline("full_pipeline.joblib")

# 2) Load feature metadata
@st.cache_data
def load_feature_metadata(csv_path: str = "../data/Training_Set_Values.csv"):
    return pd.read_csv(csv_path)

feature_df = load_feature_metadata()

# 3) Load geo_lookup.csv
@st.cache_data
def load_geo_lookup(path: str = "../data/geo_lookup.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["region"]     = df["region"].astype(str).str.lower()
    df["ward"]       = df["ward"].astype(str).str.lower()
    df["subvillage"] = df["subvillage"].astype(str).str.lower()
    df["lga"]        = df["lga"].astype(str).str.lower()
    df["basin"]      = df["basin"].astype(str).str.lower()
    df["latitude"]   = df["latitude"].astype(float)
    df["longitude"]  = df["longitude"].astype(float)
    return df

geo_lookup = load_geo_lookup()

# 4) Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ     = radians(lat2 - lat1)
    Δλ     = radians(lon2 - lon1)
    a      = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    c      = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# 5) Title & instructions
st.title("💧 Pump Status Predictor (Click-Map)")

st.markdown(
    """
    **Step 1:** Click anywhere on the map below to choose a latitude/longitude.  
    We will infer the nearest `(region, district, ward, subvillage, lga, basin, region_code)`  
    from **geo_lookup.csv**.  
    **Step 2:** Fill in the remaining dropdowns on the right and click **Predict**.
    """
)

# 6) Render Folium map
m = folium.Map(location=[-6.5, 34.0], zoom_start=6)
map_data = st_folium(m, width=700, height=300, returned_objects=["last_clicked"], key="click_map")

last_click = map_data.get("last_clicked")
if last_click is None:
    st.info("🖱️ Click on the map to pick a location.")
    st.stop()

clicked_lat = last_click["lat"]
clicked_lon = last_click["lng"]

# 7) Sidebar: show clicked location
st.sidebar.subheader("1) Clicked Location")
st.sidebar.write(f"Latitude:  {clicked_lat:.5f}")
st.sidebar.write(f"Longitude: {clicked_lon:.5f}")

# 8) Find nearest geo_lookup row
distances = geo_lookup.apply(
    lambda row: haversine(clicked_lat, clicked_lon, row["latitude"], row["longitude"]),
    axis=1,
)
best_idx = distances.idxmin()
best_row = geo_lookup.loc[best_idx]

inferred_region        = best_row["region"]
inferred_region_code   = int(best_row["region_code"])
inferred_district_code = int(best_row["district_code"])
inferred_ward          = best_row["ward"]
inferred_subvillage    = best_row["subvillage"]
inferred_lga           = best_row["lga"]
inferred_basin         = best_row["basin"]

# 9) Sidebar: display inferred geography
st.sidebar.subheader("2) Inferred Geography")
st.sidebar.write(f"• Region:         **{inferred_region}**")
st.sidebar.write(f"• Region Code:    **{inferred_region_code}**")
st.sidebar.write(f"• District Code:  **{inferred_district_code}**")
st.sidebar.write(f"• Ward:           **{inferred_ward}**")
st.sidebar.write(f"• Subvillage:     **{inferred_subvillage}**")
st.sidebar.write(f"• LGA:            **{inferred_lga}**")
st.sidebar.write(f"• Basin:          **{inferred_basin}**")

# 10) Sidebar: Other required dropdowns / inputs
st.sidebar.header("3) Other Inputs (Please fill in)")

population = st.sidebar.number_input(
    "Population Served", 0, 50000, 1000, 50, key="population"
)
amount_tsh = st.sidebar.number_input(
    "Total Static Head (TSH)", 0, 500, 50, 5, key="amount_tsh"
)
gps_height = st.sidebar.number_input(
    "GPS Height (m)", -50, 5000, 100, 10, key="gps_height"
)

st.sidebar.header("Permit & Public Meeting")
permit         = st.sidebar.selectbox("Permit? (0=No, 1=Yes)", [0, 1], key="permit")
public_meeting = st.sidebar.selectbox("Public Meeting? (0=No, 1=Yes)", [0, 1], key="public_meeting")

st.sidebar.header("Construction Year & Date Recorded")
construction_year = st.sidebar.number_input(
    "Construction Year", 1900, 2025, 2005, 1, key="construction_year"
)
date_recorded = st.sidebar.date_input(
    "Date Recorded", value=pd.to_datetime("2020-01-01"), key="date_recorded"
)

# 11) WPT & Scheme Details (pull actual options from feature_df)
wpt_name_opts          = sorted(feature_df["wpt_name"].dropna().unique().tolist())
scheme_management_opts = sorted(feature_df["scheme_management"].dropna().unique().tolist())
funder_opts            = sorted(feature_df["funder"].dropna().unique().tolist())
installer_opts         = sorted(feature_df["installer"].dropna().unique().tolist())
scheme_name_opts       = sorted(feature_df["scheme_name"].dropna().unique().tolist())
management_opts        = sorted(feature_df["management"].dropna().unique().tolist())

st.sidebar.header("WPT & Scheme Details (Frequency‐Encoded)")
wpt_name          = st.sidebar.selectbox("Waterpoint Name", wpt_name_opts, key="wpt_name")
scheme_management = st.sidebar.selectbox("Scheme Management", scheme_management_opts, key="scheme_management")
funder            = st.sidebar.selectbox("Funder", funder_opts, key="funder")
installer         = st.sidebar.selectbox("Installer", installer_opts, key="installer")
scheme_name       = st.sidebar.selectbox("Scheme Name", scheme_name_opts, key="scheme_name")
management        = st.sidebar.selectbox("Management", management_opts, key="management")

st.sidebar.header("Other Categorical Inputs")
water_quality = st.sidebar.selectbox(
    "Water Quality",
    sorted(feature_df["water_quality"].dropna().unique().tolist()),
    key="water_quality",
)
quantity = st.sidebar.selectbox(
    "Quantity",
    sorted(feature_df["quantity"].dropna().unique().tolist()),
    key="quantity",
)

st.sidebar.header("Extraction Details")
extraction_type_opts       = sorted(feature_df["extraction_type"].dropna().unique().tolist())
extraction_type_class_opts = sorted(feature_df["extraction_type_class"].dropna().unique().tolist())

extraction_type       = st.sidebar.selectbox("Extraction Type", extraction_type_opts, key="extraction_type")
extraction_type_class = st.sidebar.selectbox("Extraction Type Class", extraction_type_class_opts, key="extraction_type_class")

st.sidebar.header("Management, Payment, Source Class, Waterpoint Type")
management_group_opts = sorted(feature_df["management_group"].dropna().unique().tolist())
payment_opts          = sorted(feature_df["payment"].dropna().unique().tolist())
payment_type_opts     = sorted(feature_df["payment_type"].dropna().unique().tolist())
source_class_opts     = sorted(feature_df["source_class"].dropna().unique().tolist())
source_type_opts      = sorted(feature_df["source_type"].dropna().unique().tolist())
waterpoint_type_opts  = sorted(feature_df["waterpoint_type"].dropna().unique().tolist())

management_group = st.sidebar.selectbox("Management Group", management_group_opts, key="management_group")
payment          = st.sidebar.selectbox("Payment (raw)", payment_opts, key="payment")
payment_type     = st.sidebar.selectbox("Payment Type", payment_type_opts, key="payment_type")
source_class     = st.sidebar.selectbox("Source Class", source_class_opts, key="source_class")
source_type      = st.sidebar.selectbox("Source Type", source_type_opts, key="source_type")
waterpoint_type  = st.sidebar.selectbox("Waterpoint Type", waterpoint_type_opts, key="waterpoint_type")

# 12) Render the Predict button in a top‐level block (not nested)
predict_clicked = st.sidebar.button("🔍 Predict")

# 13) If clicked, attempt to run pipeline.predict and catch any error
if predict_clicked:
    try:
        region_with_code = f"{inferred_region}_{inferred_region_code}"

        full_input = {
            # Dropped placeholders
            "id":                    pd.NA,
            "num_private":           pd.NA,
            "recorded_by":           pd.NA,
            "extraction_type_group": pd.NA,
            "quality_group":         pd.NA,
            "quantity_group":        pd.NA,
            "source":                pd.NA,
            "waterpoint_type_group": pd.NA,

            # Inferred geography
            "region":          inferred_region,
            "region_code":     inferred_region_code,
            "district_code":   inferred_district_code,
            "ward":            inferred_ward,
            "subvillage":      inferred_subvillage,
            "lga":             inferred_lga,
            "basin":           inferred_basin,
            "region_with_code": region_with_code,

            # Click coordinates
            "latitude":  float(clicked_lat),
            "longitude": float(clicked_lon),

            # Numerics
            "population":  population,
            "amount_tsh":  amount_tsh,
            "gps_height":  gps_height,

            # Binary
            "permit":         int(permit),
            "public_meeting": int(public_meeting),

            # Construction & date
            "construction_year": construction_year,
            "date_recorded":     pd.to_datetime(date_recorded).strftime("%Y-%m-%d"),

            # WPT/Scheme
            "wpt_name":          wpt_name,
            "scheme_management": scheme_management,
            "funder":            funder,
            "installer":         installer,
            "scheme_name":       scheme_name,
            "management":        management,

            # Categorical
            "water_quality": water_quality,
            "quantity":      quantity,
            "source_type":   source_type,
            "source_class":  source_class,

            # Extraction
            "extraction_type":       extraction_type,
            "extraction_type_class": extraction_type_class,

            # Other
            "management_group": management_group,
            "payment":          payment,
            "payment_type":     payment_type,
            "waterpoint_type":  waterpoint_type,
        }

        input_df = pd.DataFrame(full_input, index=[0])
        y_pred  = pipeline.predict(input_df)[0]
        y_proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None

        status_map             = {0: "Non-Functional", 1: "Needs Repair", 2: "Functional"}
        st.session_state["last_prediction"] = status_map.get(y_pred, str(y_pred))
        st.session_state["last_proba"]      = y_proba
        st.session_state["last_lat"]       = clicked_lat
        st.session_state["last_lon"]       = clicked_lon

    except Exception as e:
        st.error(f"❌ Prediction failed:\n{e}")

# 14) After button block: display stored prediction (if present)
if "last_prediction" in st.session_state:
    st.subheader("📝 Prediction Result")
    st.write(f"**Predicted Pump Status:** :green[{st.session_state['last_prediction']}]")

    if st.session_state.get("last_proba") is not None:
        proba_df = pd.DataFrame({
            "Status": ["Non-Functional", "Needs Repair", "Functional"],
            "Probability": st.session_state["last_proba"]
        }).set_index("Status")
        st.bar_chart(proba_df)

    # Show marker at clicked location
    lat0 = st.session_state["last_lat"]
    lon0 = st.session_state["last_lon"]
    result_map = folium.Map(location=[lat0, lon0], zoom_start=8)
    folium.Marker(
        [lat0, lon0],
        popup=st.session_state["last_prediction"],
        icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
    ).add_to(result_map)
    st.subheader("📍 Selected Location (with Prediction)")
    st_folium(result_map, width=700, height=400, key="result_map")

else:
    st.info("🌟 After clicking on the map and filling in the form, press **Predict**.")

