# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, sqrt, atan2
import json

# ─────────────────────────────────────────────────────────────────────────────
# 0) Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pump Status Predictor", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load preprocessors & pipeline
# ─────────────────────────────────────────────────────────────────────────────
from preprocessors import (
    LowerCaseStrings, StringConverter, YearExtractor, IQRCapper,
    ConstructionYearTransformer, ObjectToNumericConverter, AgeCalculator,
    FrequencyEncoder, RegionCodeCombiner, ColumnDropper, AgePipeline,
    GeoContextImputer,
)

@st.cache_resource
def load_full_pipeline(path: str = "full_pipeline.joblib"):
    return joblib.load(path)

pipeline = load_full_pipeline("full_pipeline.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load feature metadata & geo_lookup
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_feature_metadata(csv_path: str = "../data/Training_Set_Values.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

feature_df = load_feature_metadata()

@st.cache_data
def load_geo_lookup(path: str = "../data/geo_lookup.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["region"]        = df["region"].str.lower()
    df["ward"]          = df["ward"].str.lower()
    df["subvillage"]    = df["subvillage"].str.lower()
    df["lga"]           = df["lga"].str.lower()
    df["basin"]         = df["basin"].str.lower()
    df["region_code"]   = df["region_code"].astype(int)
    df["district_code"] = df["district_code"].astype(int)
    df["latitude"]      = df["latitude"].astype(float)
    df["longitude"]     = df["longitude"].astype(float)
    return df

geo_lookup = load_geo_lookup()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Haversine distance (in km)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ     = radians(lat2 - lat1)
    Δλ     = radians(lon2 - lon1)
    a      = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    c      = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ─────────────────────────────────────────────────────────────────────────────
# 4) Sidebar: choose mode
# ─────────────────────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Select Mode", ["Interactive", "Batch/Test Set"])

if mode == "Interactive":
    st.header("💧 Pump Status Predictor (Interactive Mode)")

    # ─── 4a) Sidebar: click‑to‑pick map ────────────────────────────────────────────
    with st.sidebar:
        st.subheader("1) Click on Map to Pick Location")
        base_map = folium.Map(location=[-6.5, 34.0], zoom_start=6)
        map_data = st_folium(
            base_map,
            height=350,
            width=350,
            returned_objects=["last_clicked"],
            key="map_int"
        )

    last_click = map_data.get("last_clicked")
    if not last_click:
        st.sidebar.info("🖱️  Please click on the map above")
        st.stop()

    clicked_lat  = last_click["lat"]
    clicked_lon  = last_click["lng"]
    st.sidebar.write(f"Latitude: {clicked_lat:.5f}   |   Longitude: {clicked_lon:.5f}")

    # ─── 4b) Infer geography from geo_lookup ─────────────────────────────────────
    distances = geo_lookup.apply(
        lambda row: haversine(clicked_lat, clicked_lon, row["latitude"], row["longitude"]),
        axis=1
    )
    best_row = geo_lookup.loc[distances.idxmin()]
    inferred_region        = best_row["region"]
    inferred_region_code   = best_row["region_code"]
    inferred_district_code = best_row["district_code"]
    inferred_ward          = best_row["ward"]
    inferred_subvillage    = best_row["subvillage"]
    inferred_lga           = best_row["lga"]
    inferred_basin         = best_row["basin"]

    st.sidebar.subheader("2) Inferred Geography")
    st.sidebar.write(f"• Region: **{inferred_region}**")
    st.sidebar.write(f"• Region Code: **{inferred_region_code}**")
    st.sidebar.write(f"• District Code: **{inferred_district_code}**")
    st.sidebar.write(f"• Ward: **{inferred_ward}**")
    st.sidebar.write(f"• Subvillage: **{inferred_subvillage}**")
    st.sidebar.write(f"• LGA: **{inferred_lga}**")
    st.sidebar.write(f"• Basin: **{inferred_basin}**")

    # ─── 4c) Sidebar: other inputs ───────────────────────────────────────────────
    st.sidebar.markdown("### 3) Other Inputs")
    population   = st.sidebar.number_input("Population Served", 0, 50000, 1000, 50)
    amount_tsh   = st.sidebar.number_input("Total Static Head (TSH)", 0, 500, 50, 5)
    gps_height   = st.sidebar.number_input("GPS Height (m)", -50, 5000, 100, 10)

    st.sidebar.markdown("**Permit & Public Meeting**")
    permit         = st.sidebar.selectbox("Permit? (0=No, 1=Yes)", [0, 1])
    public_meeting = st.sidebar.selectbox("Public Meeting? (0=No, 1=Yes)", [0, 1])

    st.sidebar.markdown("**Construction Year & Date Recorded**")
    construction_year = st.sidebar.number_input("Construction Year", 1900, 2025, 2005, 1)
    date_recorded     = st.sidebar.date_input("Date Recorded", pd.to_datetime("2020-01-01"))

    st.sidebar.markdown("**WPT & Scheme Details (Frequency-Encoded)**")
    wpt_name_opts          = sorted(feature_df["wpt_name"].dropna().unique().tolist())
    scheme_mgmt_opts       = sorted(feature_df["scheme_management"].dropna().unique().tolist())
    funder_opts            = sorted(feature_df["funder"].dropna().unique().tolist())
    installer_opts         = sorted(feature_df["installer"].dropna().unique().tolist())
    scheme_name_opts       = sorted(feature_df["scheme_name"].dropna().unique().tolist())
    management_opts        = sorted(feature_df["management"].dropna().unique().tolist())

    wpt_name          = st.sidebar.selectbox("Waterpoint Name", wpt_name_opts)
    scheme_management = st.sidebar.selectbox("Scheme Management", scheme_mgmt_opts)
    funder            = st.sidebar.selectbox("Funder", funder_opts)
    installer         = st.sidebar.selectbox("Installer", installer_opts)
    scheme_name       = st.sidebar.selectbox("Scheme Name", scheme_name_opts)
    management        = st.sidebar.selectbox("Management", management_opts)

    st.sidebar.markdown("**Other Categorical Inputs**")
    water_quality = st.sidebar.selectbox(
        "Water Quality",
        sorted(feature_df["water_quality"].dropna().unique().tolist())
    )
    quantity      = st.sidebar.selectbox(
        "Quantity",
        sorted(feature_df["quantity"].dropna().unique().tolist())
    )

    st.sidebar.markdown("**Extraction Details**")
    extraction_type_opts       = sorted(feature_df["extraction_type"].dropna().unique().tolist())
    extraction_type_class_opts = sorted(feature_df["extraction_type_class"].dropna().unique().tolist())

    extraction_type       = st.sidebar.selectbox("Extraction Type", extraction_type_opts)
    extraction_type_class = st.sidebar.selectbox("Extraction Type Class", extraction_type_class_opts)

    st.sidebar.markdown("**Management, Payment, Source Class, Waterpoint Type**")
    management_group_opts = sorted(feature_df["management_group"].dropna().unique().tolist())
    payment_opts          = sorted(feature_df["payment"].dropna().unique().tolist())
    payment_type_opts     = sorted(feature_df["payment_type"].dropna().unique().tolist())
    source_class_opts     = sorted(feature_df["source_class"].dropna().unique().tolist())
    source_type_opts      = sorted(feature_df["source_type"].dropna().unique().tolist())
    waterpoint_type_opts  = sorted(feature_df["waterpoint_type"].dropna().unique().tolist())

    management_group = st.sidebar.selectbox("Management Group", management_group_opts)
    payment          = st.sidebar.selectbox("Payment (raw)", payment_opts)
    payment_type     = st.sidebar.selectbox("Payment Type", payment_type_opts)
    source_class     = st.sidebar.selectbox("Source Class", source_class_opts)
    source_type      = st.sidebar.selectbox("Source Type", source_type_opts)
    waterpoint_type  = st.sidebar.selectbox("Waterpoint Type", waterpoint_type_opts)

    # ─── 4d) Predict button ──────────────────────────────────────────────────────
    predict_clicked = st.sidebar.button("🔍 Predict")
    st.write("Follow steps in the sidebar, then press **Predict** to see a single‑pump prediction.")

    if predict_clicked:
        region_with_code = f"{inferred_region}_{inferred_region_code}"
        full_input = {
            "id":                    pd.NA,
            "num_private":           pd.NA,
            "recorded_by":           pd.NA,
            "extraction_type_group": pd.NA,
            "quality_group":         pd.NA,
            "quantity_group":        pd.NA,
            "source":                pd.NA,
            "waterpoint_type_group": pd.NA,
            "region":                inferred_region,
            "region_code":           inferred_region_code,
            "district_code":         inferred_district_code,
            "ward":                  inferred_ward,
            "subvillage":            inferred_subvillage,
            "lga":                   inferred_lga,
            "basin":                 inferred_basin,
            "region_with_code":      region_with_code,
            "latitude":              float(clicked_lat),
            "longitude":             float(clicked_lon),
            "population":            population,
            "amount_tsh":            amount_tsh,
            "gps_height":            gps_height,
            "permit":                int(permit),
            "public_meeting":        int(public_meeting),
            "construction_year":     construction_year,
            "date_recorded":         pd.to_datetime(date_recorded).strftime("%Y-%m-%d"),
            "wpt_name":              wpt_name,
            "scheme_management":     scheme_management,
            "funder":                funder,
            "installer":             installer,
            "scheme_name":           scheme_name,
            "management":            management,
            "water_quality":         water_quality,
            "quantity":              quantity,
            "source_type":           source_type,
            "source_class":          source_class,
            "extraction_type":       extraction_type,
            "extraction_type_class": extraction_type_class,
            "management_group":      management_group,
            "payment":               payment,
            "payment_type":          payment_type,
            "waterpoint_type":       waterpoint_type,
        }
        input_df = pd.DataFrame(full_input, index=[0])
        try:
            y_pred  = pipeline.predict(input_df)[0]
            y_proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None
            status_map     = {0: "Non-Functional", 1: "Needs Repair", 2: "Functional"}
            predicted_label = status_map.get(y_pred, str(y_pred))

            st.subheader("📝 Prediction Result")
            st.write(f"**Predicted Pump Status:** :green[{predicted_label}]")
            if y_proba is not None:
                proba_df = pd.DataFrame({
                    "Status": ["Non-Functional", "Needs Repair", "Functional"],
                    "Probability": y_proba
                }).set_index("Status")
                st.bar_chart(proba_df)

            st.subheader("📍 Location on Map")
            map_df = pd.DataFrame({"lat": [clicked_lat], "lon": [clicked_lon]})
            st.map(map_df, zoom=6)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    st.header("💧 Pump Status Predictor (Batch/Test‑Set Mode)")
    test_values = pd.read_csv("../data/Test_Set_Values.csv")

    # ─── Only show "Running batch predictions" once ─────────────────────────────
    if "test_preds" not in st.session_state:
        st.sidebar.info("Running batch predictions on the entire test set…")
        st.info("🔄 Computing predictions for the test set…")
        test_input = test_values.copy()
        preds      = pipeline.predict(test_input)
        proba_mat  = pipeline.predict_proba(test_input) if hasattr(pipeline, "predict_proba") else None
        st.session_state["test_preds"] = preds
        st.session_state["test_proba"] = proba_mat

    status_map = {0: "Non-Functional", 1: "Needs Repair", 2: "Functional"}
    test_values["predicted_code"]  = st.session_state["test_preds"]
    test_values["predicted_label"] = test_values["predicted_code"].map(status_map)
    test_df     = test_values.copy()
    test_df["pred_label"] = test_df["predicted_label"]

    view = st.sidebar.radio(
        "Choose a View",
        ["View Functionality Map", "Region‑wise Map", "Water Quality Map"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # View 1: Functionality Map (coloured points)
    # ─────────────────────────────────────────────────────────────────────────
    if view == "View Functionality Map":
        st.subheader("📍 Test Set – Pumps Coloured by Predicted Status")
        base = folium.Map(location=[-6.5, 34.0], zoom_start=6)
        color_map = {"Functional": "green", "Needs Repair": "orange", "Non-Functional": "red"}
        for _, row in test_df.iterrows():
            lat   = row["latitude"]
            lon   = row["longitude"]
            label = row["pred_label"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color_map.get(label, "gray"),
                fill=True,
                fill_color=color_map.get(label, "gray"),
                fill_opacity=0.7,
            ).add_to(base)
        st_folium(base, width=900, height=600)

    # ─────────────────────────────────────────────────────────────────────────
    # View 2: Region‑wise Map (zoom-dependent choropleths)
    # ─────────────────────────────────────────────────────────────────────────
    elif view == "Region‑wise Map":
        st.subheader("📊 Region‑wise % Functional (Choropleth)")

        # 1) Aggregate by region
        agg_region = (
            test_df
            .groupby("region")["pred_label"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["Functional", "Needs Repair", "Non-Functional"]:
            if col not in agg_region.columns:
                agg_region[col] = 0
        agg_region["Total"]             = agg_region[["Functional","Needs Repair","Non-Functional"]].sum(axis=1)
        agg_region["pct_functional"]    = (agg_region["Functional"] / agg_region["Total"] * 100).round(1)
        region_summary_df = agg_region.rename(columns={"region": "NAME_1", "pct_functional": "pct_functional"})

        # 2) Aggregate by district_code
        agg_district = (
            test_df
            .groupby("district_code")["pred_label"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["Functional", "Needs Repair", "Non-Functional"]:
            if col not in agg_district.columns:
                agg_district[col] = 0
        agg_district["Total"]          = agg_district[["Functional","Needs Repair","Non-Functional"]].sum(axis=1)
        agg_district["pct_functional"] = (agg_district["Functional"] / agg_district["Total"] * 100).round(1)

        # 2a) Load district-level GeoJSON and inspect properties
        geo_district = json.load(open("../data/gadm41_TZA_shp/gadm41_TZA_2.json"))
        first_props = geo_district["features"][0]["properties"]
        st.sidebar.write("District GeoJSON properties keys:", list(first_props.keys()))

        # 2b) Build district_code → NAME_2 mapping
        code_to_name2 = {
            feat["properties"]["ID_2"]: feat["properties"]["NAME_2"].lower()
            for feat in geo_district["features"]
            if feat["properties"].get("ID_2") is not None
        }
        agg_district["NAME_2"] = agg_district["district_code"].map(code_to_name2)
        district_summary_df = agg_district[["NAME_2", "pct_functional"]]

        # 3) Aggregate by ward (assuming ward names match NAME_3)
        agg_ward = (
            test_df
            .groupby("ward")["pred_label"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["Functional", "Needs Repair", "Non-Functional"]:
            if col not in agg_ward.columns:
                agg_ward[col] = 0
        agg_ward["Total"]             = agg_ward[["Functional","Needs Repair","Non-Functional"]].sum(axis=1)
        agg_ward["pct_functional"]    = (agg_ward["Functional"] / agg_ward["Total"] * 100).round(1)
        agg_ward["NAME_3"]            = agg_ward["ward"].str.lower()
        ward_summary_df               = agg_ward[["NAME_3","pct_functional"]]

        # 4) Load GeoJSON files for regions, districts, wards
        geo_region   = json.load(open("../data/gadm41_TZA_shp/gadm41_TZA_1.json"))
        geo_ward     = json.load(open("../data/gadm41_TZA_shp/gadm41_TZA_3.json"))

        # 5) Build base folium map
        choromap = folium.Map(location=[-6.5,34.0], zoom_start=6)

        # 6) Add region choropleth (shown when zoom < 7)
        folium.Choropleth(
            geo_data=geo_region,
            name="Regions",
            data=region_summary_df,
            columns=["NAME_1","pct_functional"],
            key_on="feature.properties.NAME_1",
            fill_color="YlGnBu",
            show=True,
            highlight=True,
            legend_name="% Functional (Region)",
        ).add_to(choromap)

        # 7) Add district choropleth (shown when 7 ≤ zoom < 10)
        folium.Choropleth(
            geo_data=geo_district,
            name="Districts",
            data=district_summary_df,
            columns=["NAME_2","pct_functional"],
            key_on="feature.properties.NAME_2",
            fill_color="YlOrRd",
            show=False,
            highlight=True,
            legend_name="% Functional (District)",
        ).add_to(choromap)

        # 8) Add ward choropleth (shown when zoom ≥ 10)
        folium.Choropleth(
            geo_data=geo_ward,
            name="Wards",
            data=ward_summary_df,
            columns=["NAME_3","pct_functional"],
            key_on="feature.properties.NAME_3",
            fill_color="PuBuGn",
            show=False,
            highlight=True,
            legend_name="% Functional (Ward)",
        ).add_to(choromap)

        # 9) Add layer control (so user can toggle manually if desired)
        folium.LayerControl(collapsed=False).add_to(choromap)

        # 10) Custom JS to toggle layers by zoom level
        toggle_js = """
            function toggleLayers() {
                var map = {{map}};
                var zoom = map.getZoom();
                var layers = map._layers;
                for (var i in layers) {
                    var layer = layers[i];
                    if (layer.options && layer.options.name == 'Regions') {
                        if (zoom < 7) { map.addLayer(layer); } else { map.removeLayer(layer); }
                    }
                    if (layer.options && layer.options.name == 'Districts') {
                        if (zoom >= 7 && zoom < 10) { map.addLayer(layer); } else { map.removeLayer(layer); }
                    }
                    if (layer.options && layer.options.name == 'Wards') {
                        if (zoom >= 10) { map.addLayer(layer); } else { map.removeLayer(layer); }
                    }
                }
            }
            {{map}}.on('zoomend', toggleLayers);
            toggleLayers();
        """
        choromap.get_root().script.add_child(folium.Element(toggle_js.replace("{{map}}","map")))

        # 11) Display the map
        st_folium(choromap, width=900, height=600)

    # ─────────────────────────────────────────────────────────────────────────
    # View 3: Water Quality Map (points by water_quality & popup)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        # 3) Water Quality Map (points by water_quality & popup)
        st.subheader("📈 Test Set – Pumps by Water Quality & Status")
        wq_map = folium.Map(location=[-6.5, 34.0], zoom_start=6)

        # Generate a small palette (one hex color per unique quality)
        unique_wq = test_df["water_quality"].unique().tolist()
        base_colors = [
            "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00",
            "#6a3d9a", "#b15928", "#a6cee3", "#b2df8a",
            "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"
        ]
        palette = base_colors[: len(unique_wq)]
        wq_color_map = dict(zip(unique_wq, palette))

        for _, row in test_df.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            wq = row["water_quality"]
            qty = row["quantity"]
            status = row["pred_label"]
            color = wq_color_map.get(wq, "#808080")
            popup_html = f"<b>Status:</b> {status}<br><b>Quality:</b> {wq}<br><b>Quantity:</b> {qty}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(wq_map)

        # Build a simple vertical legend
        legend_html = """
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 150px; height: {h}px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white; padding: 10px;">
         <b>Water Quality Legend</b><br>
        """.format(h=30 + 20 * len(unique_wq))

        for wq, col in wq_color_map.items():
            legend_html += (
                f"<i style=\"background:{col};width:10px;height:10px;"
                "float:left;margin-right:5px;\"></i>{wq}<br>"
            )
        legend_html += "</div>"

        wq_map.get_root().html.add_child(folium.Element(legend_html))
        st_folium(wq_map, width=900, height=600)

