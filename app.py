import streamlit as st
import datetime
import pandas as pd

import json

# Set default to today's date
default_date = datetime.date.today()
# Date input widget with calendar popup
selected_date = st.date_input("Select a date:", value=default_date) # after preprpocessing, the yeaer will be extracted
data = {'date_recorded': [selected_date]}
df = pd.DataFrame(data)


selected_gps_height = st.number_input("Enter GPS Height (in meters)", min_value=-90, max_value=2770, step=1)
st.caption("Range is from 0 to 2770 meters. Enter the Altitude.")
df['gps_height'] = selected_gps_height 

selected_longitude = st.number_input("Enter Longitude", min_value=0.0,   max_value=40.0, step=0.000001,  format="%.6f" )  # display format to 6 decimal places
st.caption("Range is from 0 to 40. Enter the longitude where the pump is located.")
df['longitude'] = selected_longitude 

selected_latitude = st.number_input("Enter Latitude", min_value=-11.0,   max_value=0.0, step=0.000001,  format="%.6f" )  # display format to 6 decimal places
st.caption("Range is from -11 to 0. Enter the latitude where the pump is located.")
df['latitude'] = selected_latitude

# Dealing with categorical features
with open("categories.json", "r") as f:  # Load the JSON file (saved categories)
    categories = json.load(f)

# Streamlit inputs for each categorical variable
selected_basin = st.selectbox("Select Basin", options=categories['basin'])
df['basin'] = selected_basin

selected_region = st.selectbox("Select Region", options=categories['region'])
df['region'] = selected_region

selected_region_code = st.selectbox("Select Region Code", options=categories['region_code'])
df['region_code'] = selected_region_code

selected_district_code = st.selectbox("Select District Code", options=categories['district_code'])
df['district_code'] = selected_district_code

selected_lga = st.selectbox("Select LGA", options=categories['lga'])
df['lga'] = selected_lga


selected_population = st.number_input("Enter population", min_value=0, max_value=30500, step=1)
st.caption("Range is from 0 to 30500 meters. Enter the population value arround the well.")
df['population'] = selected_population



selected_public_meeting = st.selectbox("Public Meeting", options=categories['public_meeting'])
df['public_meeting'] = selected_public_meeting


selected_permit = st.selectbox("Permit", options=categories['permit'])
df['permit'] = selected_permit



selected_construction_year  = st.number_input("Enter construction year", min_value=0, max_value=2013, step=1)
st.caption("0 will be replaced with median of nonzero values of the training data: Enter the Year the waterpoint was constructed.")
df['construction_year'] = selected_construction_year


selected_extraction_type = st.selectbox("Extraction Type", options=categories['extraction_type'])
df['extraction_type'] = selected_extraction_type

selected_management_group = st.selectbox("Management Group", options=categories['management_group'])
df['management_group'] = selected_management_group


selected_payment = st.selectbox("Payment", options=categories['payment'])
df['payment'] = selected_payment

selected_water_quality = st.selectbox("Water Quality", options=categories['water_quality'])
df['water_quality'] = selected_water_quality

selected_quantity = st.selectbox("Quantity", options=categories['quantity'])
df['quantity'] = selected_quantity

selected_source_type = st.selectbox("Source Type", options=categories['source_type'])
df['source_type'] = selected_source_type

selected_source_class = st.selectbox("Source Class", options=categories['source_class'])
df['source_class'] = selected_source_class

selected_waterpoint_type = st.selectbox("Waterpoint Type", options=categories['waterpoint_type'])
df['waterpoint_type'] = selected_waterpoint_type





# Show the DataFrame
st.write("Sample DataFrame to pass to the pipeline for pump functionality prediction:")
st.dataframe(df.style.format({'longitude': '{:.6f}'}))