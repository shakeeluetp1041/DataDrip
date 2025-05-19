import streamlit as st
import numpy as np
import datetime
import pandas as pd
import joblib
import json

st.title("DataDrip: Water Pumps Functionality Prediction")
#st.subheader("A Streamlit App for Monitoring and Predicting Pump Failures")
#st.markdown("This project uses machine learning to predict pump maintenance needs based on input parameters.")

# Load preprocessor pipeline for input column processing to make them compatible to pass through the model
preprocessor = joblib.load('preprocessor.joblib')
# Load models
dt_model = joblib.load('DecisionTree_model.joblib')
rf_model = joblib.load('RandomForest_model.joblib')
xgb_model = joblib.load('XGBoost_model.joblib')

# Model mapping
model_dict = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# Sidebar main title (larger and bold)
st.sidebar.title("User Input Pannel")

# Sidebar title: model selection
st.sidebar.subheader("Step 01: Select the Model")
# Sidebar: Select model
selected_model_name = st.sidebar.selectbox('Select a model', list(model_dict.keys()))
selected_model = model_dict[selected_model_name]
st.sidebar.subheader("Step 02: Select Input Features")

# Set default to today's date
default_date = datetime.date.today()
# Date input widget with calendar popup
selected_date = st.sidebar.date_input("Select a date:", value=default_date) # after preprpocessing, the yeaer will be extracted
data = {'date_recorded': [selected_date]}
df = pd.DataFrame(data)


selected_gps_height = st.sidebar.number_input("Enter GPS Height (in meters)", min_value=-90, max_value=2770, step=1)
st.sidebar.caption("Range is from -90 to 2770 meters. Enter the Altitude.")
df['gps_height'] = selected_gps_height 

selected_longitude = st.sidebar.number_input("Enter Longitude", min_value=0.0,   max_value=40.0, step=0.000001,  format="%.6f" )  # display format to 6 decimal places
st.sidebar.caption("Range is from 0 to 40. Enter the longitude where the pump is located.")
df['longitude'] = selected_longitude 

selected_latitude = st.sidebar.number_input("Enter Latitude", min_value=-11.0,   max_value=0.0, step=0.000001,  format="%.6f" )  # display format to 6 decimal places
st.sidebar.caption("Range is from -11 to 0. Enter the latitude where the pump is located.")
df['latitude'] = selected_latitude

# Dealing with categorical features
with open("categories.json", "r") as f:  # Load the JSON file (saved categories)
    categories = json.load(f)

# Streamlit inputs for each categorical variable
selected_basin = st.sidebar.selectbox("Select Basin", options=categories['basin'])
df['basin'] = selected_basin

selected_region = st.sidebar.selectbox("Select Region", options=categories['region'])
df['region'] = selected_region

selected_region_code = st.sidebar.selectbox("Select Region Code", options=categories['region_code'])
df['region_code'] = selected_region_code

selected_district_code = st.sidebar.selectbox("Select District Code", options=categories['district_code'])
df['district_code'] = selected_district_code

selected_lga = st.sidebar.selectbox("Select LGA", options=categories['lga'])
df['lga'] = selected_lga


selected_population = st.sidebar.number_input("Enter population", min_value=0, max_value=30500, step=1)
st.sidebar.caption("Range is from 0 to 30500 meters. Enter the population value arround the well.")
df['population'] = selected_population



selected_public_meeting = st.sidebar.selectbox("Public Meeting", options=categories['public_meeting'])
df['public_meeting'] = selected_public_meeting


selected_permit = st.sidebar.selectbox("Permit", options=categories['permit'])
df['permit'] = selected_permit



selected_construction_year  = st.sidebar.number_input("Enter construction year", min_value=0, max_value=2013, step=1)
st.sidebar.caption("0 will be replaced with median of nonzero values of the training data: Enter the Year the waterpoint was constructed.")
df['construction_year'] = selected_construction_year


selected_extraction_type = st.sidebar.selectbox("Extraction Type", options=categories['extraction_type'])
df['extraction_type'] = selected_extraction_type

selected_management_group = st.sidebar.selectbox("Management Group", options=categories['management_group'])
df['management_group'] = selected_management_group


selected_payment = st.sidebar.selectbox("Payment", options=categories['payment'])
df['payment'] = selected_payment

selected_water_quality = st.sidebar.selectbox("Water Quality", options=categories['water_quality'])
df['water_quality'] = selected_water_quality

selected_quantity = st.sidebar.selectbox("Quantity", options=categories['quantity'])
df['quantity'] = selected_quantity

selected_source_type = st.sidebar.selectbox("Source Type", options=categories['source_type'])
df['source_type'] = selected_source_type

selected_source_class = st.sidebar.selectbox("Source Class", options=categories['source_class'])
df['source_class'] = selected_source_class

selected_waterpoint_type = st.sidebar.selectbox("Waterpoint Type", options=categories['waterpoint_type'])
df['waterpoint_type'] = selected_waterpoint_type


# Show the DataFrame
st.write("User Input Sample for pump functionality prediction:")
st.dataframe(df.style.format({'longitude': '{:.6f}'}))


# Mapping from label to integer
target_map_dict = {
    'functional': 2,
    'functional needs repair': 1,
    'non functional': 0
}

# Inverse mapping: from integer back to label
inv_target_map_dict = {v: k for k, v in target_map_dict.items()}

# Make Prediction
st.title("Scenario 1: Predicting Pump Status for New Data (User Input)")
# Prediction button
if st.button("Click to Predict"):
    # Preprocess first
    transformed_input = preprocessor.transform(df)
    # Then predict
    prediction = selected_model.predict(transformed_input)[0]
    # Convert numeric prediction to readable label
    label_prediction = inv_target_map_dict.get(prediction, "Unknown")
    # Show result
    st.success(f"Predicted Pump Status: {label_prediction} (Code: {prediction})")

st.title("Scenario 2: Model Testing with Known Labels (Accuracy Evaluation)")
# Load test data
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
# Set number of samples to show
num_samples = st.slider("Select number of random test samples", min_value=1, max_value=2376, value=2376)
# Random sample
random_indices = np.random.choice(len(X_test), size=num_samples, replace=False)
X_sample = X_test.iloc[random_indices]
y_sample = y_test.iloc[random_indices]
# Preprocess and predict
X_transformed = preprocessor.transform(X_sample)
predictions = selected_model.predict(X_transformed)
# Prepare display data
results = pd.DataFrame({
    'Index': random_indices,
    'True Label': [inv_target_map_dict.get(label, "Unknown") for label in y_sample],
    'Predicted Label': [inv_target_map_dict.get(pred, "Unknown") for pred in predictions]
})
# Show results
#st.subheader("Prediction Results (Index, True Label, Predicted Label)")
#st.dataframe(results)
# Compute accuracy
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_sample, predictions)
st.subheader("Test Accuracy on Selected Samples")
st.write(f"Accuracy: {test_accuracy:.4f}")
