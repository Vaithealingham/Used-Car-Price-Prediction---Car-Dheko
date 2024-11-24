import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,PowerTransformer

current_dir = os.path.dirname(__file__)

csv_path_1 = os.path.join(current_dir,'car_dheko_cleaned_data.csv')
csv_path_2 = os.path.join(current_dir, 'Data_for_Model_Building.csv')

# Load the trained model
model = joblib.load("best_XG_boosting_model.joblib")

data = pd.read_csv(csv_path_1)
processed_data = pd.read_csv(csv_path_2)

# Streamlit App Layout
st.title("Car Dekho Used Car Price Prediction")

# Dropdown and input fields for each feature
st.sidebar.title("Select the Features")

fuel_type = ['All'] + data['FuelType'].sort_values().unique().tolist()
selected_fuel_type = st.sidebar.selectbox("Fuel Type", fuel_type)

# Filter data based on selected fuel type
filtered_data = data if selected_fuel_type == 'All' else data[data['FuelType'] == selected_fuel_type]

transmission = ['All'] + filtered_data['Transmission'].sort_values().unique().tolist()
selected_transmission = st.sidebar.selectbox("Transmission", transmission)

filtered_data = data if selected_transmission == 'All' else filtered_data[filtered_data['Transmission'] == selected_transmission]

body_type = ['All'] + filtered_data['BodyType'].sort_values().unique().tolist()
selected_body_type = st.sidebar.selectbox("Body Type", body_type)

# Further filter data based on selected body type
filtered_data = filtered_data if selected_body_type == 'All' else filtered_data[filtered_data['BodyType'] == selected_body_type]

oems = ['All'] + filtered_data['Oem'].sort_values().unique().tolist()
selected_Oem = st.sidebar.selectbox("OEM", oems)

# Further filter data based on selected OEM
filtered_data = filtered_data if selected_Oem == 'All' else filtered_data[filtered_data['Oem'] == selected_Oem]

models = ['All'] + filtered_data['Model'].sort_values().unique().tolist()
selected_model = st.sidebar.selectbox("Model", models)

# Further filter data based on selected model
filtered_data = filtered_data if selected_model == 'All' else filtered_data[filtered_data['Model'] == selected_model]

model_years = ['All'] + filtered_data['ModelYear'].sort_values().unique().tolist()
selected_model_year = st.sidebar.selectbox("Model Year", model_years)

engine_ccs = ['All'] + filtered_data['EngineDisplacement'].sort_values().unique().tolist()
engine_displacement = st.sidebar.selectbox("Engine Displacement (cc)", engine_ccs)

torque_s = ['All'] + filtered_data['Torque'].sort_values().unique().tolist()
torque = st.sidebar.selectbox("Torque (Nm)", torque_s)

seats = ['All'] + filtered_data['Seats'].sort_values().unique().tolist()
selected_seats = st.sidebar.selectbox("Seats", seats)

kilometers_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000)

insurance_validity = ['All'] + data['InsuranceValidity'].sort_values().unique().tolist()
selected_insurance_type = st.sidebar.selectbox("Insurance Validity", insurance_validity)

mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=5.0, step=0.5)

owner_no = ['All'] + data['OwnerNo'].sort_values().unique().tolist()
selected_ownerno = st.sidebar.selectbox("Number of Owners", owner_no)

city = ['All'] + data['City'].sort_values().unique().tolist()
selected_city = st.sidebar.selectbox("City",city)

all_columns = processed_data.columns[processed_data.columns != 'Price']

def preprocess_input(df, columns):

    ohe: OneHotEncoder = joblib.load("OneHotEncoder.joblib")
    label_encoder_Oem: LabelEncoder = joblib.load('LabelEncoder_Oem.joblib')
    label_encoder_City: LabelEncoder = joblib.load('LabelEncoder_City.joblib')
    label_encoder_Model: LabelEncoder = joblib.load('LabelEncoder_Model.joblib')
    scaler: StandardScaler = joblib.load('StandardScaler.joblib')

    # df = df.replace("All", np.nan)
    # df['OwnerNo'] = df['OwnerNo'].astype(str)
    # df['Seats'] = df['Seats'].astype(str)

    pt = PowerTransformer(method='yeo-johnson')
    df[['Kilometers', 'EngineDisplacement', 'Torque']] = pt.fit_transform(
        df[['Kilometers', 'EngineDisplacement','Torque']])

    numerical_cols = ['Kilometers', 'EngineDisplacement','Mileage', 'Torque' ]
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    onehot_columns = ['FuelType', 'Transmission', 'BodyType', 'InsuranceValidity']
    # df[onehot_columns] = df[onehot_columns].fillna("unknown").astype(str)
    df[onehot_columns] = df[onehot_columns].astype(str)
    df_ohe = ohe.transform(df[onehot_columns])
    df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out(input_features=onehot_columns))

    df['Oem'] = label_encoder_Oem.transform(df['Oem'])
    df['City'] = label_encoder_City.transform(df['City'])
    df['Model'] = label_encoder_Model.transform(df['Model'])
    
    df_copy = df.drop(columns=onehot_columns).reset_index(drop=True)
    df_ohe = df_ohe.reset_index(drop=True)
    
    df_all_encoded = pd.concat([df_ohe, df_copy], axis=1)

    missing_cols = set(columns) - set(df_all_encoded.columns)
    for col in missing_cols:
        df_all_encoded[col] = 0
    df_all_encoded = df_all_encoded[columns]
    return df_all_encoded


if st.sidebar.button("Estimate Used Car Price"):

    df = pd.DataFrame([[selected_fuel_type, selected_body_type, kilometers_driven, selected_transmission, selected_ownerno,
                                selected_Oem, selected_model, selected_model_year, selected_insurance_type, engine_displacement,
                                mileage, torque, selected_seats, selected_city]],
                              columns=['FuelType', 'BodyType', 'Kilometers', 'Transmission', 'OwnerNo',
                                       'Oem', 'Model', 'ModelYear','InsuranceValidity', 'EngineDisplacement',
                                       'Mileage', 'Torque', 'Seats', 'City'])
    st.write("Car Details Selected for Price Prediction:", df)

    input_data = preprocess_input(df,all_columns)


    predicted_price = model.predict(input_data)[0]
    # predicted_price = abs(predicted_price)

    # # Display predicted price
    def format_price(price):
        if price >= 1_00_00_000:
            return f"₹{price / 1_00_00_000:.2f} Crores"
        elif price >= 1_00_000:
            return f"₹{price / 1_00_000:.2f} Lakhs"
        else:
            return f"₹{price:.2f}"

    formatted_price = format_price(predicted_price)

    st.success(f"The estimated used car price is: {formatted_price}")
