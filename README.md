# Used Car Price Prediction

## Overview

This project focuses on predicting the price of used cars based on various features such as car specifications, model details, and other attributes. The dataset contains both categorical and numerical features, which are processed and used to train a machine learning model to predict the price of a car.

## Features

The dataset includes the following features:

### Categorical Features:
- **FuelType**: Type of fuel used by the car (e.g., Petrol, Diesel).
- **BodyType**: The body style of the car (e.g., Sedan, Hatchback).
- **Transmission**: Transmission type (e.g., Manual, Automatic).
- **OwnerNo**: Number of previous owners.
- **Oem**: Original Equipment Manufacturer.
- **Model**: Model name of the car.
- **VariantName**: Variant of the car model.
- **InsuranceValidity**: Validity of insurance.
- **RTO**: Regional Transport Office code.
- **Seats**: Number of seats in the car.
- **WheelSize**: The size of the wheels.
- **City**: The city where the car is located.

### Numerical Features:
- **Kilometers**: Number of kilometers the car has run.
- **ModelYear**: Year the car model was manufactured.
- **CentralVariantId**: Identifier for the car variant.
- **EngineDisplacement**: The engine displacement (in cc).
- **Mileage**: The mileage of the car (in km per liter).
- **Torque**: Torque of the car engine.
- **Price**: The price of the used car (target variable).

## Data Preprocessing

The data undergoes several preprocessing steps, including:

1. **Handling Missing Data**: Missing values are imputed using appropriate strategies.
2. **Encoding Categorical Variables**: Categorical features are encoded using methods like One-Hot Encoding or Label Encoding.
3. **Feature Scaling**: Numerical features are scaled to ensure uniformity and to improve model performance.
4. **Outlier Treatment**: Outliers are identified and handled to improve model accuracy.

## Machine Learning Model

The project uses a **Linear Regression** model (or another chosen model, depending on your implementation) to predict the price of used cars. The model is trained using the processed data, and its performance is evaluated using metrics such as **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared**.
