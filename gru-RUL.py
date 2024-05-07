# -*- coding: utf-8 -*-
"""
@author: PRINCELY 
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import Adam

# Function to get the creation time of a file
def get_file_creation_time(file_path):
    return os.path.getctime(file_path)

# Function to extract files from a folder path
def extract_files(folder_path):
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return excel_files

# Iterate over each file path and alter the cycle index with addition of more files
def process_files(folder_path, sheet_names_to_import):
    file_paths = extract_files(folder_path)
    file_paths.sort(key=get_file_creation_time)
    dfs = []
    last_cycle_index = 0
    for file_path in file_paths:
        # Get sheet names from the Excel file
        sheet_names = pd.ExcelFile(file_path).sheet_names
        for sheet_name in sheet_names:
            if sheet_name in sheet_names_to_import:
                # Read the Excel file with the current sheet name
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df['Cycle_Index'] += last_cycle_index
                last_cycle_index = df['Cycle_Index'].max()
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Function to Calculate the State of health of the battery
def calculate_soh(df, c_rate):
    # Convert 'Test_Time(s)' to hours
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    # Identify the start of each discharge cycle
    discharge_start = (df['Current(A)'] < 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_discharge_start_time = df.loc[discharge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_discharge_start_time_series = df['Cycle_Index'].map(first_discharge_start_time)
    # Calculate test time discharged for each row
    df['Test_Time_Discharged(h)'] = df['Test_Time(h)'] - first_discharge_start_time_series
    # Calculate battery capacity
    df['Battery_Capacity_Per_Point(mAh)'] = df['Test_Time_Discharged(h)'] * abs(df['Current(A)']) * 1000
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].transform('max')
    max_soh = df['Battery_Capacity(mAh)'].max()
    # Calculate the SOH values by dividing each battery capacity by the maximum value and multiplying by 100
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_soh) * 100
    df['C_rate'] = c_rate
    # Identify the start of each charge cycle
    charge_start = (df['Current(A)'] >= 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_charge_start_time = df.loc[charge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_charge_start_time_series = df['Cycle_Index'].map(first_charge_start_time)
    # Calculate test time charged for each row
    df['Test_Time_charged(h)'] = df['Test_Time(h)'] - first_charge_start_time_series
    return df

# Function to plot the SOH against the number of cycles
def plot_soh(df):
    SOH = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].max().tolist()
    max_soh = max(SOH)
    Actual_soh = [(value / max_soh) * 100 for value in SOH]
    x = df['Cycle_Index'].unique()
    y = Actual_soh
    plt.xlabel('Cycles')
    plt.ylabel('Battery State of Health (%)')
    plt.plot(x, y)
    plt.show()

# Function to create an initial heatmap correlation of all variables
def plot_heatmap(df, discharge_boundary):
    discharge_data = df[df['Current(A)'] < -(discharge_boundary)]
    features = ['Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                'Internal_Resistance(Ohm)', 'Test_Time_Discharged(h)', 'Calculated_SOH(%)', 'Test_Time_charged(h)']#, 'C_rate']
    X = pd.get_dummies(discharge_data[features], drop_first=True)
    corr_matrix = X.corr()
    corr_target = corr_matrix[['Calculated_SOH(%)']].drop(labels=['Calculated_SOH(%)'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_target, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 14})
    plt.show()
    plt.close()

# Function to rank feature importance using the gini impurity and visualize in a bar chart 
def rank_feature_importance(df):
    # Define features and labels
    features = ['Internal_Resistance(Ohm)', 'Voltage(V)', 'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Cycle_Index']
    labels = df['Calculated_SOH(%)']    
    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(df[features], labels, test_size=0.3, random_state=42)    
    # Initialize Decision Tree regressor
    regressor = DecisionTreeRegressor(random_state=42)    
    # Fit the regressor
    regressor.fit(features_train, labels_train)    
    # Get feature importances
    feature_importances = regressor.feature_importances_    
    # Sort feature importances in descending order
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_names = df[features].columns[sorted_indices]
    sorted_importances = feature_importances[sorted_indices]    
    # Create a bar plot of feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_feature_names, palette='viridis')
    plt.xlabel('Feature Importance (Gini impurity)')
    plt.ylabel('Features')
    plt.title('Feature Importance Ranking')
    plt.show()

# Function for selecting valid features
def selection(df):
    # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
    df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0

    # Select specific columns
    df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 'Battery_Capacity(mAh)',
             'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)']]

    # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
    df = df.groupby('Cycle_Index').agg({
        'Internal_Resistance(Ohm)': 'max',
        'Voltage(V)': 'mean',
        'Test_Time_Discharged(h)': 'max',
        'Test_Time_charged(h)': lambda x: x.max() - x.min(), # Corrected code for the calculation
        'Calculated_SOH(%)' : 'max',
        'Battery_Capacity(mAh)' : 'max'
    }).reset_index()    
    # Dropping rows with missing or NaN values
    df.dropna(inplace=True)
    return df

# Preprocessing functions
def preprocess_features(X_train, X_test):
    numerical_features = X_train.select_dtypes(include=['float32'])
    numerical_columns = numerical_features.columns
    ct = ColumnTransformer([('only numeric', StandardScaler(), numerical_columns)], remainder='passthrough')
    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, c='blue', label='Predictions')
    plt.xlabel('True SOH (%)')
    plt.ylabel('Predicted SOH (%)')
    plt.title('Predictions vs. True SOH')
    plt.legend()
    plt.show()
    mse, mae = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)

# Process the data
folder_paths = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38"]

sheet_names = [["Channel_1-008"], ["Channel_1-006"], ["Channel_1-007"], ["Channel_1-010"], ["Channel_1-011"]]

datasets = []
for i in range(len(folder_paths)):
    df = process_files(folder_paths[i], sheet_names[i])
    df = calculate_soh(df, 1 if i < 2 else 0.5)
    df = selection(df)
    df = df[df['Calculated_SOH(%)'] >= 70]
    df = df[df['Test_Time_charged(h)'] >= 2]
    df['RUL'] = (df['Calculated_SOH(%)'] - 70) * (100 / 30)
    datasets.append(df)

folder_path_2 = "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36"
sheet_name_2 = ["Channel_1-009"]
CS2_36 = process_files(folder_path_2, sheet_name_2)
CS2_36 = calculate_soh(CS2_36, 1)
CS2_36_mod = CS2_36[CS2_36['Current(A)'] < -0.5]
CS2_36 = selection(CS2_36)
CS2_36 = CS2_36[CS2_36['Calculated_SOH(%)'] >= 70]
CS2_36 = CS2_36[CS2_36['Test_Time_charged(h)'] >= 2]
CS2_36['RUL'] = (CS2_36['Calculated_SOH(%)'] - 70) * (100 / 30)

# Merge datasets and split into train and test sets
numerical_columns = ['Cycle_Index', 'Battery_Capacity(mAh)']

X_train = pd.concat(datasets).loc[:, numerical_columns].values
y_train = pd.concat(datasets).loc[:, 'RUL'].values.reshape(-1, 1)

X_test = CS2_36.loc[:, numerical_columns].values
y_test = CS2_36.loc[:, 'RUL'].values.reshape(-1, 1)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_shape = X_train.shape[1]
model = Sequential([
        GRU(units=128, return_sequences=True, input_shape=(input_shape, 1)),
        GRU(units=128, return_sequences=False),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary() 
model.fit(X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200, batch_size=100, verbose=1, validation_data=(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)) 
mse, mae = model.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1), y_test)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
predictions = model.predict(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1))
plt.figure(figsize=(15, 8))
plt.xlabel('Cycle')
plt.ylabel('RUL')
plt.plot(predictions, label='Prediction')
plt.plot(y_test, label='Actual')
plt.legend()