# -*- coding: utf-8 -*-
"""
@author: PRINCELY
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Reshape, ZeroPadding1D
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Function to plot the SOH against the number of cycles (Used in preliminary analysis)
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

# Function to create a initial heatmap correlation of all variables
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
    
# Function for selecting valid points and features from each dataset
def selection(df):
    # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
    df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
    # Select specific columns
    df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
             'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)']]
    
    # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
    df = df.groupby('Cycle_Index').agg({
        'Internal_Resistance(Ohm)': 'max',
        'Voltage(V)': 'mean',
        'Test_Time_Discharged(h)': 'max',
        'Test_Time_charged(h)': lambda x: x.max() - x.min(), # Corrected code for the calculation
        'Calculated_SOH(%)' : 'max'
    }).reset_index()    
    # Dropping rows with missing or NaN values
    df.dropna(inplace=True)
    return df

# Scalling features for prediction
def preprocess_features(X_train, X_test):
    numerical_features = X_train.select_dtypes(include=['float32'])
    numerical_columns = numerical_features.columns
    ct = ColumnTransformer([('only numeric', StandardScaler(), numerical_columns)], remainder='passthrough')
    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to optimize data types and reduce memory usage
def optimize_data_types(df):
    df['Cycle_Index'] = df['Cycle_Index'].astype('int32')
    df['Voltage(V)'] = df['Voltage(V)'].astype('float32')
    df['Internal_Resistance(Ohm)'] = df['Internal_Resistance(Ohm)'].astype('float32')
    df['Test_Time_Discharged(h)'] = df['Test_Time_Discharged(h)'].astype('float32')
    df['Test_Time_charged(h)'] = df['Test_Time_charged(h)'].astype('float32')
    df['Calculated_SOH(%)'] = df['Calculated_SOH(%)'].astype('float32')
    return df

# Function to train feedforward neural network model
def create_fnn_model(X_train_scaled, y_train):
    # Reshape the input data to include the time dimension
    input_shape = X_train.shape[1]
    # Initialize a sequential model
    model = Sequential([
    Dense(128, input_shape=(input_shape,), activation='relu'),
    Dense(1)  # Output layer with 1 neuron for regression
    ])
    model.summary()
    opt = Adam(learning_rate=0.005)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    model.fit(X_train_scaled, y_train, epochs=400, batch_size=50, verbose=1)
    return model

# Function to train deep neural network model
def create_dnn_model(X_train_scaled, y_train):
    # Reshape the input data to include the time dimension
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    # Initialize a sequential model
    model = Sequential()
    model.add(Conv1D(16, 3, activation='relu',padding='same', input_shape=(X_train_scaled.shape[1], 1)))
    model.add(ZeroPadding1D(padding=1))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(8, 3, activation='relu',padding='same'))
    model.add(ZeroPadding1D(padding=1))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(8, 3, activation='relu',padding='same'))
    model.add(ZeroPadding1D(padding=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(140, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1))
    model.summary()
    opt = Adam(learning_rate=0.005)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    model.fit(X_train_scaled, y_train, epochs=400, batch_size=50, verbose=1)
    return model

# 0.06 MSE, loss = 0.1617 , 0.1902 MAE
# Function to train GRU_RNN model
def create_gru_model(X_train_scaled, y_train):
    input_shape = (X_train_scaled.shape[1], 1)
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    model = Sequential([
        GRU(units=128, return_sequences=True, input_shape=input_shape),
        GRU(units=128, return_sequences=False),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary() 
    model.fit(X_train_scaled, y_train, epochs=300, batch_size=50, verbose=1, validation_data=(X_test_scaled, y_test)) 
    return model

# MSE of 0.0398, 0.1566 MAE
# Function to train LSTM neural network model
def create_lstm_model(X_train_scaled, y_train):
    input_shape = (X_train_scaled.shape[1], 1)
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary() 
    model.fit(X_train_scaled, y_train, epochs=400, batch_size=50, verbose=1, validation_data=(X_test_scaled, y_test)) 
    return model

def create_gpr_model(X_train, X_test, y_train, y_test):
    '''Gaussian Process Model Initiation'''
    kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
               1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
               1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                     length_scale_bounds=(0.1, 10.0),
                                     periodicity_bounds=(1.0, 10.0)),
               ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
               1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]
    gp = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b',  n_restarts_optimizer=0, normalize_y=True, copy_X_train=False)
    pipeline = Pipeline(
      [
                ('scl', StandardScaler()),
                ('clf', gp)
            ]
        )
    param = {"clf__kernel": kernels,
             "clf__alpha": np.round(np.random.uniform(-0.0001, 30, 1000), 2),
            }
    model = RandomizedSearchCV(pipeline, param_distributions=param, cv=10, n_iter=3, verbose=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    
# Function to evaluate NN
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
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38"]

sheet_names = [["Channel_1-008"], ["Channel_1-009"], ["Channel_1-006"], ["Channel_1-007"], ["Channel_1-010"], ["Channel_1-011"]]

datasets = []
for i in range(len(folder_paths)):
    df = process_files(folder_paths[i], sheet_names[i])
    df = calculate_soh(df, 1 if i < 2 else 0.5)
    df = selection(df)
    df = df[df['Calculated_SOH(%)'] >= 70]
    df = df[df['Test_Time_charged(h)'] >= 2]
    # df['RUL'] = (df['Calculated_SOH(%)'] - 70) * (100 / 30)
    datasets.append(df)
    
numerical_columns = ['Cycle_Index', 'Test_Time_Discharged(h)', 'Test_Time_charged(h)']
combined_df = pd.concat(datasets, ignore_index=True)
combined_df = optimize_data_types(combined_df)
X = combined_df[['Cycle_Index', 'Test_Time_Discharged(h)', 'Test_Time_charged(h)']]
y = combined_df['Calculated_SOH(%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
model = create_dnn_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)

# FOR GPR
#create_gpr_model(X_train_scaled, X_test_scaled, y_train, y_test)
# train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)