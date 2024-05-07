# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:06:44 2024

@author: PRINCELY
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to create a heatmap correlation graph
def plot_heatmap(df, discharge_boundary):
    discharge_data = df[df['Current(A)'] < -(discharge_boundary)]
    features = ['Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                'Internal_Resistance(Ohm)', 'Test_Time_Discharged(h)', 'Battery_Capacity(mAh)', 'Calculated_SOH(%)', 'C_rate']#, 'C_rate']
    X = pd.get_dummies(discharge_data[features], drop_first=True)
    corr_matrix = X.corr()
    corr_target = corr_matrix[['Calculated_SOH(%)']].drop(labels=['Calculated_SOH(%)'])
    # plt.figure()
    sns.heatmap(corr_target, annot=True, cmap='RdBu_r')
    plt.show()
    plt.close()

folder_path_1 = "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35"
sheet_name_1 = ["Channel_1-008"]
CS2_35 = process_files(folder_path_1, sheet_name_1)
CS2_35 = calculate_soh(CS2_35, 1)
discharge = CS2_35[CS2_35['Current(A)'] < -0.5]
discharge.reset_index(inplace = True)
# plot_soh(CS2_35)
# plot_heatmap(CS2_35, 0.5)

plt.figure(figsize=(10, 6))
plt.plot(discharge['Cycle_Index'], discharge['Internal_Resistance(Ohm)'])
plt.xlabel('Cycles)')
plt.ylabel('Internal_Resistance(Ohm)')
plt.title('Incremental Increase of the Internal Resistance with multiple Discharge Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Get data for specific cycle indices (1st, 20th, 60th, and 200th)
cycle_indices = [1, 20, 60, 200, 300, 400, 500, 800]
cycle_labels = ['1', '20', '60', '200', '300', '400', '500', '800']

plt.figure(figsize=(10, 6))
for cycle_index, label in zip(cycle_indices, cycle_labels):
    cycle_data = discharge[discharge['Cycle_Index'] == cycle_index]
    plt.plot(cycle_data['Test_Time_Discharged(h)'], cycle_data['Voltage(V)'], label=f'Cycle {label}')

plt.xlabel('Test Time(h)')
plt.ylabel('Voltage(h)')
plt.title('Voltage vs Test Time for Specific Cycle Indices in the Discharge Cycle')
plt.legend()
plt.grid(True)
plt.show()

charge = CS2_35[CS2_35['Test_Time_Discharged(h)'] < 0]
discharge.reset_index(inplace = True)

cycle_indice = [1, 20, 60, 200, 300, 400, 500, 800]
cycle_label = ['1', '20', '60', '200', '300', '400', '500', '800']

plt.figure(figsize=(10, 6))
for cycle_index, label in zip(cycle_indice, cycle_label):
    cycle_data = charge[charge['Cycle_Index'] == cycle_index]
    plt.plot(cycle_data['Test_Time_charged(h)'], cycle_data['Voltage(V)'], label=f'Cycle {label}')

plt.xlabel('Test Time(h)')
plt.ylabel('Voltage(h)')
plt.title('Voltage vs Test Time for Specific Cycle Indices in the charge Cycle')
plt.legend()
plt.grid(True)
plt.show()