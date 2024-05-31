"""
data_pipeline.py

This script processes and merges multiple datasets related to radiographs, radiology reports, 
admissions, and patient information from the MIMIC database. It performs data cleaning, 
filtering, and categorization, then saves the processed data to CSV files for further analysis.

Classes:
    - None

Functions:
    - get_data_directory: Retrieves the absolute path of the data directory.
    - replace_extension: Replaces the file extension from .dcm to .jpg in the file path.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data_directory(relative_path: str) -> str:
    """
    Get the absolute path of the data directory.

    Args:
        relative_path (str): Relative path to the data directory.

    Returns:
        str: Absolute path to the data directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


def replace_extension(file_path: str) -> str:
    """
    Replace the file extension from .dcm to .jpg in the file path.

    Args:
        file_path (str): Path of the file with .dcm extension.

    Returns:
        str: File path with .jpg extension.
    """
    return file_path.replace('.dcm', '.jpg')


# Define the relative path to the data directory
relative_path = '../mimic-data'
DATA_DIR = get_data_directory(relative_path)

# Load all the CSV files
radiograph_data = pd.read_csv(f"{DATA_DIR}/cxr-record-list.csv")
radio_report_data = pd.read_csv(f"{DATA_DIR}/cxr-study-list.csv")
admissions_data = pd.read_csv(f"{DATA_DIR}/admissions.csv")
patients_data = pd.read_csv(f"{DATA_DIR}/patients.csv")

# Filter and clean up admissions data
admissions_data = admissions_data[["subject_id", "race"]].drop_duplicates(keep="first").reset_index(drop=True)

# Merge patients and admissions datasets and clean the merged dataframe
patients_info = admissions_data.merge(patients_data, how='left', on=['subject_id']).drop_duplicates(keep="first")
patients_info = patients_info.rename(columns={"anchor_age": "age"})

# Rename columns in radiograph_data and radio_report_data to avoid confusion while merging
radiograph_data = radiograph_data.rename(columns={"path": "radiograph_path"})
radio_report_data = radio_report_data.rename(columns={"path": "radio_report_path"})

# Merge the radiograph and radiology reports datasets
radiology = radiograph_data.merge(radio_report_data, how='left', on=['subject_id', "study_id"])

# Merge all the datasets to get the final data
final_data = radiology.merge(patients_info, how='left', on=['subject_id']).drop_duplicates(keep="first")

# Categorize the races into 5 race groups
final_data['race'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
final_data['race'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
final_data['race'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
final_data['race'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
final_data['race'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER',
                            'UNKNOWN/NOT SPECIFIED', 'UNKNOWN'], value='OTHER/UNKNOWN', inplace=True)

# Compress the number of ethnicity categories
final_data['race'].loc[~final_data['race'].isin(final_data['race'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN'

# Extract the paths to the radiographs and radiology reports for the data loader
data_loader_final = final_data[["radiograph_path", "radio_report_path", "gender"]]
data_loader_final['radiograph_path'] = data_loader_final['radiograph_path'].apply(replace_extension)
data_loader_final.to_csv(f"{DATA_DIR}/graph_report.csv", index=False)

# Extract the whole data and remove specified columns
columns_to_remove = ['anchor_year', 'anchor_year_group']
final = final_data.drop(columns=columns_to_remove)
final.to_csv(f"{DATA_DIR}/final.csv", index=False)

# View the final data
print(final.head(20))
