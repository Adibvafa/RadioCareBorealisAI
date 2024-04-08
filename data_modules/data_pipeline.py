import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# To do: Change the path based on your directory where data is stored

# Loaded all the csv files
radiograph_data = pd.read_csv("data/cxr-record-list.csv")
radio_report_data = pd.read_csv("data/cxr-study-list.csv")
admissions_data = pd.read_csv("data/admissions.csv")
patients_data = pd.read_csv("data/patients.csv")
# label_mimic_data = pd.read_csv("data/mimic_chexpert.csv")

# Filtered and cleaned up admissions data
admissions_data = admissions_data[["subject_id","race"]].drop_duplicates(keep="first").reset_index(drop=True)

# Filtered the mimic label to only include only one label - "Pleural Effusion"
# label_mimic_data = label_mimic_data[["subject_id","study_id","Pleural Effusion"]]

# Merged patients and admissions datasets and cleaned the merged dataframe
patients_info = admissions_data.merge(patients_data, how='left', on=['subject_id']).drop_duplicates(keep="first")
patients_info = patients_info.rename(columns={"anchor_age": "age"})

# Renamed the columns in radiograph_data and radio_report_data to avoid confusion while merging
radiograph_data = radiograph_data.rename(columns={"path": "radiograph_path"})
radio_report_data = radio_report_data.rename(columns={"path": "radio_report_path"})

# Merged the radiograph and radiology reports datasets  
radiology = radiograph_data.merge(radio_report_data, how='left', on=['subject_id',"study_id"])

# Merged all the datasets to get the final data
final_data = radiology.merge(patients_info, how='left', on=['subject_id']).drop_duplicates(keep="first")

# final_data = final_df.merge(label_mimic_data, how='left', on=['subject_id',"study_id"]).drop_duplicates(keep="first")


# Categorised the races into 5 race groups
# Compress the number of ethnicity categories
final_data['race'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
final_data['race'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
final_data['race'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
final_data['race'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
final_data['race'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 
                         'UNKNOWN/NOT SPECIFIED','UNKNOWN'], value='OTHER/UNKNOWN', inplace=True)

#take into consideration just the top-5 categories with biggest value_count, the others will fall into OTHER category
final_data['race'].loc[~final_data['race'].isin(final_data['race'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN' 

# print(final_data.head(20))


# Extracted the paths to the radiographs and radiology reports for the data loader
data_loader_final = final_data[["radiograph_path","radio_report_path","gender"]]

def replace_extension(file_path):
    return file_path.replace('.dcm', '.jpg')

data_loader_final['radiograph_path'] = data_loader_final['radiograph_path'].apply(replace_extension)

data_loader_final.to_csv("graph_report.csv")
# print(data_loader_final.head(20))


# Extracted the whole data and removed 2 columns 
columns_to_remove = ['anchor_year', 'anchor_year_group']
final = final_data.drop(columns=columns_to_remove)
final.to_csv("final.csv")

# To view the final data
print(final.head(20))

