import pandas as pd

# Load cleaned-up ADNI merge spreadsheet
adni_df = pd.read_csv('cleaned_up_adni_merge.csv')

# Load cleaned-up and preprocessed Freesurfer spreadsheet
freesurfer_df = pd.read_csv('cleaned_up_freesurfer.csv')

# Merge the two dataframes on RID (participant ID)
merged_df = pd.merge(adni_df, freesurfer_df, on='RID', how='inner')

# Selecting required columns for final dataframe
selected_columns = ['RID', 'age', 'years_bl', 'sex', 'education'] + bilateral_features

# Creating the final dataframe containing complete information
final_df = merged_df[selected_columns]

# Save the final dataframe to a CSV file
final_df.to_csv('final_dataframe.csv', index=False)
