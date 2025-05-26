#1 - who's going to convert based on baseline and brain age gap, can get some idea of who will get worse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

features_total = ['bankssts_total_volume', 'caudalanteriorcingulate_total_volume',
            'caudalmiddlefrontal_total_volume', 'cuneus_total_volume',
            'entorhinal_total_volume', 'fusiform_total_volume',
            'inferiorparietal_total_volume', 'inferiortemporal_total_volume',
            'isthmuscingulate_total_volume', 'lateraloccipital_total_volume',
            'lateralorbitofrontal_total_volume', 'lingual_total_volume',
            'medialorbitofrontal_total_volume', 'middletemporal_total_volume',
            'parahippocampal_total_volume', 'paracentral_total_volume',
            'parsopercularis_total_volume', 'parsorbitalis_total_volume',
            'parstriangularis_total_volume', 'pericalcarine_total_volume',
            'postcentral_total_volume', 'posteriorcingulate_total_volume',
            'precentral_total_volume', 'precuneus_total_volume',
            'rostralanteriorcingulate_total_volume', 'rostralmiddlefrontal_total_volume',
            'superiorfrontal_total_volume', 'superiorparietal_total_volume',
            'superiortemporal_total_volume', 'supramarginal_total_volume',
            'frontalpole_total_volume', 'temporalpole_total_volume',
            'transversetemporal_total_volume', 'insula_total_volume']

# 1b - define function to compare OLS-based and ridge-regression-based brain age prediction models
def compare_OLS_ridge(dataframe, features):

    # split data
    train, rest = train_test_split(dataframe, train_size=0.5, random_state=96)
    val, test = train_test_split(rest, test_size=0.5, random_state=96)

    # separate predictors & target
    X_train = train[features]
    y_train = train['age_bl']
    X_val = val[features]
    y_val = val['age_bl']
    X_test = test[features]
    y_test = test['age_bl']

    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # fit OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_test_pred_ols = ols.predict(X_test_scaled)
    rmse_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))
    print("OLS RMSE on test data:", rmse_ols)

    # tune lambda for Ridge regression with validation set
    lambdas = np.logspace(-6, 6, 13)
    best_lambda = None
    best_rmse = np.inf

    for lambda_ in lambdas:
        ridge = Ridge(alpha=lambda_)
        ridge.fit(X_train_scaled, y_train)
        y_val_pred_ridge = ridge.predict(X_val_scaled)
        rmse_ridge = np.sqrt(mean_squared_error(y_val, y_val_pred_ridge))

        if rmse_ridge < best_rmse:
            best_rmse = rmse_ridge
            best_lambda = lambda_

    # print best lambda & validation RMSE
    print("Best ridge model is lambda =", best_lambda, "with RMSE validation error of", best_rmse, "years.")

    # evaluate best Ridge model on test data
    best_ridge = Ridge(alpha=best_lambda)
    best_ridge.fit(X_train_scaled, y_train)
    y_test_pred_ridge = best_ridge.predict(X_test_scaled)
    rmse_ridge_test = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
    print("Ridge RMSE on test data:", rmse_ridge_test)

    # output the comparison
    print(f"Brain age model comparison: OLS out-of-sample RMSE: {rmse_ols} years, Ridge out-of-sample RMSE: {rmse_ridge_test} years.")

def brain_age_vs_chron(dataframe, features_total, diagnosis_column='DX', age_column='age_bl', model=None, scaler=None):

    # filter diagnostic groups
    cn_subjects = dataframe[dataframe[diagnosis_column] == 'CN']
    mci_ad_subjects = dataframe[dataframe[diagnosis_column].isin(['MCI', 'Dementia'])]

    # split CN subjects into training and test sets
    cn_train, cn_test = train_test_split(cn_subjects, train_size=0.75, random_state=96)

    # prepare test (combine test CN with MCI & Dementia)
    test_data = pd.concat([cn_test, mci_ad_subjects])

    # separate predictors and target for training and testing
    X_train = cn_train[features_total]
    y_train = cn_train[age_column]
    X_test = test_data[features_total]
    y_test = test_data[age_column]

    # standardize features - can either train new model or use existing one
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # fit OLS model if not provided
    if model is None:
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

    # predict & evaluate
    y_pred_test = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return model, X_train_scaled, X_test_scaled, y_test, y_train, scaler, test_data

def brain_age_vs_chron_nonCN(dataframe, features, model=None, scaler=None):

    # filter diagnostic groups
    mci_ad_subjects = freemerge_df[freemerge_df['DX'].isin(['MCI', 'Dementia'])]

    # separate predictors and target
    X = mci_ad_subjects[features]
    y = mci_ad_subjects['age_bl']

    # standardize features - can either train new model or use existing one
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # fit model if not provided
    if model is None:
        raise ValueError("A pre-trained model must be provided for this function.")

    # predict
    y_pred = model.predict(X_scaled)

    return X_scaled, y_pred


if __name__ == '__main__':

    # load Freesurfer regions
    mri_FS_df = pd.read_excel('ADNI_Freesurfer.xlsx')

    regions = [
        'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal',
        'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital',
        'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal',
        'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
        'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
        'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
        'supramarginal', 'frontalpole', 'temporalpole', 'transversetemporal', 'insula'
    ]

    for region in regions:
        left_column = f'lh_{region}_volume'
        right_column = f'rh_{region}_volume'

        if left_column in mri_FS_df.columns and right_column in mri_FS_df.columns:
            mri_FS_df[f'{region}_total_volume'] = mri_FS_df[left_column] + mri_FS_df[right_column]
        else:
            print(f"Missing columns for region: {region}")

 # define measures we care about
    measures_we_care_about = ['bankssts_total_volume', 'caudalanteriorcingulate_total_volume',
                              'caudalmiddlefrontal_total_volume', 'cuneus_total_volume',
                              'entorhinal_total_volume', 'fusiform_total_volume',
                              'inferiorparietal_total_volume', 'inferiortemporal_total_volume',
                              'isthmuscingulate_total_volume', 'lateraloccipital_total_volume',
                              'lateralorbitofrontal_total_volume', 'lingual_total_volume',
                              'medialorbitofrontal_total_volume', 'middletemporal_total_volume',
                              'parahippocampal_total_volume', 'paracentral_total_volume',
                              'parsopercularis_total_volume', 'parsorbitalis_total_volume',
                              'parstriangularis_total_volume', 'pericalcarine_total_volume',
                              'postcentral_total_volume', 'posteriorcingulate_total_volume',
                              'precentral_total_volume', 'precuneus_total_volume',
                              'rostralanteriorcingulate_total_volume', 'rostralmiddlefrontal_total_volume',
                              'superiorfrontal_total_volume', 'superiorparietal_total_volume',
                              'superiortemporal_total_volume', 'supramarginal_total_volume',
                              'frontalpole_total_volume', 'temporalpole_total_volume',
                              'transversetemporal_total_volume', 'insula_total_volume'
                              ]

# load ADNIMERGE_final_proj.xlsx into new dataframe
demog_df = pd.read_excel('ADNIMERGE_final_proj.xlsx')

# renaming columns in demog_df
columns_to_rename = {
        'AGE_bl': 'age_bl',
        'PTEDUCAT': 'edu',
        'Years_bl': 'years_bl',
        'PTGENDER': 'sex',
    }
# rename columns in demog_df
demog_df = demog_df.rename(columns=columns_to_rename)

# drop NAs from FreeSurfer data
mri_FS_df = mri_FS_df.dropna(subset=measures_we_care_about)

# drop NAs from ADNIMERGE data
demog_df = demog_df.dropna(subset=['IMAGEUID', 'RID', 'age_bl', 'edu', 'years_bl', 'sex', 'DX'])

# merge mri_FS_df and demog_df at IMAGEUID
freemerge_df                      = pd.merge(demog_df, mri_FS_df, how='inner', on='IMAGEUID')

# drop RID_x column
freemerge_df.drop(columns=['RID_x'], inplace=True)

# rename RID_y column to RID
freemerge_df.rename(columns={'RID_y': 'RID'}, inplace=True)

# filter to include only CN subjects
cn_freemerge_df = freemerge_df[freemerge_df['DX'] == 'CN']

# sort by 'RID' and 'years_bl' preparing to grab first visit
cn_freemerge_df = cn_freemerge_df.sort_values(by=['RID', 'years_bl'])

# group by 'RID' and take first occurrence
first_visit_df = cn_freemerge_df.groupby('RID').first().reset_index()

compare_OLS_ridge(first_visit_df, features_total)


# 1c i

# filter freemerge_df to get all DX at 1st visit

print(freemerge_df.columns)

# sort by 'RID" and 'years_bl' in prep to grab 1st visit
all_DX_df = freemerge_df.sort_values(by=['RID', 'years_bl'])

# group by 'RID', take first occurrence
all_first_visit_df = all_DX_df.groupby ('RID').first().reset_index()

#drop duplicates from all_first_visit_df
all_first_visit_df = all_first_visit_df.drop_duplicates(subset='RID')

brain_age_vs_chron(all_first_visit_df, features_total)

# capture values from function
model, X_train_scaled, X_test_scaled, y_test, y_train, scaler, test_data = brain_age_vs_chron(all_first_visit_df, features_total)

# make plot of estimated brain age vs chronological age for training set

# predict brain age for training set subjects
y_train_pred = model.predict(X_train_scaled)

# make scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)  # Line for perfect predictions
plt.xlabel('Actual Age (years)')
plt.ylabel('Predicted Brain Age (years)')
plt.title('Predicted brain age vs. actual age in training set')
plt.show()

# 1c ii make boxplot + scatter plot of brain age gap across diagnostic groups in test subjects

# predict brain age for test set subjects
y_test_pred = model.predict(X_test_scaled)

# compute brain age gap
brain_age_gap = y_test_pred - y_test

# grouping by diagnostic groups for plotting
groups = test_data.groupby('DX')

test_data['brain_age_gap'] = brain_age_gap

# change 'DX' from 'Dementia' to 'AD'
test_data['DX'] = test_data['DX'].replace('Dementia', 'AD')

# define order of DX categories
diag_order = ['CN', 'MCI', 'AD']

# boxplot of brain age gap across diagnostic groups
plt.figure(figsize=(8, 6))

# create boxplot
ax = sns.boxplot(x='DX', y='brain_age_gap', data=test_data, order = diag_order,
            boxprops={'facecolor':'None', 'edgecolor': 'black'},  # Unfilled boxes with gray edges
            medianprops={'color': 'orange', 'linewidth': 2})  # Orange median line

# overlay scatter plot
sns.stripplot(x='DX', y='brain_age_gap', data=test_data, order = diag_order, palette='bright', size=4, jitter=True, alpha=0.7)

# remove x-axis and y-axis labels
ax.set_xlabel('')
ax.set_ylabel('')

plt.title('Boxplots of brain age gap across diagnostic groups')
plt.show()

# 1d i: compute brain age gap and make boxplot to show age difference gap between 2+ year MCI converters vs.
# non-converters. Use the earliest 2+ year visit for each subject.

features = [
    'lh_bankssts_volume', 'lh_caudalanteriorcingulate_volume', 'lh_caudalmiddlefrontal_volume',
    'lh_cuneus_volume', 'lh_entorhinal_volume', 'lh_fusiform_volume', 'lh_inferiorparietal_volume',
    'lh_inferiortemporal_volume', 'lh_isthmuscingulate_volume', 'lh_lateraloccipital_volume',
    'lh_lateralorbitofrontal_volume', 'lh_lingual_volume', 'lh_medialorbitofrontal_volume',
    'lh_middletemporal_volume', 'lh_parahippocampal_volume', 'lh_paracentral_volume',
    'lh_parsopercularis_volume', 'lh_parsorbitalis_volume', 'lh_parstriangularis_volume',
    'lh_pericalcarine_volume', 'lh_postcentral_volume', 'lh_posteriorcingulate_volume', 'lh_precentral_volume',
    'lh_precuneus_volume', 'lh_rostralanteriorcingulate_volume', 'lh_rostralmiddlefrontal_volume',
    'lh_superiorfrontal_volume', 'lh_superiorparietal_volume', 'lh_superiortemporal_volume',
    'lh_supramarginal_volume', 'lh_frontalpole_volume', 'lh_temporalpole_volume', 'lh_transversetemporal_volume',
    'lh_insula_volume', 'rh_bankssts_volume', 'rh_caudalanteriorcingulate_volume', 'rh_caudalmiddlefrontal_volume',
    'rh_cuneus_volume', 'rh_entorhinal_volume', 'rh_fusiform_volume', 'rh_inferiorparietal_volume',
    'rh_inferiortemporal_volume', 'rh_isthmuscingulate_volume', 'rh_lateraloccipital_volume',
    'rh_lateralorbitofrontal_volume', 'rh_lingual_volume', 'rh_medialorbitofrontal_volume', 'rh_middletemporal_volume',
    'rh_parahippocampal_volume', 'rh_paracentral_volume', 'rh_parsopercularis_volume', 'rh_parsorbitalis_volume',
    'rh_parstriangularis_volume', 'rh_pericalcarine_volume', 'rh_postcentral_volume', 'rh_posteriorcingulate_volume',
    'rh_precentral_volume', 'rh_precuneus_volume', 'rh_rostralanteriorcingulate_volume', 'rh_rostralmiddlefrontal_volume',
    'rh_superiorfrontal_volume', 'rh_superiorparietal_volume', 'rh_superiortemporal_volume', 'rh_supramarginal_volume',
    'rh_frontalpole_volume', 'rh_temporalpole_volume', 'rh_transversetemporal_volume', 'rh_insula_volume'
]

# update the model for new features
model, X_train_scaled, X_test_scaled, y_test, y_pred_test, scaler, test_data = brain_age_vs_chron(all_first_visit_df, features)

# make df of MCI subjects at baseline
mci_baseline = freemerge_df[(freemerge_df['DX'] == 'MCI') & (freemerge_df['years_bl'] == 0)].drop_duplicates(subset='RID')

# extract features for mci_baseline
mci_baseline_features = mci_baseline[features]

# scale the features
X_Train_Scaled = scaler.transform(mci_baseline_features)

# filter for those with follow-up visit after 2 years
mci_followup = freemerge_df[freemerge_df['years_bl'] >= 2].drop_duplicates(subset='RID', keep='first').reset_index(drop=True)

# get indices of MCIs
mci_indices = mci_followup[mci_followup['DX'] == 'MCI'].index

# filter the follow-up data using the MCI indices
mci_followup_data = mci_followup.loc[mci_indices].copy()  # Make a copy to avoid modifying the original DataFrame

# add suffix '_baseline' to 'age_bl' and '_followup' to 'years_bl'
mci_followup_data = mci_followup_data.rename(columns={'age_bl': 'age_bl_baseline', 'years_bl': 'years_bl_followup'})

# reset mci_baseline index
mci_baseline_reset = mci_baseline.reset_index()

# merge using index as regular column
mci_1_followup_data_dx = mci_followup_data['RID','DX']

mci_1_followup_important = mci_baseline_reset.merge(mci_1_followup_data_dx, on='RID')

# drop extra 'index' column
mci_1_followup_important.drop(columns=['index'], inplace=True)

# sort merged dataframe by RID and follow-up visit years_bl
mci_1_followup_sorted = mci_1_followup_important.sort_values(by=['RID_baseline', 'years_bl_followup'])

# call nonCN function with necessary arguments
X_scaled, y_pred = brain_age_vs_chron_nonCN(mci_1_followup_sorted, features, model=model, scaler=scaler)

# get age at baseline and follow-up for each subject
age_bl_plus_years_bl = np.array(mci_1_followup_sorted['age_bl_baseline'] + mci_1_followup_sorted['years_bl_followup'])

# get predicted brain age gap for MCIs
predictions_years = model.predict(X_Train_Scaled)

# calculate brain age gap
brain_age_gap_2 = predictions_years - mci_baseline['age_bl']

# extract diagnoses at baseline and follow-up
baseline_diag = mci_1_followup_sorted['DX_baseline']
visit_after_2_years_diag = mci_1_followup_sorted[mci_1_followup_sorted['years_bl_followup'] == 2]['DX_followup']

# make dataframe to store diagnoses and brain age gap
diag_age_gap_df = pd.DataFrame({'DX_baseline': baseline_diag, 'DX_followup': visit_after_2_years_diag, 'Brain age gap': brain_age_gap_2})

# categorize into converters and non-converters
converters = diag_age_gap_df[diag_age_gap_df['DX_followup'] == 'AD']
non_converters = diag_age_gap_df[diag_age_gap_df['DX_followup'] == 'MCI']

# set plot size
plt.figure(figsize=(8, 6))

# make boxplot
ax = sns.boxplot(x='DX_followup', y='brain_age_gap_baseline', data=non_converters,
                 boxprops={'facecolor': 'None', 'edgecolor': 'black'},
                 medianprops={'color': 'orange', 'linewidth': 2})

# define colors for converters & non-converters
color_converters = 'orange'
color_non_converters = 'blue'

# overlay scatter plot for converters
sns.stripplot(x='DX_followup', y='brain_age_gap_baseline', data=converters, color=color_converters,
              marker='o', size=4, jitter=True, alpha=0.7, ax=ax)

# overlay scatter plot for non-converters
sns.stripplot(x='DX_followup', y='brain_age_gap_baseline', data=non_converters, color=color_non_converters,
              marker='o', size=4, jitter=True, alpha=0.7, ax=ax)

# remove x-axis and y-axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# set plot title
plt.title('Boxplots of baseline brain age gap for 2+ year MCI non-conv./conv.')

plt.show()





