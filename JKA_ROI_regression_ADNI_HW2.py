import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

if __name__ == '__main__':

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw                  = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']     = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns   = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #convert cubic mm to cubic cm
    df_raw.ICV              = df_raw.ICV/1e3
    df_raw.Hippocampus      = df_raw.Hippocampus/1e3
    df_raw.WholeBrain       = df_raw.WholeBrain/1e3

    #retain 3T scans
    df_raw                      = df_raw.loc[df_raw.FLDSTRENG=='3 Tesla MRI']

    # we care about all the basics as before, plus a few Freesurfer-derived ROIs
    measures_we_care_about = ['RID', 'VISCODE', 'Age_visit', 'Years_bl', 'Sex', 'Education_Years', 'DX', 'Hippocampus',
                              'WholeBrain', 'ICV']
    df = df_raw[measures_we_care_about]

    # throw out rows with missing data in any of these fields
    df_1 = df.dropna(subset=measures_we_care_about)

    # sort df by RID and Years_bl then drop duplicates to get first available measure for each subject
    df = df.sort_values(['RID', "Years_bl"])
    df_1 = df_1.drop_duplicates(subset=['RID'])

    # create sex (e.g. 1 for Female, 0 Male), is_not_CN, is_AD coding variables
    sex = (df_1.Sex == 'Female').astype(int)
    is_not_CN = (df_1.DX != 'CN').astype(int)
    is_AD = (df_1.DX == 'Dementia').astype(int)

    # design matrix columns: intercept, age, sex, education, intra-cranial volume (ICV), coding for not CN, coding for AD
    x_train = np.c_[np.ones((df_1.shape[0], 1)), df_1.Age_visit, sex, df_1.Education_Years, df_1.ICV, is_not_CN, is_AD]

    # form the training targets for the Hippocampus and WholeBrain ROI volumes
    y_train_HC = df_1['Hippocampus'].to_numpy().reshape(df_1.shape[0], 1)
    y_train_brain = df_1['WholeBrain'].to_numpy().reshape(df_1.shape[0], 1)

    # build OLS models for Hippocampus and WholeBrain ROIs
    model_HC = sm.OLS(y_train_HC, x_train)
    model_brain = sm.OLS(y_train_brain, x_train)

    # generate results
    results_HC = model_HC.fit()
    results_Brain = model_brain.fit()

    print(results_HC.summary(
        xname=['Intercept', 'Age at visit', 'Sex', 'Ed.Years', 'ICV', '(AD+MCI) vs CN', 'AD vs (MCI+CN)']))
    print(results_Brain.summary(
        xname=['Intercept', 'Age at visit', 'Sex', 'Ed.Years', 'ICV', '(AD+MCI) vs CN', 'AD vs (MCI+CN)']))

    print('done.')
