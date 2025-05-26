import pandas as pd
import numpy as np

#FILL IN THIS FUNCTION THAT CALCULATES MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
#THEN PRINTS OUT THE RATE OF CONVERSION/NON-CONVERSION/REGRESSION AS A PERCENTAGE

 # Count how many MCI values for conversion, non-conversion, regression
def calc_MCI_rates(df):
    # Count how many MCI values for conversion, non-conversion, regression
    total_count = df.shape[0]
    changed_to_AD = np.sum(df['DX'] == 'Dementia')
    stayed_same = np.sum(df['DX'] == 'MCI')
    changed_to_CN = np.sum(df['DX'] == 'CN')

     # Calculate percentages
    percentage_Dem = (changed_to_AD / total_count) * 100
    percentage_same = (stayed_same / total_count) * 100
    percentage_CN = (changed_to_CN / total_count) * 100

    print('Conversion rate: %5.1f%%' % (percentage_Dem))
    print('Non-conversion rate: %5.1f%%' % (percentage_same))
    print('Regression rate: %5.1f%%' % (percentage_CN))

if __name__ == '__main__':

    # needs: pip install xlrd==1.2.0
    df_raw = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    #keep the measures we care about, drop missing data and sort by RID/Years_bl
    measures_we_care_about  = ['RID', 'Years_bl', 'DX']
    df                      = df_raw[measures_we_care_about]
    df                      = df.dropna(subset=measures_we_care_about)

    # Filter rows in dataframe where diagnosis is MCI and years since baseline is 0, then extract unique RIDs for those rows

    df_bl_MCI               = df.loc[(df.DX=='MCI') & (df.Years_bl==0)]
    RID_MCI_bl              = np.unique(df_bl_MCI.RID)

    # Filter df to include only rows where RID present in RID_MCI_bl, sort first by RID & then years since baseline
    df                      = df.loc[np.in1d(df.RID, RID_MCI_bl)]
    df                      = df.sort_values(by = ['RID','Years_bl'])

    #Filter df to include only rows where years since baseline is >=2, sort by subject ID & years since baseline, then remove duplicate RIDs
    df_2year                = df.loc[df.Years_bl >= 2]
    df_2year                = df_2year.sort_values(by = ['RID','Years_bl'])
    df_2year                = df_2year.drop_duplicates(subset=['RID'])

    #ADD FUNCTION TO FIND MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
    #AND THEN CALCULATE RATES OF MCI CONVERTION TO DEMENTIA, NON-CONVERSION (STAYING AS MCI) AND REGRESSION FROM MCI TO CN
    calc_MCI_rates(df_2year)

    #NOW FIND 4-YEAR CONVERSION/NON-CONV./REGRESSION RATES USING SAME FUNCTION (calc_MCI_rates)

    # Filter df to include only rows where years since baseline is >=4, sort by subject ID & years since baseline, then remove duplicate RIDs
    df_4year = df.loc[df.Years_bl >= 4]
    df_4year = df_4year.sort_values(by=['RID', 'Years_bl'])
    df_4year = df_4year.drop_duplicates(subset=['RID'])

    calc_MCI_rates(df_4year)