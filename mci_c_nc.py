import pandas as pd
import numpy as np


#FILL IN THIS FUNCTION THAT CALCULATES MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
#THEN PRINTS OUT THE RATE OF CONVERSION/NON-CONVERSION/REGRESSION AS A PERCENTAGE
def calc_mci_rates(df):

    pass

if __name__ == '__main__':

    # needs: pip install xlrd==1.2.0
    df_raw                  = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    #keep the measures we care about, drop missing data and sort by RID/Years_bl
    measures_we_care_about  = ['RID', 'Years_bl', 'DX']
    df                      = df_raw[measures_we_care_about]
    df                      = df.dropna(subset=measures_we_care_about)

    # Define dataframe of baseline MCI diagnoses:
    df_bl_MCI               = df.loc[(df.DX=='MCI') & (df.Years_bl==0)]
    RID_MCI_bl              = np.unique(df_bl_MCI.RID)

    #ADD COMMENT:
    df                      = df.loc[np.in1d(df.RID, RID_MCI_bl)]
    df                      = df.sort_values(by = ['RID','Years_bl'])

    #ADD COMMENT:
    df_2year                = df.loc[df.Years_bl >= 2]
    df_2year                = df_2year.sort_values(by = ['RID','Years_bl'])
    df_2year                = df_2year.drop_duplicates(subset=['RID'])

    #ADD FUNCTION TO FIND MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
    #AND THEN CALCULATE RATES OF MCI CONVERTION TO DEMENTIA, NON-CONVERSION (STAYING AS MCI) AND REGRESSION FROM MCI TO CN
    calc_mci_rates(df_2year)

    #NOW FIND 4-YEAR CONVERSION/NON-CONV./REGRESSION RATES USING SAME FUNCTION (calc_MCI_rates)
