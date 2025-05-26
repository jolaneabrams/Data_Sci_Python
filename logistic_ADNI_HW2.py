import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

if __name__ == '__main__':
    # read data
    df_raw = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    # define features
    df_raw['Age_visit'] = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns={'PTGENDER': 'Sex', 'PTEDUCAT': 'Education_Years'}, inplace=True)
    #df_raw.ICV = df_raw.ICV / 1e3
    df_raw.Hippocampus = df_raw.Hippocampus / 1e3
    df_raw = df_raw.loc[df_raw.FLDSTRENG == '3 Tesla MRI']

    # select relevant columns
    measures_we_care_about = ['RID', 'VISCODE', 'Age_visit', 'Years_bl', 'Sex', 'Education_Years', 'DX', 'Hippocampus']
    df = df_raw[measures_we_care_about].dropna()

    #Pt 1
    # create binary variables for sex and diagnosis
    df['Sex'] = (df['Sex'] == 'Female').astype(int)
    df['is_CN'] = (df['DX'] == 'CN').astype(int)
    df['is_AD'] = (df['DX'] == 'Dementia').astype(int)


    # filter out MCIs
    df = df[(df['DX'] == 'CN') | (df['DX'] == 'Dementia')]

    # define design matrix
    X = sm.add_constant(df[['Age_visit', 'Sex', 'Education_Years', 'Hippocampus']])
    y = df['is_AD']

    # do logistic regression
    model = sm.Logit(y, X)
    results = model.fit()

    # Print summary table
    print(results.summary(xname=['Intercept', 'Age at visit', 'Sex', 'Ed.Years', 'Hippocampus']))

    #Pt 2 - the odds

    #odds = exp(x_hipp * beta_hipp)--> beta_hipp is hippo coef from previous logistic regression = -2.2172; x_hipp is mean HC vol = 7.5 cc
   # define x_hipp values for 1% and 5% volume losses
    x_hipp = 7.5
    x_hipp_1 = 7.5 -(7.5 * 0.01)
    x_hipp_5 = 7.5 - (7.5 * 0.05)

    #baseline odds at average CN hippocampal volume
    baseline_odds = np.exp(7.5 * -2.2172)
    print('Average CN hippocampal volume is associated with a ' + str(baseline_odds/100) + ('% chance of AD.'))

    # odds for 1% loss and increase in odds percentage
    odds_1_increase = (np.exp((x_hipp_1 * -2.2172) - baseline_odds) / baseline_odds) * 100
    print('A decrease of 1% of average CN hippocampal volume accounts for an increase in the odds of AD of:',
          odds_1_increase, (' %'))

    # calculate odds for 5% loss and compute the increase in odds percentage
    odds_5_increase = (np.exp((x_hipp_5 * -2.2172) - baseline_odds) / baseline_odds) * 100
    print('A decrease of 5% of average CN hippocampal volume accounts for an increase in the odds of AD of:',
          odds_5_increase, (' %'))


#Pt3 - Simpler regression

    # define design matrix
    X_design_matrix = np.c_[np.ones((df.shape[0], 1)), df.Hippocampus]
    y = df['is_AD']

    # do logistic regression
    model = sm.Logit(y, X_design_matrix)
    results = model.fit()

    # make x values for HC volume
    x_line = np.linspace(df.Hippocampus.min(), df.Hippocampus.max(), 100)

    # make y values using logistic function
    logistic_function = lambda x: 1 / (1 + np.exp(-(results.params[0] + results.params[1] * x)))
    y_line = logistic_function(x_line)

    #plot logistic function based on estimated parameters
    plt.figure(1)
    plt.plot(df.Hippocampus[y == 0], y[y == 0], '.b', label='CN (y=0)')
    plt.plot(df.Hippocampus[y == 1], y[y == 1], '.r', label='AD (y=1)')
    plt.plot(x_line, y_line, color=('#ADD8E6'))
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Hippocampus Volume (cm3)', fontsize=12)
    plt.ylabel('P(AD|hipp. vol.)', fontsize=12)

    # Set the ticks on the x-axis starting from 3
    plt.xticks(np.arange(3, df.Hippocampus.max() + 1, 1))

    plt.show()

    print('done.')








    # just intercept, hippocampus

