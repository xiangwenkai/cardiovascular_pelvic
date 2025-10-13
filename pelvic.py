import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def is_pelvic(x):
    if x == 1.:
        return 1.
    elif x == 2.:
        return 0.
    else:
        return None

def vaginal(x):
    if x['RHQ166'] > 0:
        return 1
    else:
        return 0

def caesarean(x):
    if x['RHQ169'] > 0:
        return 1
    else:
        return 0


def age_group(x):
    group = []
    if x is None:
        return None
    elif x < 50:
        return '<50'
    else:
        return '>=50'


def race_group(x):
    if x is None:
        return None
    elif x == 1.:
        return 'Mexican American'
    elif x == 2.:
        return 'Other Hispanic'
    elif x == 3.:
        return 'Non-Hispanic White'
    elif x == 4.:
        return 'Non-Hispanic Black'
    else:
        return 'Other Race'

def edu_group(x):
    if x is None:
        return None
    elif x in [1., 2.]:
        return 'Lower than high school'
    elif x == 3.:
        return 'High school'
    elif x in [4., 5.]:
        return 'Some college or above'

def income_group(x):
    if x is None:
        return None
    elif x < 1.5:
        return '<1.5'
    elif x < 5.:
        return '1.5-4.99'
    elif x >= 5.:
        return '>=5'

def parity_group(x):
    if x is None:
        return None
    elif x < 1:
        return '<1'
    elif x <= 2:
        return '1-2'
    elif x >= 3:
        return '>=3'


def sdx1(x):
    if x['SXQ251'] in [7., 9.]:
        return 0
    else:
        return x['SXQ251']


def sdx2(x):
    if x['SXQ753'] == 1. or x['SXQ260'] == 1. or x['SXQ265'] == 1. or x['SXQ270'] == 1. or x['SXQ272'] == 1.:
        return 1
    else:
        return 0

if __name__ == "__main__":
    '''
    • KIQ042 - Leak urine during physical activities?
    • KIQ046 - Leak urine during nonphysical activities
    '''
    fix_map = {'2013-2014': '_H', '2015-2016': '_I', '2017-2020': '_P', '2021-2023': '_L'}
    years = ['2013-2014', '2015-2016', '2017-2020', '2021-2023']
    fs = ['RHQ_H.XPT', 'RHQ_I.XPT', 'P_RHQ.XPT', 'RHQ_L.XPT']
    kind = 'questionnaire'
    for year, f in zip(years, fs):
        try:
            # print(df_cvh.shape[0])
            print(df_y.shape[0])
            cvh_append = pd.read_pickle(f'processed/{year}/df_cvh.pkl')
            y_append = pd.read_sas(f'y/{year}/{kind}/{f}')
            y_append['year'] = str(year)
            df_cvh = pd.concat([df_cvh, cvh_append])
            df_y = pd.concat([df_y, y_append])
        except:
            df_cvh = pd.read_pickle(f'processed/{year}/df_cvh.pkl')
            df_y = pd.read_sas(f'y/{year}/{kind}/{f}')
            df_y['year'] = str(year)

    cols = ['RHQ078']
    for col in cols:
        df_y[col] = df_y[col].apply(lambda x: None if x in [7., 9.] else x)
    #
    df_y['pelvic'] = df_y['RHQ078'].apply(lambda x: is_pelvic(x))
    df_y = df_y[['SEQN', 'pelvic', 'year']].dropna()

    df_y = pd.merge(df_y, df_cvh[['SEQN', 'CVH', 'DASH', 'PA', 'SMQ', 'SLQ', 'BMI', 'CHO', 'LBX', 'BP']], on='SEQN', how='inner')
    df_y['CVH'] = df_y['CVH'].apply(lambda x: round(x, 2))
    df_y['CVH_group'] = pd.qcut(df_y['CVH'], q=4, duplicates='drop')
    # df_y_col[col].value_counts()


    # sex； age； pregnant
    for year in years:
        try:
            print(df_demo.shape[0])
            demo_append = pd.read_sas(f'y/{year}/raw/DEMO{fix_map[year]}.XPT')
            demo_append = demo_append[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDEXPRG', 'SDMVPSU','SDMVSTRA']]
            df_demo = pd.concat([df_demo, demo_append])
        except:
            df_demo = pd.read_sas(f'y/{year}/raw/DEMO{fix_map[year]}.XPT')
            df_demo = df_demo[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDEXPRG', 'SDMVPSU','SDMVSTRA']]

    df_demo = df_demo[df_demo['RIAGENDR'] == 2.]
    df_demo = df_demo[df_demo['RIDAGEYR'] >= 20.]
    df_demo = df_demo[df_demo['RIDEXPRG'] != 1.]
    # df_y1 = pd.merge(df_demo, df_cvh, on='SEQN', how='inner')
    # df_y2 = pd.merge(df_y, df_y1, on='SEQN', how='inner')
    # df_y2['pelvic'].value_counts()
    df_y = pd.merge(df_y, df_demo, on='SEQN', how='inner')
    df_y = df_y.rename(columns={'RIDAGEYR': 'AGE'})

    del df_demo
    # Race; Education leve; family income
    for year in years:
        try:
            print(df_demo.shape[0])
            demo_append = pd.read_sas(f'y/{year}/raw/DEMO{fix_map[year]}.XPT')
            if year == '2017-2020':
                demo_append = demo_append[['SEQN', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'WTINTPRP', 'WTMECPRP']]
                demo_append.columns = ['SEQN', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'WTINT2YR', 'WTMEC2YR']
            else:
                demo_append = demo_append[['SEQN', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'WTINT2YR', 'WTMEC2YR']]
            df_demo = pd.concat([df_demo, demo_append])
        except:
            df_demo = pd.read_sas(f'y/{year}/raw/DEMO{fix_map[year]}.XPT')
            df_demo = df_demo[['SEQN', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'WTINT2YR', 'WTMEC2YR']]
    df_demo['DMDEDUC2'] = df_demo['DMDEDUC2'].apply(lambda x: None if x in [7., 9.] else x)
    df_y = pd.merge(df_y, df_demo, on='SEQN', how='left')
    df_y = df_y.rename(columns={'RIDRETH1': 'Race', 'DMDEDUC2': 'Educational level', 'INDFMPIR': 'Family income'})

    # add SDX 2013-2016
    for year in ['2013-2014', '2015-2016']:
        try:
            sxq_append = pd.read_sas(f'y/{year}/raw/SXQ{fix_map[year]}.XPT')
            sxq_append = sxq_append[['SEQN', 'SXD031', 'SXQ251', 'SXQ753', 'SXQ260', 'SXQ265', 'SXQ270', 'SXQ272']]
            df_sxq = pd.concat([df_sxq, sxq_append])
        except:
            df_sxq = pd.read_sas(f'y/{year}/raw/SXQ{fix_map[year]}.XPT')
            df_sxq = df_sxq[['SEQN', 'SXD031', 'SXQ251', 'SXQ753', 'SXQ260', 'SXQ265', 'SXQ270', 'SXQ272']]
    df_sxq['SXQ251'] = df_sxq['SXQ251'].fillna(0.)
    df_sxq['SXQ251'] = df_sxq.apply(lambda x: sdx1(x), axis=1)
    df_sxq['STD'] = df_sxq.apply(lambda x: sdx2(x), axis=1)
    df_sxq = df_sxq[['SEQN', 'SXD031', 'SXQ251', 'STD']]
    df_y = pd.merge(df_y, df_sxq, on='SEQN', how='left')

    # add WTDRD1
    for year in years:
        try:
            wtd_append = pd.read_sas(f'y/{year}/raw/DR1TOT{fix_map[year]}.XPT')
            wtd_append = wtd_append[['SEQN', 'WTDRD1']]
            df_wtd = pd.concat([df_wtd, wtd_append])
        except:
            if year == '2017-2020' or year == '2021-2023':
                df_wtd = pd.read_sas(f'y/{year}/raw/DR1TOT{fix_map[year]}.XPT')
                df_wtd = df_wtd[['SEQN', 'WTDRD1PP']]
                df_wtd.columns = ['SEQN', 'WTDRD1']
            else:
                df_wtd = pd.read_sas(f'y/{year}/raw/DR1TOT{fix_map[year]}.XPT')
                df_wtd = df_wtd[['SEQN', 'WTDRD1']]
    df_y = pd.merge(df_y, df_wtd, on='SEQN', how='left')
    # add WTSAF2YR

    # del df_wts
    for year in years:
        try:
            wts_append = pd.read_sas(f'y/{year}/raw/GLU{fix_map[year]}.XPT')
            if year in ['2011-2012', '2017-2020']:
                wts_append = wts_append[['SEQN', 'WTSAFPRP']]
                wts_append.columns = ['SEQN', 'WTSAF2YR']
            else:
                wts_append = wts_append[['SEQN', 'WTSAF2YR']]
            df_wts = pd.concat([df_wts, wts_append])
        except:
            df_wts = pd.read_sas(f'y/{year}/raw/GLU{fix_map[year]}.XPT')
            df_wts = df_wts[['SEQN', 'WTSAF2YR']]
    df_y = pd.merge(df_y, df_wts, on='SEQN', how='left')

    # group
    data = df_y.copy()
    data['AGE_group'] = data['AGE'].apply(age_group)
    data['Race_group'] = data['Race'].apply(race_group)
    data['Educational level group'] = data['Educational level'].apply(edu_group)
    data['Family income group'] = data['Family income'].apply(income_group)

    data.to_excel('/Users/wkx/Documents/data_pelvic.xlsx', index=False)

    data = pd.read_excel('/Users/wkx/Documents/data_pelvic.xlsx')
    cols1 = ['DASH', 'PA', 'SMQ', 'SLQ']
    data['LE1_mean'] = data[cols1].mean(axis=1, skipna=True)
    data['LE1_group'] = pd.qcut(data['LE1_mean'], q=4, labels=[1, 2, 3, 4])
    cols2 = ['BMI', 'CHO', 'LBX', 'BP']
    data['LE2_mean'] = data[cols2].mean(axis=1, skipna=True)
    data['LE2_group'] = pd.qcut(data['LE2_mean'], q=4, labels=[1, 2, 3, 4])
    data['CVH_group'] = pd.qcut(data['CVH'], q=4, labels=[1, 2, 3, 4])
    data.to_excel('/Users/wkx/Documents/data_pelvic1.xlsx', index=False)


    # analysis
    factors = ['Race', 'Educational level']
    factors_group = ['Race_group', 'Educational level group']
    reses = pd.DataFrame(columns=['Variable', 'Overall', 'UL', 'Non-UL'])
    for fea, fea_group in zip(factors, factors_group):
        res = data[['pelvic', fea]].dropna()
        res_head = res.groupby('pelvic').count().reset_index().T[1:]
        res_head.columns = ['Non-UL', 'UL']
        res_head['Overall'] = res[fea].count()
        res_head['Variable'] = fea
        res_head = res_head[['Variable', 'Overall', 'UL', 'Non-UL']]
        reses = pd.concat([reses, res_head], axis=0)

    factors = ['AGE', 'Family income']
    factors_group = ['AGE_group', 'Family income group']
    for fea, fea_group in zip(factors, factors_group):
        res = data[['pelvic', fea]].dropna()
        res_head = res.groupby('pelvic').mean().reset_index().T[1:]
        res_head.columns = ['Non-UL', 'UL']
        res_head['Overall'] = res[fea].mean()
        res_head['Variable'] = fea
        res_head = res_head[['Variable', 'Overall', 'UL', 'Non-UL']]
        reses = pd.concat([reses, res_head], axis=0)

    factors = ['Race', 'Educational level', 'AGE', 'Family income']
    factors_group = ['Race_group', 'Educational level group', 'AGE_group', 'Family income group']
    for fea, fea_group in zip(factors, factors_group):
        res = data[['pelvic', fea, fea_group]].dropna()
        res[fea] = 1
        res_group = pd.pivot_table(res, values=fea, index=fea_group, columns='pelvic', aggfunc=sum).reset_index()
        res_group.columns = ['Variable', 'Non-UL', 'UL']
        res_group['Overall'] = res[[fea, fea_group]].groupby(fea_group).sum().reset_index()[fea]
        res_group = res_group[['Variable', 'Overall', 'UL', 'Non-UL']]
        reses = pd.concat([reses, res_group])

    factors = ['DASH', 'PA', 'SMQ', 'SLQ', 'BMI', 'CHO', 'LBX', 'BP']
    for fea in factors:
        res = data[['pelvic', fea]].dropna()
        res_head = res.groupby('pelvic').mean().reset_index().T[1:]
        res_head.columns = ['Non-UL', 'UL']
        res_head['Overall'] = res[fea].mean()
        res_head['Variable'] = fea
        res_head = res_head[['Variable', 'Overall', 'UL', 'Non-UL']]
        reses = pd.concat([reses, res_head], axis=0)

    reses.to_excel('/Users/wkx/Documents/res_pelvic.xlsx', index=False)


    
