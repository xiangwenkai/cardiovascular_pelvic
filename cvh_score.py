import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def DASH_score(x):
 s = 0.
 if x['SaturatedFat'] <= 0.06:
  s += 1.
 elif x['SaturatedFat'] <= 0.11:
  s += 0.5

 if x['TotalFat'] <= 0.27:
  s += 1.
 elif x['TotalFat'] <= 0.32:
  s += 0.5

 if x['Protein'] >= 0.18:
  s += 1.
 elif x['Protein'] >= 0.165:
  s += 0.5

 if x['Cholesterol'] <= 71.4:
  s += 1.
 elif x['Cholesterol'] <= 107.1:
  s += 0.5

 if x['Fibre'] >= 14.8:
  s += 1.
 elif x['Fibre'] >= 9.5:
  s += 0.5

 if x['Mg'] >= 238:
  s += 1.
 elif x['Mg'] >= 158:
  s += 0.5

 if x['Ca'] >= 590:
  s += 1.
 elif x['Ca'] >= 402:
  s += 0.5

 if x['K'] >= 2238:
  s += 1.
 elif x['K'] >= 1534:
  s += 0.5

 if x['Na'] <= 1143:
  s += 1.
 elif x['Na'] <= 1286:
  s += 0.5
 return s


def cal_pa(x):
 if x < 1:
  return 0
 elif x < 30:
  return 20
 elif x < 60:
  return 40
 elif x < 90:
  return 60
 elif x < 120:
  return 80
 elif x < 150:
  return 100


def cal_smq(x):
 s = 0
 if year == '2011-2012':
  if x['SMD410'] == 1.:
   s -= 20
 else:
  if x['SMQ876'] == 1.:
   s -= 20
  if x['SMD470'] in [1., 2., 3.]:
   s -= 20
 if abs(x['SMD030']) < 1e-8:
  return 100 + s
 v = x['SMQ050Q']
 q = x['SMQ050U']
 if np.isnan(v) or np.isnan(q) or v is None or q is None:
  return None
 if v > 99.:
  return None
 if q == 1.:
  q /= 365
 elif q == 2.:
  q /= 52
 elif q == 3.:
  q /= 12
 elif q == 4.:
  q = q
 else:
  return None
 y = v * q
 if y >= 5.:
  return 75 + s
 elif y >= 1.:
  return 50 + s
 elif y >= 0.:
  if x['SMQ040'] == 3.:
   return max(0, 25 + s)
  else:
   return 0


def cal_slq(x):
 if x > 24:
  return None
 elif x >= 10:
  return 40
 elif x >= 9:
  return 90
 elif x >= 7:
  return 100
 elif x >= 6:
  return 70
 elif x >= 5:
  return 40
 elif x >= 4:
  return 20
 else:
  return 0


def cal_cho(x):
 s = 0
 if x['BPQ090D'] == 1.:
  s = -20
 lbdhdd = x['LBXTC'] - x['LBDHDD']
 if np.isnan(lbdhdd):
  return None
 if lbdhdd < 130:
  return 100 + s
 elif lbdhdd <= 159:
  return 60 + s
 elif lbdhdd <= 189:
  return 40 + s
 elif lbdhdd <= 219:
  return 20 + s
 else:
  return 0


def cal_bmi(x):
 if np.isnan(x) or x is None:
  return None
 if x < 25:
  return 100
 elif x < 29.9:
  return 70
 elif x < 34.9:
  return 30
 elif x < 39.9:
  return 15
 else:
  return 0


def cal_lbx(x):
 if x['DIQ010'] == 2.:
  if x['LBXGH'] < 5.7 or x['LBXGLU'] < 100:
   return 100
  elif x['LBXGH'] <= 6.4 or x['LBXGLU'] < 125:
   return 60
 else:
  if x['LBXGH'] < 7.:
   return 40
  elif x['LBXGH'] <= 7.9:
   return 30
  elif x['LBXGH'] <= 8.9:
   return 20
  elif x['LBXGH'] <= 9.9:
   return 10
  else:
   return 0


def cal_bp(x, year='2013-2014'):
 s = 0
 if year=='2021-2023':
  if x['BPQ150'] == 1.:
   s -= 20
 else:
  if x['BPQ040A'] == 1.:
   s -= 20
  if x['BPQ050A'] == 1.:
   s -= 20
 sys, dia = [], []
 if year == '2011-2012' or year == '2013-2014' or year == '2015-2016':
  cols1 = ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']
  cols2 = ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']
 elif year == '2017-2020' or year == '2021-2023':
  cols1 = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
  cols2 = ['BPXODI1', 'BPXODI2', 'BPXODI3']
 for i in cols1:
  if np.isnan(x[i]) == False:
   sys.append(x[i])
 for i in cols2:
  if np.isnan(x[i]) == False:
   dia.append(x[i])
 if len(sys) == 0 or len(dia) == 0:
  return None
 sys_mean = sum(sys) / len(sys)
 dia_mean = sum(dia) /len(dia)
 if sys_mean < 120 and dia_mean < 80:
  return 100 + s
 elif sys_mean <= 129 and dia_mean < 80:
  return 75 + s
 elif sys_mean <= 139 and dia_mean <= 89:
  return 50 + s
 elif sys_mean <= 159 and dia_mean <= 99:
  return max(0, 25 + s)
 else:
  return 0

if __name__ == "__main__":
 fix_map = {'2011-2012': '_G', '2013-2014': '_H', '2015-2016': '_I', '2017-2020': '_P', '2021-2023': '_L'}
 for year in ['2011-2012', '2013-2014', '2015-2016', '2017-2020', '2021-2023']:
  # year = '2011-2012'
  # year = '2013-2014'
  # year = '2015-2016'
  # year = '2017-2020'
  # year = '2021-2023'
  if year == '2011-2012':
   data_flag = '_G'
  elif year == '2013-2014':
   data_flag = '_H'
  elif year == '2015-2016':
   data_flag = '_I'
  elif year == '2017-2020':
   data_flag = '_P'
  elif year == '2021-2023':
   data_flag = '_L'
  df = pd.read_sas(f'data/{year}/DR1TOT{data_flag}.XPT')

  total_rm = {'DR1TSFAT': 'Total saturated fatty acids (gm)', 'DR1TTFAT': 'Total fat (gm)',
   'DR1TPROT': 'Protein (gm)', 'DR1TCHOL': 'Cholesterol (mg)', 'DR1TFIBE': 'Dietary fiber (gm)',
   'DR1TMAGN': 'Magnesium (mg)', 'DR1TCALC': 'Calcium (mg)', 'DR1TPOTA': 'Potassium (mg)',
   'DR1TSODI': 'Sodium (mg)', 'DR1TKCAL': 'Energy (kcal)'}

  df_feature = df[['SEQN']+list(total_rm.keys())]
  # df_feature.isnull().sum()

  cols = list(total_rm.keys())[:-1]
  df_feature.dropna(subset=['DR1TSFAT', 'DR1TTFAT', 'DR1TPROT', 'DR1TCHOL','DR1TFIBE', 'DR1TMAGN', 'DR1TCALC', 'DR1TPOTA',
                            'DR1TSODI'], how='all', inplace=True)

  ## DASH ###
  df_feature['SaturatedFat'] = df_feature.apply(lambda x: x['DR1TSFAT']/x['DR1TKCAL']*9., axis=1)
  df_feature['TotalFat'] = df_feature.apply(lambda x: x['DR1TTFAT']/x['DR1TKCAL']*9., axis=1)
  df_feature['Protein'] = df_feature.apply(lambda x: x['DR1TPROT']/x['DR1TKCAL']*4., axis=1)
  df_feature['Cholesterol'] = df_feature.apply(lambda x: x['DR1TCHOL']/x['DR1TKCAL']*1000, axis=1)
  df_feature['Fibre'] = df_feature.apply(lambda x: x['DR1TFIBE']/x['DR1TKCAL']*1000, axis=1)
  df_feature['Mg'] = df_feature.apply(lambda x: x['DR1TMAGN']/x['DR1TKCAL']*1000, axis=1)
  df_feature['Ca'] = df_feature.apply(lambda x: x['DR1TCALC']/x['DR1TKCAL']*1000, axis=1)
  df_feature['K'] = df_feature.apply(lambda x: x['DR1TPOTA']/x['DR1TKCAL']*1000, axis=1)
  df_feature['Na'] = df_feature.apply(lambda x: x['DR1TSODI']/x['DR1TKCAL']*1000, axis=1)

  df_feature['DASH'] = df_feature.apply(lambda x: DASH_score(x), axis=1)
  df_cvh = df_feature[['SEQN', 'DASH']]
  # dfy = pd.read_sas(f'data/{year}/RHQ{data_flag}.XPT')
  # y_cols =['RHQ010', 'RHQ031', 'RHD043', 'RHQ060',
  #        'RHQ074', 'RHQ076', 'RHQ078', 'RHQ131', 'RHQ160', 'RHQ162',
  #        'RHQ166', 'RHQ169', 'RHQ172', 'RHQ171', 'RHD180',
  #        'RHD190', 'RHD280', 'RHQ305',
  #        'RHQ420', 'RHQ540']
  #
  # for col in y_cols:
  #  df_y_col = dfy[['SEQN', col]].dropna()
  #  df_tem = pd.merge(df_feature[['SEQN', 'DASH']], df_y_col[['SEQN', col]], on='SEQN', how='inner')
  #  corr = df_tem[['DASH', col]].corr().iloc[0,1]
  #  print(f"y: {col}; shape: {df_tem.shape[0]}; corr: {corr}")
  #  plt.scatter(df_tem[col], df_tem['DASH'])
  #  plt.show()
  #  plt.close()

  # col = 'RHD180'
  # col = 'RHQ540'
  # df_y_col = dfy[['SEQN', col]].dropna()
  # df_tem = pd.merge(df_feature[['SEQN', 'DASH']], df_y_col[['SEQN', col]], on='SEQN', how='inner')
  # corr = df_tem[['DASH', col]].corr().iloc[0,1]
  # print(f"y: {col}; shape: {df_tem.shape[0]}; corr: {corr}")
  # plt.scatter(df_tem[col], df_tem['DASH'])
  # plt.show()
  # plt.close()


  ## PA ##
  df = pd.read_sas(f'data/{year}/PAQ{data_flag}.XPT')
  if year == '2021-2023':
   df = df.dropna(subset=['PAD790Q', 'PAD790U', 'PAD800', 'PAD810Q', 'PAD810U', 'PAD820'])
  else:
   df = df.dropna(subset=['PAQ610', 'PAD615', 'PAQ625', 'PAD630', 'PAQ655', 'PAD660','PAQ670', 'PAD675'], how='all')

  df['PA_t'] = 0
  def cal_pa_ex(x):
   s = 0
   if x['PAD800'] == 9999.:
    t1 = 0.
   elif x['PAD790U'] == b"D":
    t1 = x['PAD800'] * 7.
   elif x['PAD790U'] == b'M':
    t1 = x['PAD800'] / 4.28
   elif x['PAD790U'] == b'Y':
    t1 = x['PAD800'] / 52
   elif x['PAD790U'] == b'W':
    t1 = x['PAD800']
   else:
    t1 = 0.
   s += t1/x['PAD790Q']
   if x['PAD820'] == 9999.:
    t1 = 0.
   elif x['PAD810U'] == b"D":
    t1 = x['PAD820'] * 7.
   elif x['PAD810U'] == b'M':
    t1 = x['PAD820'] / 4.28
   elif x['PAD810U'] == b'Y':
    t1 = x['PAD820'] / 52
   elif x['PAD810U'] == b'W':
    t1 = x['PAD820']
   else:
    t1 = 0
   s += t1/x['PAD810Q']
   pa_score = cal_pa(s)
   return pa_score

  if year == '2021-2023':
   df['PA'] = df.apply(lambda x: cal_pa_ex(x), axis=1)
  else:
   for x in [['PAQ610', 'PAD615'], ['PAQ625', 'PAD630'], ['PAQ655', 'PAD660'], ['PAQ670', 'PAD675']]:
    x1, x2 = x[0], x[1]
    df[x1] = df[x1].fillna(0)
    df[x2] = df[x2].fillna(0)
    df[x1].loc[df[x1]>=77] = 0
    df[x2].loc[df[x2]>=7777] = 0
    df['PA_t'] += df[x1]*df[x2]
   df['PA'] = df['PA_t'].apply(lambda x: cal_pa(x))

  df_cvh = pd.merge(df_cvh, df[['SEQN', 'PA']], on='SEQN', how='outer')

  # nicotine exposure
  # 1.确认吸没吸过烟，没吸过100分
  # 2.确认当前有没有戒烟，没戒烟直接0分
  df = pd.read_sas(f'data/{year}/SMQ{data_flag}.XPT')
  if year == '2011-2012':
   df_s2 = pd.read_sas(f'data/{year}/SMQFAM{data_flag}.XPT')
   df = pd.merge(df, df_s2[['SEQN', 'SMD410']], on='SEQN', how='left')
  else:
   if year == '2021-2023':
    df_s2 = pd.read_sas(f'data/{year}/SMQFAM{data_flag}.XPT')
    df = pd.merge(df, df_s2[['SEQN', 'SMD470']], on='SEQN', how='left')
   else:
    df_s1 = pd.read_sas(f'data/{year}/SMQSHS{data_flag}.XPT')
    df_s2 = pd.read_sas(f'data/{year}/SMQFAM{data_flag}.XPT')
    df = pd.merge(df, df_s1[['SEQN', 'SMQ876']], on='SEQN', how='left')
    df = pd.merge(df, df_s2[['SEQN', 'SMD470']], on='SEQN', how='left')
  if year == '2021-2023':
   df['SMQ050Q'] = 99.
   df['SMQ050U'] = 99.
   df['SMQ876'] = 99.
   df['SMD030'] = 99.
  df['SMQ'] = df.apply(lambda x: cal_smq(x), axis=1)
  df_cvh = pd.merge(df_cvh, df[['SEQN', 'SMQ']], on='SEQN', how='outer')

  # sleep health
  df = pd.read_sas(f'data/{year}/SLQ{data_flag}.XPT')
  # df['SLD010H'].value_counts()
  if year == '2013-2014' or year == '2011-2012':
   df['SLQ'] = df['SLD010H'].apply(lambda x: cal_slq(x))
  elif year == '2015-2016' or year == '2017-2020':
   df['SLQ'] = df['SLD012'].apply(lambda x: cal_slq(x))
  elif year=='2021-2023':
   df['SLD012'] = df['SLD012']*5/7+df['SLD013']*2/7
   df['SLQ'] = df['SLD012'].apply(lambda x: cal_slq(x))
  df_cvh = pd.merge(df_cvh, df[['SEQN', 'SLQ']], on='SEQN', how='outer')

  # BMI
  df = pd.read_sas(f'data/{year}/BMX{data_flag}.XPT')
  df['BMI'] = df['BMXBMI'].apply(lambda x: cal_bmi(x))

  df_cvh = pd.merge(df_cvh, df[['SEQN', 'BMI']], on='SEQN', how='outer')

  # Cholesterol
  df = pd.read_sas(f'data/{year}/TCHOL{data_flag}.XPT')
  df_s1 = pd.read_sas(f'data/{year}/BPQ{data_flag}.XPT')
  df_s2 = pd.read_sas(f'data/{year}/HDL{data_flag}.XPT')
  if year=='2021-2023':
   df_s2 = df_s2[['SEQN', 'LBDHDD']]
   df_s1 = df_s1[['SEQN', 'BPQ101D']]
   df_s1.columns = ['SEQN', 'BPQ090D']
   df = pd.merge(df, df_s1, on='SEQN', how='left')
   df = pd.merge(df, df_s2[['SEQN', 'LBDHDD']], on='SEQN', how='left')
   df['CHO'] = df.apply(lambda x: cal_cho(x), axis=1)
  else:
   df_s2 = df_s2[['SEQN', 'LBDHDD']]
   df = pd.merge(df, df_s1[['SEQN', 'BPQ090D']], on='SEQN', how='left')
   df = pd.merge(df, df_s2[['SEQN', 'LBDHDD']], on='SEQN', how='left')
   df['CHO'] = df.apply(lambda x: cal_cho(x), axis=1)

  df_cvh = pd.merge(df_cvh, df[['SEQN', 'CHO']], on='SEQN', how='outer')

  # LBX
  # 缺糖尿病史
  df = pd.read_sas(f'data/{year}/GHB{data_flag}.XPT')
  df_s1 = pd.read_sas(f'data/{year}/GLU{data_flag}.XPT')
  df_s2 = pd.read_sas(f'data/{year}/DIQ{data_flag}.XPT')
  df = pd.merge(df, df_s1[['SEQN', 'LBXGLU']], on='SEQN', how='left')
  df = pd.merge(df, df_s2[['SEQN', 'DIQ010']], on='SEQN', how='left')
  df['LBX'] = df.apply(lambda x: cal_lbx(x), axis=1)

  df_cvh = pd.merge(df_cvh, df[['SEQN', 'LBX']], on='SEQN', how='outer')

  # BP
  if year == '2011-2012' or year == '2013-2014' or year == '2015-2016':
   df = pd.read_sas(f'data/{year}/BPX{data_flag}.XPT')
  elif year == '2017-2020' or year == '2021-2023':
   df = pd.read_sas(f'data/{year}/BPXO{data_flag}.XPT')
  df_s1 = pd.read_sas(f'data/{year}/BPQ{data_flag}.XPT')
  if year == '2021-2023':
   df_s1 = df_s1[['SEQN', 'BPQ150']]
  else:
   df_s1 = df_s1[['SEQN', 'BPQ040A', 'BPQ050A']]

  df = pd.merge(df, df_s1, on='SEQN', how='left')
  df['BP'] = df.apply(lambda x: cal_bp(x, year=year), axis=1)
  df_cvh = pd.merge(df_cvh, df[['SEQN', 'BP']], on='SEQN', how='outer')

  thresh = 6
  # df_cvh.dropna(subset=['DASH', 'PA', 'SMQ', 'SLQ', 'BMI', 'CHO', 'LBX', 'BP'], thresh=4, how='all', inplace=True)
  df_cvh.dropna(subset=['DASH', 'PA', 'SMQ', 'SLQ', 'BMI', 'CHO', 'LBX', 'BP'], thresh=thresh, axis=0, inplace=True)
  # df_cvh.isnull().sum()
  quantiles = df_cvh['DASH'].quantile([0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 0.94, 0.95])
  bins = [
   df_cvh['DASH'].min(),  # 0%分位数
   quantiles[0.24],  # 24%分位数
   quantiles[0.49],  # 49%分位数
   quantiles[0.74],  # 74%分位数
   quantiles[0.94],  # 94%分位数
   df_cvh['DASH'].max()  # 100%分位数
  ]
  labels = [0, 25, 50, 80, 100]
  df_cvh['DASH'] = pd.cut(df_cvh['DASH'], bins=bins, labels=labels, include_lowest=True)
  df_cvh['DASH'] = df_cvh['DASH'].astype('Int64')
  print(df_cvh['DASH'].mean())
  df_cvh['CVH'] = df_cvh[['DASH', 'PA', 'SMQ', 'SLQ', 'BMI', 'CHO', 'LBX', 'BP']].mean(axis=1)
  print(f"Year: {year}; Samples that not nan sub score >= {thresh}: {df_cvh.shape[0]}")
  plt.hist(df_cvh['CVH'], bins=20)
  plt.show()

  df_cvh.to_pickle(f'./processed/{year}/df_cvh.pkl')
  print("finished")

