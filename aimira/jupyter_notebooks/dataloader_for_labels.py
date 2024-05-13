import pandas as pd

file_fpath = "/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/TE_TreatmentResponse_cleanforJingnan_2.xlsx"
# read the excel file as a dataframe.
df = pd.read_excel(file_fpath)

# extract all rows with 'treatment' equals to 1
df_trt = df[df['treatment'] == 1]

# in the new dataframe, df_trt, select all the rows with VISNUMMER equal to 1
df_trt_bl = df_trt[df_trt['VISNUMMER'] == 1]

# the remaining rows build a new dataframe called 'df_trt_fu'.
df_trt_fu = df_trt[df_trt['VISNUMMER'] != 1]

# In df_trt_fu, remove the rows whose 'TENR' value is not in the df_trt_bl.
df_trt_fu = df_trt_fu[df_trt_fu['TENR'].isin(df_trt_bl['TENR'])]

# use df_trt_fu minus df_trt_bl, get a new dataframe, called df_trt_change.
df_trt_change = df_trt_fu.set_index('TENR') - df_trt_bl.set_index('TENR')

# Update column names
df_trt_change.columns = [str(col) + '_change' for col in df_trt_change.columns]

# concatenate df_trt_change to df_trt_bl to make sure their 'TENR' alligned
result = pd.concat([df_trt_bl.set_index('TENR'), df_trt_change], axis=1).reset_index()


print('yes')