import pandas as pd

# Read the first CSV file
df1 = pd.read_csv('./mergeResult/results5.csv')
df2 = pd.read_csv('./mergeResult/results_nerualCF3.csv')
df3 = pd.read_csv('./mergeResult/results_stefano.csv')
df4 = pd.read_csv('./mergeResult/result_stefano98204.csv')

df5 = pd.read_csv('./mergeResult/results_nerualCF4.csv')
df6 = pd.read_csv('./mergeResult/result_last_stefano.csv')
df7 = pd.read_csv('./mergeResult/result_svd.csv')
df8 = pd.read_csv('./mergeResult/results_DeeperNCF.csv')
df9 = pd.read_csv('./mergeResult/results_DeeperNCF2.csv')

df10 = pd.read_csv('./mergeResult/results_DeeperNCFOverfitt.csv')
#df11 = pd.read_csv('./mergeResult/resultst.csv')

# Merge the dataframes on the 'Id' column
merged_df = pd.merge(df1, df2, on='Id', suffixes=('_1', '_2'))
merged_df = pd.merge(merged_df, df3, on='Id', suffixes=('', '_3'))
merged_df = pd.merge(merged_df, df4, on='Id', suffixes=('', '_4'))

merged_df = pd.merge(merged_df, df5, on='Id', suffixes=('', '_5'))
merged_df = pd.merge(merged_df, df6, on='Id', suffixes=('', '_6'))
merged_df = pd.merge(merged_df, df7, on='Id', suffixes=('', '_7'))
merged_df = pd.merge(merged_df, df8, on='Id', suffixes=('', '_8'))
merged_df = pd.merge(merged_df, df9, on='Id', suffixes=('', '_9'))
merged_df = pd.merge(merged_df, df10, on='Id', suffixes=('', '_10'))
#merged_df = pd.merge(merged_df, df11, on='Id', suffixes=('', '_11'))

# Calculate the average of the 'Prediction' columns
merged_df['Prediction'] = (merged_df['Prediction_1'] + merged_df['Prediction_2'] + merged_df['Prediction'] +
                           merged_df['Prediction_4'] + merged_df['Prediction_5'] + merged_df['Prediction_6'] +
                           merged_df['Prediction_7'] + merged_df['Prediction_8'] + merged_df['Prediction_9'] +
                           merged_df['Prediction_10']) / 10


# Select the relevant columns
result_df = merged_df[['Id', 'Prediction']]
print("ASDFASDF")
# Write the result to a new CSV file
result_df.to_csv('merged_10OVerfitPredictions.csv', index=False)