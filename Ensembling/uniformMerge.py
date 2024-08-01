import os
import pandas as pd


directory_path = './mergeRes2'

combined_df = pd.DataFrame()
id_order = None

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, dtype={'Id': str, 'Prediction': float})
        # Store the order of Ids for later reordering again
        if id_order is None:
            id_order = df['Id']
        combined_df = pd.concat([combined_df, df])

# Group by 'Id' and calculate the mean 'Prediction' for each 'Id'

result_df = combined_df.groupby('Id', as_index=False)['Prediction'].mean()
result_df['Prediction'] = result_df['Prediction']
# need to reorder again as groupby probably mixes it
result_df = result_df.set_index('Id').loc[id_order].reset_index()
result_df.to_csv('ensemble_predictions_last_weighted.csv', index=False)

print("Ensemble predictions saved to 'ensemble_predictions.csv'.")