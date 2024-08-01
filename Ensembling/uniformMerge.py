import os
import pandas as pd


directory_path = './mergeResults'

combined_df = pd.DataFrame()
id_order = None
# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, dtype={'Id': str, 'Prediction': float})
        # Store the order of Ids from the first file
        if id_order is None:
            id_order = df['Id']
        # Append the current file's data to the combined DataFrame
        combined_df = pd.concat([combined_df, df])

# Group by 'Id' and calculate the mean 'Prediction' for each 'Id'

result_df = combined_df.groupby('Id', as_index=False)['Prediction'].mean()
result_df['Prediction'] = result_df['Prediction'].round(7)
# Reorder the final DataFrame to match the order of Ids in the first file
result_df = result_df.set_index('Id').loc[id_order].reset_index()
# Save the result to a new CSV file
result_df.to_csv('ensemble_predictions.csv', index=False)

print("Ensemble predictions saved to 'ensemble_predictions.csv'.")