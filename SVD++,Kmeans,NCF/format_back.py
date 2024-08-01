import pandas as pd


def adjust_prediction(prediction):
    if prediction < 1:
        return 1
    elif prediction > 5:
        return 5
    else:
        return prediction
    

# Load the data from the CSV file
input_file = 'predicted_user_ratings.csv'
df = pd.read_csv(input_file)

# Create the new "Id" column by combining user_id and item_id
df['Id'] = 'r' + df['user_id'].astype(str) + '_c' + df['item_id'].astype(str)

# Select the relevant columns for the output file
output_df = df[['Id', 'prediction']]
output_df['prediction'] = output_df['prediction'].apply(adjust_prediction)
# Save the transformed data to a new CSV file
output_file = 'cosine_similarity_users_clamped.csv'
output_df.to_csv(output_file, index=False)

print(f"Transformed predictions saved to '{output_file}'")