#https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data_train.csv')

user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

user_item_matrix_filled = user_item_matrix.fillna(0)
item_similarity = cosine_similarity(user_item_matrix_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("comuted simipalirty between items")
# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# Function to get item recommendations
def predict_rating(user_id, item_id, user_item_matrix, similarity_df):
    if item_id not in user_item_matrix.columns or user_id not in user_item_matrix.index:
        return np.nan  # Return NaN if the item or user is not in the matrix
    
    similar_items = similarity_df[item_id].sort_values(ascending=False)
    similar_items = similar_items[similar_items.index != item_id]  # Exclude the item itself
    print(user_id, item_id)
    user_ratings = user_item_matrix.loc[user_id, similar_items.index]
    
    weighted_sum = (similar_items * user_ratings).sum()
    item_similarity_sum = similar_items.sum()
    
    # Normalize
    if item_similarity_sum == 0:
        return np.nan
    predicted_rating = weighted_sum / item_similarity_sum

    
    return predicted_rating

to_predict_df = pd.read_csv('data_sampleSubmission.csv')
print("asdfasdf")
# Add a new column for the predicted ratings
to_predict_df['prediction'] = to_predict_df.apply(
    lambda row: predict_rating(row['user_id'], row['item_id'], user_item_matrix, item_similarity_df), axis=1)
print("ASDF")
# Save the predictions to a new CSV file
to_predict_df[['user_id', 'item_id', 'prediction']].to_csv('predicted_item_ratings.csv', index=False)

print("Predicted ratings saved to 'predicted_ratings.csv'")