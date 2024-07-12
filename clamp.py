import pandas as pd

# Read the original CSV file
df = pd.read_csv('./formatted_predictions.csv')

# Function to adjust predictions
def adjust_prediction(prediction):
    if prediction < 1:
        return 1
    elif prediction > 5:
        return 5
    else:
        return prediction

# Apply the adjustment to the Prediction column
df['Prediction'] = df['Prediction'].apply(adjust_prediction)

# Save the adjusted DataFrame to a new CSV file
df.to_csv('results7.csv', index=False, float_format='%.7f')

print("Adjusted predictions saved to adjusted_predictions.csv")

# {'num_factors': 200, 'num_users': 10000, 'num_items': 1000, 'wd': 1e-05, 'lr': 0.0001, 'optimizer': 'Adam', 'num_epochs': 150}
# Training with config: {'num_factors': 100, 'num_users': 10000, 'num_items': 1000, 'wd': 0.0001, 'lr': 0.001, 'optimizer': 'Adam', 'num_epochs': 150}
# Training with config: {'num_factors': 200, 'num_users': 10000, 'num_items': 1000, 'wd': 1e-06, 'lr': 0.001, 'optimizer': 'Adam', 'num_epochs': 150}