#https://dl.acm.org/doi/abs/10.1145/1401890.1401944?casa_token=tZHDSBhztHEAAAAA:lkb_CQw_VKPJ8TIFmPc8Y7YDACAqltEn6guZzcpblnISX0vEiYIgBj3ynrTTgo_nJ0wl2XG8nHpk
import csv
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from itertools import product
from functools import partial
import random
#from svdpp import *
from neuralCF import *


def read_data(data_dir):
    data = pd.read_csv(os.path.join(data_dir))
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    data['user_id'] = data['user_id'] -1 
    data['item_id'] = data['item_id'] -1 
    return data, num_users, num_items


def split_data(data, num_users, num_items, test_ratio=0.1):
    mask = [True if x == 1 else False for x in np.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
    neg_mask = [not x for x in mask]
    train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data


def load_data(data, num_users, num_items):
    users = []
    items = []
    scores = []
    inter = np.zeros((num_items, num_users))
    for line in data.itertuples():
        user_index, item_index = line[1], line[2]
        score = float(line[3])
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        inter[item_index, user_index] = score
    u_i_dict = dict(data.groupby([data.columns[0], data.columns[1]])[data.columns[2]].unique())
    

    return users, items, scores, inter, u_i_dict



def split_and_load(input_csv, test_ratio=0.1, batch_size=256):
    #input_csv = './data_train.csv'
    data, n_users, n_items = read_data(input_csv)
    train_data, test_data = split_data(data, n_users, n_items, test_ratio)
    
    train_u, train_i, train_r, _, train_ui_dict = load_data(train_data, n_users, n_items)
    test_u, test_i, test_r, _, test_ui_dict = load_data(test_data, n_users, n_items)

    train_set = torch.utils.data.TensorDataset(torch.tensor(train_u), torch.tensor(train_i), torch.tensor(train_r))
    test_set = torch.utils.data.TensorDataset(torch.tensor(test_u), torch.tensor(test_i), torch.tensor(test_r))

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size)
    #return num_users, num_items, train_iter, test_iter, train_u_i_dict, test_u_i_dict
    return n_users, n_items, train_loader, test_loader, train_ui_dict, test_ui_dict

def try_gpu(i=0):
    return f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu'

# Function to sample random configurations
def sample_config(config_space):
    return {key: random.choice(values) if isinstance(values, list) else values for key, values in config_space.items()}


##first find out best model and then pick that config
def load_best_model(checkpoint_dir, config, device):
    model = get_net(config, device)
    model_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    model_state, _ = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def predict_ratings(model, user_ids, item_ids, device):
    model.eval()
    user_ids = torch.tensor(user_ids).to(device)
    item_ids = torch.tensor(item_ids).to(device)
    with torch.no_grad():
        ratings = model(user_ids, item_ids)
    return ratings.cpu().numpy()

def adjust_prediction(prediction):
    if prediction < 1:
        return 1
    elif prediction > 5:
        return 5
    else:
        return prediction

def main(input_csv, mode):
  if(mode == "train"):
    data, num_rows, num_cols = read_data(input_csv)
  
    num_users, num_items, train_iter, valid_iter, train_u_i_dict, valid_u_i_dict = split_and_load(input_csv ,test_ratio=0.1, batch_size=256)

    config_space = {
    "num_factors": [100],
    "num_hiddens": [50],
    "num_users": num_users,
    "num_items": num_items,
    #"wd": [1e-6, 1e-5, 1e-4],
    #"lr": [1e-4, 1e-3, 1e-2],
    "wd": [0.00001],
    "lr": [0.0001],
    "optimizer": ["Adam"],
    "num_epochs": [50]
    }

    # Data loaders (replace with actual data loaders)
    train_loader = train_iter
    valid_loader = valid_iter

    device = try_gpu()
    best_config = None
    best_val_rmse = float('inf')

    # Random search with num_samples
    num_samples = 1  # Adjust as needed
    for _ in range(num_samples):
        config = sample_config(config_space)
        print(f"Training with config: {config}")
        val_rmse = trainable_function(config, train_loader, valid_loader, train_u_i_dict, valid_u_i_dict, device, os.path.abspath("./checkpoints"))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = config
    print(f"Best configuration: {best_config}")
    print(f"Best validation RMSE: {best_val_rmse}")

    #test
  else: 

    data = pd.read_csv(os.path.join(input_csv))
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    data['user_id'] = data['user_id'] -1 
    data['item_id'] = data['item_id'] -1 

    user_ids, item_ids, scores = [], [], []
    inter = np.zeros((num_items, num_users))
    for line in data.itertuples():
        user_index, item_index = line[1], line[2]
        score = float(line[3])
        user_ids.append(user_index)
        item_ids.append(item_index)
        scores.append(score) #default value 
        inter[item_index, user_index] = score
    u_i_dict = dict(data.groupby([data.columns[0], data.columns[1]])[data.columns[2]].unique())


    best_checkpoint_dir = "./best_checkpoint"
    best_config = {'num_factors': 100, 'num_hiddens': 50, 'num_users': 10000, 'num_items': 1000, 'wd': 0.0001, 'lr': 0.001, 'optimizer': 'Adam', 'num_epochs': 50}
    #100, 50
    #100, 100
    best_model = load_best_model(best_checkpoint_dir, best_config, device=None)
    # Predict ratings for missing pairs
    predicted_ratings = predict_ratings(best_model, user_ids, item_ids, device=None)

    # Combine user-item pairs with predicted ratings
    predictions = list(zip(user_ids, item_ids, predicted_ratings))
    predictions_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])

    # Save the DataFrame to a CSV file
    #predictions_df.to_csv('predictions.csv', index=False)

    predictions_df['Id'] = predictions_df.apply(lambda row: f"r{int(row['user'])+1}_c{int(row['item'])+1}", axis=1)

    # Create a new DataFrame with Id and Prediction columns
    formatted_df = predictions_df[['Id', 'prediction']].rename(columns={'prediction': 'Prediction'})
    #all nums inbetween 1 and 5
    formatted_df['Prediction'] = formatted_df['Prediction'].apply(adjust_prediction)

    # Save the adjusted DataFrame to a new CSV file
    formatted_df.to_csv('results_current.csv', index=False, float_format='%.7f')

if __name__ == "__main__":
    input_csv = './data_train.csv'
    #input_csv = './data_sampleSubmission.csv'
    main(input_csv, "train")
