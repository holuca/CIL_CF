#https://dl.acm.org/doi/abs/10.1145/1401890.1401944?casa_token=tZHDSBhztHEAAAAA:lkb_CQw_VKPJ8TIFmPc8Y7YDACAqltEn6guZzcpblnISX0vEiYIgBj3ynrTTgo_nJ0wl2XG8nHpk
import csv
import os
import pandas as pd
import numpy as np
import torch
from torch import nn



class SVDpp(nn.Module):
    def __init__(self, num_factors, num_users, num_items, device, **kwargs):
        super(SVDpp, self).__init__(**kwargs)
        self.device = device
        # plain MF params
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        # implicit feedback params
        self.y_j = nn.Embedding(num_items, num_factors) 

    def forward(self, user_id, item_id, u_i_dict):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        mu = self.global_bias


        if len(b_u) < 2:
          outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + (P_u * Q_i).sum() 
        else:
          outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + (P_u * Q_i).sum(axis=1) 
        return outputs.flatten()

def get_net(config, device=None):
  model = SVDpp(config['num_factors'], config['num_users'], config['num_items'], device)
  model = model.to(device)
  return model


def trainable_fn(config, train_loader=None, valid_loader=None, train_u_i_dict=None, valid_u_i_dict=None, device=None, checkpoint_dir=None):
    # Define a model
    model = get_net(config, device=device)

    # Define a loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Define an optimizer

    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), 
        weight_decay=config["wd"], lr=config["lr"]
    )


    
    best_val_rmse = float('inf')
    

    ######################################################################
    # Train & Eval & Save Model
    ######################################################################

    # Train
    for epoch in range(config['num_epochs']):
        tr_rmse = 0
        model.train()
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            output = model(u, i, train_u_i_dict)
            l = loss_fn(output, r)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                tr_rmse += np.sqrt(loss_fn(output, r).cpu().numpy())
        
        # Evaluate on Valid-set
        val_rmse = 0
        model.eval()
        for u, i, r in valid_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            r_hat = model(u, i, valid_u_i_dict)
            with torch.no_grad():
                val_rmse += np.sqrt(loss_fn(r_hat, r).cpu().numpy())
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        # Save Checkpoint
        #if val_rmse < best_val_rmse:
        #    best_val_rmse = val_rmse
        #    if checkpoint_dir:
        #        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        #        torch.save((model.state_dict(), optimizer.state_dict()), path)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train RMSE: {tr_rmse / len(train_loader)}, Valid RMSE: {val_rmse / len(valid_loader)}")

    return best_val_rmse


def try_gpu(i=0): 
    return f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu'

def get_best_config(result, metric="val_rmse", mode="min", device=None):
    # Assuming `result` is a dictionary with relevant trial information
    best_trial = min(result, key=lambda trial: trial["last_result"][metric]) if mode == "min" else max(result, key=lambda trial: trial["last_result"][metric])
    print("Best trial config: {}".format(best_trial["config"]))
    print("Best trial final validation {}: {}".format(metric, best_trial["last_result"][metric]))

    best_checkpoint_dir = best_trial["checkpoint"]
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint.pth"))
    
    return best_trial["config"], best_checkpoint_dir
