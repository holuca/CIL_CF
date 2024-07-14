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
        #same variable names as in matrix factorization paper
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, item_id, u_i_dict):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        mu = self.global_bias
        #only diff: added dot product at the end
        dot = torch.mul(P_u, Q_i).sum

        if len(b_u) < 2:
          outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + (P_u * Q_i).sum() 
          #outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + dot (same as before)
        else:
          outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + (P_u * Q_i).sum(axis=1) 
          #outputs = mu + torch.squeeze(b_u) + torch.squeeze(b_i) + dot(dim=1)
        return outputs.flatten()

def get_net(config, device=None):
  #num_factors: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
  model = SVDpp(config['num_factors'], config['num_users'], config['num_items'], device)
  model = model.to(device)
  return model


def trainable_function(config, train_loader=None, valid_loader=None, train_u_i_dict=None, valid_u_i_dict=None, device=None, checkpoint_dir=None):
    model = get_net(config, device=device)

    loss_fn = nn.MSELoss(reduction='mean')
    # Define an optimizer (change Adam to SGD if SGD preferred)
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), 
        weight_decay=config["wd"], lr=config["lr"]
    )

    best_val_rmse = float('inf')

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
        
        # Eval
        val_rmse = 0
        model.eval()
        for u, i, r in valid_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            r_hat = model(u, i, valid_u_i_dict)
            with torch.no_grad():
                val_rmse += np.sqrt(loss_fn(r_hat, r).cpu().numpy())
        #path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        #torch.save((model.state_dict(), optimizer.state_dict()), path)
        #Save Checkpoint -> uncomment if looking at more than one combination
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train RMSE: {tr_rmse / len(train_loader)}, Valid RMSE: {val_rmse / len(valid_loader)}")

    return best_val_rmse
