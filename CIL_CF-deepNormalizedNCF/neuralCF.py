#https://dl.acm.org/doi/pdf/10.1145/3038912.3052569    (Neural Collaborative Filtering by Xiangnan HE et. al)
import os
import numpy as np
import torch
from torch import nn
class NeuCF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, num_hiddens,
                 **kwargs):
        super(NeuCF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential(
            nn.Linear(num_factors*2, num_hiddens, bias=True),
            nn.GELU(),
            nn.Linear(num_hiddens, num_hiddens, bias=True),
            nn.GELU(),
            nn.Linear(num_hiddens, num_hiddens, bias=True),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hiddens, 2 * num_hiddens, bias=True),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * num_hiddens, 2 * num_hiddens, bias=True),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * num_hiddens, 4 * num_hiddens, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * num_hiddens, 4 * num_hiddens, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * num_hiddens, 8 * num_hiddens, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8 * num_hiddens, 4 * num_hiddens, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * num_hiddens, 4 * num_hiddens, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * num_hiddens, 2 * num_hiddens, bias=True),

            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(2 * num_hiddens, num_hiddens, bias=True),

            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(num_hiddens, num_hiddens, bias=True)
        )
        #regularization, dropout to prevent overfitting and improve generalization of the model (based on 2 tries only) proved to be not so useful
        self.prediction_layer = nn.Linear(num_factors + num_hiddens, 1, bias=False)
        #self.dropout = nn.Dropout(0.25)

        #nn.ReLU(),
        #nn.Linear(32, 1)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp_res = self.mlp(torch.concat([p_mlp, q_mlp], axis=1))
        con_res = torch.concat([gmf, mlp_res], axis=1)
        return self.prediction_layer(con_res).flatten()
    
        #user_embedding = self.user_embedding(user_indices)
        #item_embedding = self.item_embedding(item_indices)
        #x = torch.cat([user_embedding, item_embedding], dim=-1)
        #output = self.fc_layers(x)

def get_net(config, device=None):
  model = NeuCF(config['num_factors'], config['num_users'], config['num_items'], config['num_hiddens'])
  model = model.to(device)
  return model  



#could still do: (psuedocode)
# BPR loss function(integrating pairwise ranking loss) better model user preferences over pairs of items than individual item scor
# Function to add adversarial noise to embeddings to improve generalization
def add_adversarial_noise(model, user_indices, item_indices, ratings, epsilon=0.01):
    loss_fn = nn.MSELoss(reduction='mean')
    user_indices = user_indices.long()
    item_indices = item_indices.long()

    # get original embeddings and set requires_grad
    # user_embeddings = model.P(user_indices).detach().requires_grad_(True)
    # item_embeddings = model.Q(item_indices).detach().requires_grad_(True)
    
    # concatenate embeddings and get predictions
    # concatenated_embeddings = torch.cat([user_embeddings, item_embeddings], dim=-1)
    # predictions = model(user_indices, item_indices).squeeze()

    # get original embeddings and set requires_grad
    user_embeddings = model.P(user_indices).detach().clone().requires_grad_(True)
    item_embeddings = model.Q(item_indices).detach().clone().requires_grad_(True)

    # forward pass with original embeddings
    gmf = user_embeddings * item_embeddings
    mlp_res = model.mlp(torch.cat([user_embeddings, item_embeddings], dim=-1))
    con_res = torch.cat([gmf, mlp_res], dim=-1)
    predictions = model.prediction_layer(con_res).flatten()

    # calculate loss and backpropagate
    loss = loss_fn(predictions, ratings)
    loss.backward(retain_graph=True)
    
    # add adversarial noise
    user_adversarial_noise = epsilon * user_embeddings.grad.sign()
    item_adversarial_noise = epsilon * item_embeddings.grad.sign()

    perturbed_user_embeddings = user_embeddings + user_adversarial_noise
    perturbed_item_embeddings = item_embeddings + item_adversarial_noise

    return perturbed_user_embeddings, perturbed_item_embeddings


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
            #u = torch.LongTensor([u]).to(device)
            #i = torch.LongTensor([i]).to(device)
            #r = torch.FloatTensor([r]).to(device)
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()

            # Adversarial training
            #perturbed_user_embeddings, perturbed_item_embeddings = add_adversarial_noise(
            #    model, u, i, r, epsilon=0.01
            #)

            # replacing the original embeddings with perturbed ones for this batch
            #model.P.weight.data[u] = perturbed_user_embeddings
            #model.Q.weight.data[i] = perturbed_item_embeddings

            #output = model(perturbed_user_embeddings, perturbed_item_embeddings)
            #no ui dictiannlry in model now

            # forward pass with perturbed embeddings
            output = model(u, i)
            l = loss_fn(output, r)

            # adding L1 regularization
            l1_norm = sum(param.abs().sum() for param in model.parameters())
            l += 0.0001 * l1_norm

            l.backward()
            optimizer.step()
            with torch.no_grad():
                tr_rmse += np.sqrt(loss_fn(output, r).cpu().numpy())
        
        # Eval
        val_rmse = 0
        model.eval()
        for u, i, r in valid_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            r_hat = model(u, i)
            with torch.no_grad():
                val_rmse += np.sqrt(loss_fn(r_hat, r).cpu().numpy())
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        #Save Checkpoint -> uncomment if looking at more than one combination
        #if val_rmse < best_val_rmse:
        #    best_val_rmse = val_rmse
        #    if checkpoint_dir:
        #        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        #        torch.save((model.state_dict(), optimizer.state_dict()), path)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train RMSE: {tr_rmse / len(train_loader)}, Valid RMSE: {val_rmse / len(valid_loader)}")

    return best_val_rmse