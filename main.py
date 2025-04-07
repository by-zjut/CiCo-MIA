import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import wasserstein_distance


class RecommendationDataset(Dataset):
  def __init__(self, user_ids, item_ids, labels):
    self.users = torch.LongTensor(user_ids)
    self.items = torch.LongTensor(item_ids)
    self.labels = torch.FloatTensor(labels)

  def __len__(self):
    return len(self.users)

  def __getitem__(self, idx):
    return self.users[idx], self.items[idx], self.labels[idx]


class CausalRecModel(nn.Module):
  def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
    super().__init__()
    self.user_emb = nn.Embedding(num_users, embedding_dim)
    self.item_emb = nn.Embedding(num_items, embedding_dim)


    self.z_network = nn.Sequential(
      nn.Linear(2 * embedding_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, embedding_dim)  # 潜在变量Z
    )


    self.predictor = nn.Sequential(
      nn.Linear(embedding_dim, 1),
      nn.Sigmoid()
    )

  def forward(self, users, items, do_intervention=False):
    u_emb = self.user_emb(users)
    i_emb = self.item_emb(items)

    if do_intervention:

      u_emb = u_emb.detach()
      i_emb = i_emb.detach()


    z_input = torch.cat([u_emb, i_emb], dim=1)
    z = self.z_network(z_input)


    y_pred = self.predictor(z)
    return y_pred.squeeze()


# 动态联邦聚合
class DynamicAggregator:
  def __init__(self, base_model, alpha=0.5, temp=0.1):
    self.global_model = base_model
    self.client_histories = {}
    self.alpha = alpha
    self.temp = temp

  def compute_weights(self, client_grads):

    global_grad = self._get_global_grad()
    weights = []

    for c_id, grad in client_grads.items():

      cos_sim = torch.cosine_similarity(grad, global_grad, dim=0)
      dir_weight = torch.exp(cos_sim / self.temp)


      if c_id not in self.client_histories:
        self.client_histories[c_id] = []
      hist = self.client_histories[c_id]
      if len(hist) > 1:
        stability = 1 / (1 + torch.std(torch.stack(hist)))
      else:
        stability = 1.0
      hist.append(grad.clone())


      weight = dir_weight * stability
      weights.append(weight)


    weights = torch.softmax(torch.tensor(weights), dim=0)
    return weights

  def aggregate(self, client_models, client_grads):
    weights = self.compute_weights(client_grads)
    global_state = self.global_model.state_dict()


    for key in global_state:
      global_state[key] = sum(w * client_models[i].state_dict()[key]
                              for i, w in enumerate(weights))

    self.global_model.load_state_dict(global_state)
    return self.global_model



class MembershipInference:
  def __init__(self, shadow_model):
    self.shadow_model = shadow_model
    self.emb_sim_threshold = 0.7

  def detect_leakage(self, target_emb, shadow_emb):

    distance = wasserstein_distance(target_emb, shadow_emb)
    similarity = 1 - distance
    return similarity > self.emb_sim_threshold


def train_shadow_model(shadow_data, num_epochs=50):

  num_users = len(torch.unique(shadow_data.users))
  num_items = len(torch.unique(shadow_data.items))
  model = CausalRecModel(num_users, num_items)


  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)


  loader = DataLoader(shadow_data, batch_size=256, shuffle=True)


  for epoch in range(num_epochs):
    for users, items, labels in loader:
      optimizer.zero_grad()


      preds = model(users, items, do_intervention=True)
      loss = criterion(preds, labels)


      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

  return model



if __name__ == "__main__":

  user_ids = np.random.randint(0, 100, 1000)
  item_ids = np.random.randint(0, 200, 1000)
  labels = np.random.randint(0, 2, 1000).astype(float)
  shadow_data = RecommendationDataset(user_ids, item_ids, labels)


  print("Training Shadow Model...")
  shadow_model = train_shadow_model(shadow_data)


  aggregator = DynamicAggregator(shadow_model)
  attacker = MembershipInference(shadow_model)


  client_models = [CausalRecModel(100, 200) for _ in range(5)]
  client_grads = {i: torch.randn(100) for i in range(5)}


  global_model = aggregator.aggregate(client_models, client_grads)


  target_item_emb = shadow_model.item_emb(torch.LongTensor([0]))
  for client_model in client_models:
    client_item_emb = client_model.item_emb(torch.LongTensor([0]))
    if attacker.detect_leakage(client_item_emb.detach().numpy(),
                               target_item_emb.detach().numpy()):
      print("Potential membership leakage detected!")
