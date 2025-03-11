import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import xgboost as xgb

# ==============================
# Step 1: Argument Parsing
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# Step 2: Load Data and Build Graph Structuresㄱ
# ==============================

# Load Repo feature, edge data 
df = pd.read_csv("data_preprocessing/data/d1_transform_edge_and_label_file_mapping/v2020_lc_2020_features_mapped.csv")
df_edge = pd.read_csv("data_preprocessing/data/d1_transform_edge_and_label_file_mapping/v2020_lc_mapped.csv")

# num repo & user
num_users = df_edge["actor"].nunique()
num_repos = df_edge["repo"].nunique()
num_event_types = df_edge["event_type"].nunique()

# Build the repository feature matrix (Excluding column names)
repo_feature_matrix = df.iloc[:, 1:].to_numpy()  

# Global z-score normalization (all values together)
global_mean = repo_feature_matrix.mean()
global_std = repo_feature_matrix.std()
repo_feature_matrix = (repo_feature_matrix - global_mean) / global_std

# Convert to torch tensor and move to device
repo_features = torch.tensor(repo_feature_matrix, dtype=torch.float).to(device)

# Initialize user features as zeros (same number of features as repo features) and move to device
user_features = torch.zeros((num_users, num_event_types), dtype=torch.float).to(device)

# repo -> actor edges and corresponding types (move to device)
edge_index_ru = torch.tensor(df_edge[["repo", "actor"]].values, dtype=torch.long).T.to(device)  
edge_type_ru = torch.tensor(pd.factorize(df_edge["event_type"])[0], dtype=torch.long).to(device)

# For the reverse edges (user->repo), simply flip the edge_index_ru:
edge_index_ur = edge_index_ru[[1, 0]] 
edge_type_ur = edge_type_ru.clone()  

# ==============================
# Step 3: Load Labels and Prepare Train/Test Masks
# ==============================

# Load labels from label.csv (already mapped)
df_labels = pd.read_csv("OSS_proj/data_preprocessing/data/d1_transform_edge_and_label_file_mapping/v2020_lc_2021_12_label_mapped.csv")  

# Initialize label array for all repos with -1 (for missing labels)
repo_labels = np.full(num_repos, -1)  

# Assign labels directly using new_repo_index
repo_labels[df_labels["new_repo_index"].values] = df_labels["label"].values

# Convert to PyTorch tensor and move to device
repo_labels = torch.tensor(repo_labels, dtype=torch.long).to(device)

# Create train/test split (only for repos with valid labels, i.e., label != -1)
valid_idx = (repo_labels != -1).nonzero(as_tuple=True)[0]
train_idx, test_idx = train_test_split(valid_idx.cpu().numpy(), test_size=0.3, random_state=42)
train_mask = torch.zeros(num_repos, dtype=torch.bool).to(device)
test_mask = torch.zeros(num_repos, dtype=torch.bool).to(device)
train_mask[train_idx] = True
test_mask[test_idx] = True

# ==============================
# Step 4: Combine Node Features into a Single Tensor
# ==============================
x = torch.cat([repo_features, user_features], dim=0).to(device)  # shape: (num_repos+num_users, num_event_types)

# ==============================
# Step 5: Define the Bipartite R-GCN Model
# ==============================

class BipartiteRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout_p=0.5):
        super().__init__()
        self.mlp = nn.Linear(in_channels, hidden_channels)
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x, edge_index_ru, edge_index_ur, edge_type_ru, edge_type_ur):
        x = self.mlp(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv1(x, edge_index_ru, edge_type_ru)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index_ur, edge_type_ur)
        x = self.relu(x)
        x = self.dropout(x)
        
        repo_emb = x[:num_repos]
        out = self.classifier(repo_emb)
        return out

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.fc3 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.fc4 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.fc5 = nn.Linear(hidden_channels, out_channels, bias=False)  # Output Layer

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.01)  # 음수에 대해 0.01 배율 적용
        self.dropout = nn.Dropout(p=dropout_p)  # Dropout 추가

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn3(self.fc3(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn4(self.fc4(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)  # No activation on output layer
        return x
    
# Initialize models
rgcn_model = BipartiteRGCN(in_channels=num_event_types, hidden_channels=128, out_channels=2, num_relations=num_event_types).to(device)
mlp_model = MLPModel(in_channels=num_event_types, hidden_channels=128, out_channels=2).to(device)

optimizer_rgcn = optim.Adam(rgcn_model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
# ==============================
# Step 6: Training and Testing Functions
# ==============================

def train_model(model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    if isinstance(model, BipartiteRGCN):
        out = model(x, edge_index_ru, edge_index_ur, edge_type_ru, edge_type_ur)
    else:
        out = model(repo_features)  
    
    loss = criterion(out[train_mask], repo_labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, test_mask):
    model.eval()
    with torch.no_grad():
        if isinstance(model, BipartiteRGCN):
            out = model(x, edge_index_ru, edge_index_ur, edge_type_ru, edge_type_ur)
        else:
            out = model(repo_features)
    
    pred = out.argmax(dim=1)
    acc = accuracy_score(repo_labels[test_mask].cpu(), pred[test_mask].cpu())
    return acc

# Train XGBoost
def train_xgboost():
    
    # NumPy 배열 변환
    train_idx_np = train_idx.cpu().numpy() if isinstance(train_idx, torch.Tensor) else train_idx
    test_idx_np = test_idx.cpu().numpy() if isinstance(test_idx, torch.Tensor) else test_idx
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss")
    xgb_model.fit(repo_feature_matrix[train_idx_np], repo_labels.cpu().numpy()[train_idx_np])  # train 데이터 사용
    xgb_preds = xgb_model.predict(repo_feature_matrix[test_idx_np])  # test 데이터 예측
    
    return accuracy_score(repo_labels.cpu().numpy()[test_idx_np], xgb_preds)

# ==============================
# Step 7: Training Loop
# ==============================

random_states = [42, 55, 67, 89, 101]
rgcn_acc_list = []
mlp_acc_list = []
xgb_acc_list = []

for seed in random_states:
    print(f"\nEvaluating with random state {seed}...\n")

    # Train/Test split with new seed
    train_idx, test_idx = train_test_split(valid_idx.cpu().numpy(), test_size=0.3, random_state=seed)
    train_mask = torch.zeros(num_repos, dtype=torch.bool).to(device)
    test_mask = torch.zeros(num_repos, dtype=torch.bool).to(device)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Initialize models
    rgcn_model = BipartiteRGCN(in_channels=num_event_types, hidden_channels=128, out_channels=2, num_relations=num_event_types).to(device)
    mlp_model = MLPModel(in_channels=num_event_types, hidden_channels=128, out_channels=2).to(device)

    optimizer_rgcn = optim.Adam(rgcn_model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Train RGCN
    for epoch in range(300):  
        loss = train_model(rgcn_model, optimizer_rgcn, train_mask)
    rgcn_acc_list.append(evaluate_model(rgcn_model, test_mask))

    # Train MLP
    for epoch in range(600):  
        loss = train_model(mlp_model, optimizer_mlp, train_mask)
    mlp_acc_list.append(evaluate_model(mlp_model, test_mask))

    # Train XGBoost
    xgb_acc_list.append(train_xgboost())

# ==============================
# Step 8: 결과 출력
# ==============================
print("\nFinal Accuracy Results (Mean ± Std):")
print(f"✅ RGCN Accuracy: {np.mean(rgcn_acc_list):.4f} ± {np.std(rgcn_acc_list):.4f}")
print(f"✅ MLP Accuracy: {np.mean(mlp_acc_list):.4f} ± {np.std(mlp_acc_list):.4f}")
print(f"✅ XGBoost Accuracy: {np.mean(xgb_acc_list):.4f} ± {np.std(xgb_acc_list):.4f}")