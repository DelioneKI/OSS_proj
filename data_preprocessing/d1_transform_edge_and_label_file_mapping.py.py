import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--edge_filename", type=str, required=True, help="Edge CSV file name (without path, .csv optional)")
parser.add_argument("--label_filename", type=str, required=True, help="Label CSV file name (without path, .csv optional)")
parser.add_argument("--feature_filename", type=str, required=True, help="Feature CSV file name (without path, .csv optional)")
args = parser.parse_args()


edge_base_name = os.path.splitext(args.edge_filename)[0]  
label_base_name = os.path.splitext(args.label_filename)[0]  
feature_base_name = os.path.splitext(args.feature_filename)[0]  

edge_filename = edge_base_name + ".csv"
label_filename = label_base_name + ".csv"
feature_filename = feature_base_name + ".csv"


EDGE_FILE_PATH = os.path.join("data_preprocessing/data/b1_extract_largest_component_edge", edge_filename)
LABEL_FILE_PATH = os.path.join("data_preprocessing/data/c1_gen_label", label_filename)
FEATURE_FILE_PATH = os.path.join("data_preprocessing/data/c2_gen_features", feature_filename)

OUTPUT_DIR = "data_preprocessing/data/d1_transform_edge_and_label_file_mapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)  

#  Output file path
OUTPUT_REPO_MAPPING_FILE = os.path.join(OUTPUT_DIR, f"{edge_base_name}_repo_mapping.csv")
OUTPUT_USER_MAPPING_FILE = os.path.join(OUTPUT_DIR, f"{edge_base_name}_user_mapping.csv")
OUTPUT_EDGE_FILE = os.path.join(OUTPUT_DIR, f"{edge_base_name}_mapped.csv")
OUTPUT_LABEL_FILE = os.path.join(OUTPUT_DIR, f"{label_base_name}_mapped.csv")
OUTPUT_FEATURE_FILE = os.path.join(OUTPUT_DIR, f"{feature_base_name}_mapped.csv")


df_edges = pd.read_csv(EDGE_FILE_PATH)
df_labels = pd.read_csv(LABEL_FILE_PATH)
df_features = pd.read_csv(FEATURE_FILE_PATH)

#  generate node mapping
repo_nodes = sorted(df_edges["repo_index"].unique())  
user_nodes = sorted(df_edges["actor_name"].unique())  

repo_mapping = {repo: i for i, repo in enumerate(repo_nodes)}
user_mapping = {user: i + len(repo_nodes) for i, user in enumerate(user_nodes)}
node_mapping = {**repo_mapping, **user_mapping}


pd.DataFrame(repo_mapping.items(), columns=["original_repo", "new_repo"]).to_csv(OUTPUT_REPO_MAPPING_FILE, index=False)
pd.DataFrame(user_mapping.items(), columns=["original_user", "new_user"]).to_csv(OUTPUT_USER_MAPPING_FILE, index=False)

# ==========================
# **Convert Edge List**
# ==========================
edge_index = []
edge_type = []
num_events = []

for _, row in df_edges.iterrows():
    src = node_mapping[row["repo_index"]]
    tgt = node_mapping[row["actor_name"]]
    
    edge_index.append([src, tgt])
    edge_type.append(row["event_type"])  
    num_events.append(row["num_event"])

df_new_edges = pd.DataFrame({
    "repo": [e[0] for e in edge_index],
    "actor": [e[1] for e in edge_index],
    "event_type": edge_type,  
    "num_event": num_events
})
df_new_edges.to_csv(OUTPUT_EDGE_FILE, index=False)

print(f"\n Processed Edge File Saved: {OUTPUT_EDGE_FILE}")

# ==========================
# **Convert Label File**
# ==========================
df_labels["new_repo_index"] = df_labels["repo_index"].map(repo_mapping)
df_labels = df_labels.dropna().astype({"new_repo_index": "int"})  
df_labels = df_labels.sort_values("new_repo_index")[["new_repo_index", "label"]]
df_labels.to_csv(OUTPUT_LABEL_FILE, index=False)

print(f"\n Processed Label File Saved: {OUTPUT_LABEL_FILE}")

# ==========================
# **Convert Feature File**
# ==========================
df_features["new_repo_index"] = df_features["repo_index"].map(repo_mapping)
df_features = df_features.dropna().astype({"new_repo_index": "int"})  

feature_columns = ["new_repo_index"] + [col for col in df_features.columns if col != "repo_index" and col != "new_repo_index"]
df_features = df_features[feature_columns]

df_features.to_csv(OUTPUT_FEATURE_FILE, index=False)

print(f"\n Processed Feature File Saved: {OUTPUT_FEATURE_FILE}")