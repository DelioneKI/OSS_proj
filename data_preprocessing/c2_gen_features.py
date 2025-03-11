import os
import pandas as pd
import argparse
from collections import defaultdict

# ==========================
# **Argument Parsing**
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--edge_filename", type=str, required=True, help="Input Edge file name (without path, .csv optional)")
parser.add_argument("--feature_year", type=int, required=True, help="Year to extract features")
args = parser.parse_args()

# ==========================
# **File Path Settings**
# ==========================
edge_base_name, _ = os.path.splitext(args.edge_filename)
edge_filename = edge_base_name + ".csv"
output_filename = f"{edge_base_name}_{args.feature_year}_features.csv"

INPUT_EDGE_FILE_PATH = os.path.join("data_preprocessing/data/b1_extract_largest_component_edge", edge_filename)
EVENT_DATA_DIR = "GraphQL/data/repository_events"
OUTPUT_FEATURE_FILE_PATH = os.path.join("data_preprocessing/data/c2_gen_features", output_filename)

EVENT_TYPES = ["CommitEvent", "CommitCommentEvent", "IssueEvent", "IssueCommentEvent", "PullRequestEvent", "PullRequestCommentEvent"]

# ==========================
# **Load Edge File and Format Repo Indices**
# ==========================
df_edges = pd.read_csv(INPUT_EDGE_FILE_PATH, usecols=["repo_index"])

# **Ensure repo_index is always stored as a 5-digit string during processing**
repo_indices = {str(repo).zfill(5) for repo in df_edges["repo_index"].astype(str).unique()}

# **Initialize Feature Dictionary**
repo_features = {repo: defaultdict(int) for repo in repo_indices}

# ==========================
# **Process Event Files**
# ==========================
for filename in os.listdir(EVENT_DATA_DIR):
    parts = filename.split("_")
    if len(parts) < 3:
        continue  

    repo_index = parts[0].zfill(5)  # Ensure repo_index is 5 digits

    if repo_index not in repo_indices:
        continue

    event_file_path = os.path.join(EVENT_DATA_DIR, filename)

    with open(event_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) != 3:
                continue  

            event_type, _, event_time = parts
            event_year = event_time[:4]  # YYYY-MM-DDTHH:MM:SSZ -> YYYY

            if event_year == str(args.feature_year) and event_type in EVENT_TYPES:
                repo_features[repo_index][event_type] += 1

# ==========================
# **Convert to DataFrame and Save**
# ==========================
feature_data = []
for repo, counts in repo_features.items():
    feature_row = {"repo_index": int(repo)}  # Convert repo_index to integer before saving
    for event in EVENT_TYPES:
        feature_row[event] = counts[event]
    feature_data.append(feature_row)

df_features = pd.DataFrame(feature_data)

# ✅ **Sort by repo_index in ascending order**
df_features.sort_values(by="repo_index", inplace=True)

# ✅ **Save as CSV**
os.makedirs(os.path.dirname(OUTPUT_FEATURE_FILE_PATH), exist_ok=True)
df_features.to_csv(OUTPUT_FEATURE_FILE_PATH, index=False)

print(f"✅ Feature file saved: {OUTPUT_FEATURE_FILE_PATH}")