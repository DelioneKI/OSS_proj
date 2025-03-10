import argparse
import pandas as pd
import os
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--edge_filename", type=str, required=True, help="Input Edge file name (without path)")
parser.add_argument("--output_filename", type=str, required=True, help="Output Edge file name (without path)")
args = parser.parse_args()

edge_base_name = os.path.splitext(args.edge_filename)[0]  
output_base_name = os.path.splitext(args.output_filename)[0]  

edge_filename = edge_base_name + ".csv"
output_filename = output_base_name + ".csv"

INPUT_EDGE_FILE_PATH = os.path.join("data_preprocessing/data/a1_gen_Repo_USER_graph_edge_per_year", edge_filename)  
OUTPUT_EDGE_FILE_PATH = os.path.join("data_preprocessing/data/b1_extract_largest_component_edge", output_filename)  
# REPO_LIST_FILE = "data_preprocessing/data/b2_extract_largest_component_edge/b2_largest_component_repos_list.csv"  

if not os.path.exists(INPUT_EDGE_FILE_PATH):
    print(f"The input edge file does not exist: {INPUT_EDGE_FILE_PATH}")
    exit(1)

df = pd.read_csv(INPUT_EDGE_FILE_PATH)
print(f"File loading complete: {INPUT_EDGE_FILE_PATH}")

repo_indices = set(df["repo_index"].astype(str))  

# Create a graph (using repo_index and actor_name as nodes)
G = nx.Graph()
for _, row in df.iterrows():
    repo = "repo"+str(row["repo_index"]) 
    actor = "user"+row["actor_name"]
    G.add_edge(repo, actor)
    G.add_edge(actor, repo)

# Find all connected components
components = list(nx.connected_components(G))

# Find the largest connected component
largest_component = max(components, key=len)

# Filter only repositories included in Largest Component
cleaned_repos = {node[4:] for node in largest_component if node.startswith("repo")}
largest_repos = sorted(cleaned_repos & repo_indices)

# Save Largest Component repository list
# pd.DataFrame(largest_repos, columns=["repo_index"]).to_csv(REPO_LIST_FILE, index=False)

# Extract only the edges of the repositories included in the Largest Component repository
df_lcc_edges = df[df["repo_index"].astype(str).isin(largest_repos)]

df_lcc_edges.to_csv(OUTPUT_EDGE_FILE_PATH, index=False)
print(f"Edge save of the largest component completed: {OUTPUT_EDGE_FILE_PATH}")