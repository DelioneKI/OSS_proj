import pandas as pd


edge_file_path = "data_preprocessing/data/b1_extract_largest_component_edge/v1_large_comp.csv"  
df = pd.read_csv(edge_file_path)


df["repo_index"] = df["repo_index"].astype(str)
df["actor_name"] = df["actor_name"].astype(str)


repo_set = set(df["repo_index"])
actor_set = set(df["actor_name"])


matching_values = repo_set & actor_set


if matching_values:
    print("The following values are present in BOTH repo_index and actor_name:")
    for value in sorted(matching_values):
        print(value)
    
else:
    print(" No matching values found between repo_index and actor_name!")