import pandas as pd
import networkx as nx

edge_file_path = "data_preprocessing/data/b1_extract_largest_component_edge/v1_large_comp.csv"  
df = pd.read_csv(edge_file_path)

G = nx.Graph()
for _, row in df.iterrows():
    repo = f"repo_{row['repo_index']}"  
    actor = f"user_{row['actor_name']}" 
    G.add_edge(repo, actor)
    G.add_edge(actor, repo)


components = list(nx.connected_components(G))


largest_component = max(components, key=len)


num_nodes = G.number_of_nodes()
largest_comp_size = len(largest_component)


if largest_comp_size == num_nodes:
    print(" Every node is a single connected component!")
else:
    print(f"Nodes are split into multiple components! (Full node:{num_nodes}, Largest number of component nodes: {largest_comp_size})")

    disconnected_components = [comp for comp in components if len(comp) < largest_comp_size]
    print(f"\n Small number of components: {len(disconnected_components)}")
    for idx, comp in enumerate(disconnected_components):
        print(f"🔹 Component {idx + 1}: {len(comp)} nodes - {comp}")