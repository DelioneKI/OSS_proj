import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Load the bipartite graph edge file
df = pd.read_csv("data_preprocessing/data/d1_transform_edge_and_label_file_mapping/v2020_lc_mapped.csv")

# Create a bipartite graph using networkx
B = nx.Graph()
repo_nodes = df["repo"].unique()
user_nodes = df["actor"].unique()

# Add nodes (repositories and users)
B.add_nodes_from(repo_nodes, bipartite=0)  # Repository nodes
B.add_nodes_from(user_nodes, bipartite=1)  # User nodes

# Add edges (repo - user connections)
edges = list(zip(df["repo"], df["actor"]))
B.add_edges_from(edges)

# =============================
# **Overall Degree Distribution (All Event Types)**
# =============================

# Compute degree for each repository node
repo_degrees = {node: B.degree(node) for node in repo_nodes}  # repo만 고려

# Define bins and labels (5단위)
bin_labels_with_0 = ["0", "1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "50+"]

# Initialize dictionary for counting
degree_counts = {label: 0 for label in bin_labels_with_0}

print(f"{len(repo_degrees)} repositories considered.")

# Manually count degrees
for degree in repo_degrees.values():
    if degree == 0:
        degree_counts["0"] += 1
    elif 1 <= degree <= 5:
        degree_counts["1-5"] += 1
    elif 6 <= degree <= 10:
        degree_counts["6-10"] += 1
    elif 11 <= degree <= 15:
        degree_counts["11-15"] += 1
    elif 16 <= degree <= 20:
        degree_counts["16-20"] += 1
    elif 21 <= degree <= 25:
        degree_counts["21-25"] += 1
    elif 26 <= degree <= 30:
        degree_counts["26-30"] += 1
    elif 31 <= degree <= 35:
        degree_counts["31-35"] += 1
    elif 36 <= degree <= 40:
        degree_counts["36-40"] += 1
    elif 41 <= degree <= 45:
        degree_counts["41-45"] += 1
    elif 46 <= degree <= 50:
        degree_counts["46-50"] += 1
    else:
        degree_counts["50+"] += 1  # 50+ bin

# Convert dictionary values to list
degree_counts_list = [degree_counts[label] for label in bin_labels_with_0]

# Plot overall degree distribution with labels
plt.figure(figsize=(8, 5))
bars = plt.bar(bin_labels_with_0, degree_counts_list, width=0.6, align="center")

# Add value labels on top of bars
for bar, count in zip(bars, degree_counts_list):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), ha="center", va="bottom")

plt.xlabel("Degree Ranges")
plt.ylabel("Number of Repositories")
plt.title("Repository Degree Distribution (All Event Types)")
plt.xticks(rotation=45)
plt.show()

# =============================
# **Per Event Type Degree Distribution**
# =============================

event_types = df["event_type"].unique()

for event in event_types:
    print(f"\nProcessing {event}...")
    
    # Create a new graph for this event type
    B_event = nx.Graph()
    B_event.add_nodes_from(repo_nodes, bipartite=0)  # Repository nodes
    B_event.add_nodes_from(user_nodes, bipartite=1)  # User nodes

    # Filter edges by event type and add them
    edges_event = list(zip(df[df["event_type"] == event]["repo"], df[df["event_type"] == event]["actor"]))
    B_event.add_edges_from(edges_event)

    # Compute degree per repository
    repo_degrees_event = {node: B_event.degree(node) for node in repo_nodes}

    # Initialize dictionary for counting
    degree_counts_event = {label: 0 for label in bin_labels_with_0}

    # Count degrees per bin
    for degree in repo_degrees_event.values():
        if degree == 0:
            degree_counts_event["0"] += 1
        elif 1 <= degree <= 5:
            degree_counts_event["1-5"] += 1
        elif 6 <= degree <= 10:
            degree_counts_event["6-10"] += 1
        elif 11 <= degree <= 15:
            degree_counts_event["11-15"] += 1
        elif 16 <= degree <= 20:
            degree_counts_event["16-20"] += 1
        elif 21 <= degree <= 25:
            degree_counts_event["21-25"] += 1
        elif 26 <= degree <= 30:
            degree_counts_event["26-30"] += 1
        elif 31 <= degree <= 35:
            degree_counts_event["31-35"] += 1
        elif 36 <= degree <= 40:
            degree_counts_event["36-40"] += 1
        elif 41 <= degree <= 45:
            degree_counts_event["41-45"] += 1
        elif 46 <= degree <= 50:
            degree_counts_event["46-50"] += 1
        else:
            degree_counts_event["50+"] += 1  # 50+ bin

    # Convert dictionary values to list
    degree_counts_list_event = [degree_counts_event[label] for label in bin_labels_with_0]

    # Plot event-type specific degree distribution with labels
    plt.figure(figsize=(8, 5))
    bars = plt.bar(bin_labels_with_0, degree_counts_list_event, width=0.6, align="center")

    # Add value labels on top of bars
    for bar, count in zip(bars, degree_counts_list_event):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), ha="center", va="bottom")

    plt.xlabel("Degree Ranges")
    plt.ylabel("Number of Repositories")
    plt.title(f"Repository Degree Distribution ({event})")
    plt.xticks(rotation=45)
    plt.show()