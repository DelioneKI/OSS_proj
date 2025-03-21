import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--edge_filename", type=str, required=True, help="Input Edge file name (without path, .csv optional)")
parser.add_argument("--label_year", type=int, required=True, help="Year to check if it has continued")
parser.add_argument("--num_commit", type=int, required=True, help="Number of commits to determine label")
args = parser.parse_args()

edge_base_name, _ = os.path.splitext(args.edge_filename)
edge_filename = edge_base_name + ".csv"
output_filename = f"{edge_base_name}_{args.label_year}_{args.num_commit}_label.csv"

INPUT_EDGE_FILE_PATH = os.path.join("data_preprocessing/data/b1_extract_largest_component_edge", edge_filename)
EVENT_DATA_DIR = "GraphQL/data/repository_events"
BOT_LIST_PATH = "data_preprocessing/data/additional_bot_list.txt"
OUTPUT_LABEL_FILE_PATH = os.path.join("data_preprocessing/data/c1_gen_label", output_filename)

# Extract repo_index directly from Edge file (Convert to integer)
df_edges = pd.read_csv(INPUT_EDGE_FILE_PATH, usecols=["repo_index"])
repo_indices = set(df_edges["repo_index"].astype(int).unique())  # Convert to integer

# Load bot list
with open(BOT_LIST_PATH, "r") as f:
    bot_list = set(line.strip().lower() for line in f)

label_data = []

# Process event files
for filename in os.listdir(EVENT_DATA_DIR):
    parts = filename.split("_")
    if len(parts) < 3:
        continue

    try:
        repo_index = str(int(parts[0]))  # Convert 5-digit string to integer (e.g., "00001" → 1)
    except ValueError:
        continue  # Skip invalid filenames

    # Process only repo indices extracted from the edge file
    if int(repo_index) not in repo_indices:
        continue

    event_file_path = os.path.join(EVENT_DATA_DIR, filename)

    # Read files and count CommitEvents
    commit_count = 0
    with open(event_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) != 3:
                continue

            event_type, user_name, event_time = parts
            event_year = event_time[:4]  # YYYY-MM-DDTHH:MM:SSZ → YYYY

            # 'label_year' year & CommitEvent & non-bot user check
            if event_year == str(args.label_year) and event_type == "CommitEvent":
                user_name_lower = user_name.lower()
                if user_name_lower not in bot_list and "bot" not in user_name_lower:
                    commit_count += 1

    #  Assign label (1 if CommitEvent count is 'num_commit' or more, else 0)
    label = 1 if commit_count >= args.num_commit else 0
    label_data.append((int(repo_index), label))  # Convert repo_index back to integer

#  Create DataFrame & Sort by repo_index
label_df = pd.DataFrame(label_data, columns=["repo_index", "label"])
label_df.sort_values(by="repo_index", inplace=True)  # Sort by repo_index in ascending order

#  Save the label file
label_df.to_csv(OUTPUT_LABEL_FILE_PATH, index=False)
print(f" Label file saved: {OUTPUT_LABEL_FILE_PATH}")