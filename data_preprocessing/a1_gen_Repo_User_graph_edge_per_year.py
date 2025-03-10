import os
import pandas as pd
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, help="Extract event data for a specific year")
parser.add_argument("--output_filename", type=str, required=True, help="Output file name (without path)")
args = parser.parse_args()

edge_base_name = os.path.splitext(args.output_filename)[0]  
edge_filename = edge_base_name + ".csv"

EVENT_DIR = "GraphQL/data/repository_events"  
OUTPUT_EDGE_FILE_DIR = "data_preprocessing/data/a1_gen_Repo_USER_graph_edge_per_year"
OUTPUT_EDGE_FILE_PATH = os.path.join(OUTPUT_EDGE_FILE_DIR,edge_filename)
BOT_LIST_FILE_PATH = "data_preprocessing/data/additional_bot_list.txt"


def load_bot_list(file_path):
    bot_set = set()
    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} does not exist. Only basic filtering is applied.")
        return bot_set

    with open(file_path, "r") as file:
        for line in file:
            bot_name = line.strip().lower()
            if bot_name:
                bot_set.add(bot_name)
    
    return bot_set


bot_list = load_bot_list(BOT_LIST_FILE_PATH)
target_year = str(args.year)  


# Edge data for save (repo_index → source, user → destination, count by event_type)
edge_data = defaultdict(int)


print("Processing event files...")
event_files = [f for f in os.listdir(EVENT_DIR) if f.endswith("_events.txt")]

if not event_files:
    print("⚠️ No event files found in directory. Please check EVENT_DIR.")
    exit()

total_files = len(event_files)

for idx, file_name in enumerate(event_files):
    file_path = os.path.join(EVENT_DIR, file_name)
    
    # extract repo_index
    repo_name_parts = file_name.split("_")
    if len(repo_name_parts) < 3:
        continue
    repo_index = repo_name_parts[0]  

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            if len(parts) != 3:
                continue
            
            event_type, actor_name, timestamp = parts
            event_year = timestamp[:4]  # YYYY-MM-DD 

            # remove bot
            if not actor_name or actor_name.lower() == "none" or "bot" in actor_name.lower() or actor_name.lower() in bot_list:
                continue

            if event_year == target_year:
                edge_data[(repo_index, actor_name, event_type)] += 1

    if idx % 100 == 0 or idx == total_files - 1:
        print(f"Processed {idx+1}/{total_files} files...")

if not edge_data:
    print("⚠️ No matching events found for the target year. Check the event data and filters.")
    exit()

df_edges = pd.DataFrame([
    {"repo_index": source, "actor_name": destination, "event_type": event, "num_event": count}
    for (source, destination, event), count in edge_data.items()
])

print("\n🔍 Checking DataFrame Columns...")
print(df_edges.columns) 

print("\n🔍 Checking DataFrame Sample...")
print(df_edges.head())  


if df_edges.empty:
    print("⚠️ Warning: df_edges is empty! Check if events were correctly processed.")
    exit()


os.makedirs(OUTPUT_EDGE_FILE_DIR, exist_ok=True)
df_edges.to_csv(OUTPUT_EDGE_FILE_PATH, index=False, encoding="utf-8")


num_unique_repos = df_edges["repo_index"].nunique()
num_unique_actors = df_edges["actor_name"].nunique()

print(f"\n Total unique repositories (repo_index): {num_unique_repos}")
print(f" Total unique actors (users): {num_unique_actors}")
print(f" Process completed! Edge file saved to: {OUTPUT_EDGE_FILE_PATH}")