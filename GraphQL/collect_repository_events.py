import os
import requests
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--start_index", type=int, help="Repository start index")
parser.add_argument("--end_index", type=int, help="Repository end index")
args = parser.parse_args()

# GitHub API Token
GITHUB_TOKEN = ""
GRAPHQL_API_URL = "https://api.github.com/graphql"

START_INDEX = args.start_index
END_INDEX = args.end_index

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}"
}

# GraphQL Query for multiple event types
graphql_query = """
query ($owner: String!, $repo: String!, $refName: String!, $commitCursor: String, $commentCursor: String, $issueCursor: String, $prCursor: String) {
  repository(owner: $owner, name: $repo) {
    ref(qualifiedName: $refName) {
      target {
        ... on Commit {
          history(first: 100, after: $commitCursor) {
            edges {
              node {
                committedDate
                author {
                  user {
                    login
                  }
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
      }
    }
    commitComments(first: 100, after: $commentCursor) {
      edges {
        node {
          createdAt
          author {
            login
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
    issues(first: 100, after: $issueCursor) {
      edges {
        node {
          createdAt
          author {
            login
          }
          comments(first: 100) {
            edges {
              node {
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
    pullRequests(first: 100, after: $prCursor) {
      edges {
        node {
          createdAt
          author {
            login
          }
          comments(first: 100) {
            edges {
              node {
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

def get_default_branch(owner, repo):
    branch_query = """
    query ($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) {
        defaultBranchRef {
          name
        }
      }
    }
    """
    variables = {"owner": owner, "repo": repo}
    response = requests.post(
        GRAPHQL_API_URL, json={"query": branch_query, "variables": variables}, headers=headers
    )
    if response.status_code == 200:
        data = response.json().get("data", {}).get("repository", {}).get("defaultBranchRef")
        return data["name"] if data else None
    else:
        print(f"Failed to request default branch: {response.json()}")
        return None


def fetch_events(owner, repo, repo_idx, total_repos):
    ref_name = get_default_branch(owner, repo)
    if not ref_name:
        print(f"Event collection is stopped because the base branch could not be fetched.")
        return []

    after_cursor = {"commitCursor": None, "commentCursor": None, "issueCursor": None, "prCursor": None}
    events = []

    # Number of processed per event
    event_counts = {
        "CommitEvent": 0,
        "CommitCommentEvent": 0,
        "IssueEvent": 0,
        "IssueCommentEvent": 0,
        "PullRequestEvent": 0,
        "PullRequestCommentEvent": 0
    }

    while True:
        print(f"[{repo_idx}/{total_repos}] {owner}/{repo} 처리 중... 현재 cursors 상태: {after_cursor}")
        
        variables = {"owner": owner, "repo": repo, "refName": ref_name, **after_cursor}
        response = requests.post(
            GRAPHQL_API_URL, json={"query": graphql_query, "variables": variables}, headers=headers
        )

        if response.status_code == 403:  # the request limit is reached
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait_time = reset_time - int(time.time())
            print(f"Request limit. Try again in {wait_time} seconds.")
            time.sleep(wait_time + 1)
            continue
        elif response.status_code != 200:
            print(f"GraphQL request failed: {response.json()}")
            break

        data = response.json().get("data", {}).get("repository")
        if not data:
            print(f"{owner}/{repo}:There is no data in the GraphQL response.")
            break

        # Commit Event 
        ref_data = data.get("ref", {}).get("target", {}).get("history", {}).get("edges", [])
        event_counts["CommitEvent"] += len(ref_data)
        for edge in ref_data:
            commit_node = edge["node"]
            events.append({
                "event_type": "CommitEvent",
                "user": commit_node["author"]["user"]["login"] if commit_node["author"] and commit_node["author"]["user"] else None,
                "created_at": commit_node["committedDate"]
            })

        # Commit Comment Event 
        comment_data = data.get("commitComments", {}).get("edges", [])
        event_counts["CommitCommentEvent"] += len(comment_data)
        for edge in comment_data:
            comment_node = edge["node"]
            events.append({
                "event_type": "CommitCommentEvent",
                "user": comment_node["author"]["login"] if comment_node["author"] else None,
                "created_at": comment_node["createdAt"]
            })

        # Issue Event 
        issue_data = data.get("issues", {}).get("edges", [])
        event_counts["IssueEvent"] += len(issue_data)
        for edge in issue_data:
            issue_node = edge["node"]
            events.append({
                "event_type": "IssueEvent",
                "user": issue_node["author"]["login"] if issue_node["author"] else None,
                "created_at": issue_node["createdAt"]
            })

            # Issue Comment 
            issue_comments = issue_node.get("comments", {}).get("edges", [])
            event_counts["IssueCommentEvent"] += len(issue_comments)
            for comment_edge in issue_comments:
                comment_node = comment_edge["node"]
                events.append({
                    "event_type": "IssueCommentEvent",
                    "user": comment_node["author"]["login"] if comment_node["author"] else None,
                    "created_at": comment_node["createdAt"]
                })

        # Pull Request Event 
        pr_data = data.get("pullRequests", {}).get("edges", [])
        event_counts["PullRequestEvent"] += len(pr_data)
        for edge in pr_data:
            pr_node = edge["node"]
            events.append({
                "event_type": "PullRequestEvent",
                "user": pr_node["author"]["login"] if pr_node["author"] else None,
                "created_at": pr_node["createdAt"]
            })

            # Pull Request Comment 
            pr_comments = pr_node.get("comments", {}).get("edges", [])
            event_counts["PullRequestCommentEvent"] += len(pr_comments)
            for comment_edge in pr_comments:
                comment_node = comment_edge["node"]
                events.append({
                    "event_type": "PullRequestCommentEvent",
                    "user": comment_node["author"]["login"] if comment_node["author"] else None,
                    "created_at": comment_node["createdAt"]
                })


        has_next_page = False
        if data.get("ref", {}).get("target", {}).get("history", {}).get("pageInfo", {}).get("hasNextPage"):
            after_cursor["commitCursor"] = data["ref"]["target"]["history"]["pageInfo"]["endCursor"]
            has_next_page = True
        for key in ["commentCursor", "issueCursor", "prCursor"]:
            if data.get(key.replace("Cursor", ""), {}).get("pageInfo", {}).get("hasNextPage"):
                after_cursor[key] = data[key.replace("Cursor", "")]["pageInfo"]["endCursor"]
                has_next_page = True

        print(f"[{repo_idx}/{total_repos}] {owner}/{repo} Event Processing Cumulative Status:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}개")

        if not has_next_page:
            break

    return events


def save_events_to_file(events, output_dir, repo_name, repo_idx):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{str(repo_idx).zfill(5)}_{repo_name.replace('/', '_')}_events.txt")
    with open(file_path, "w") as file:
        for event in events:
            file.write(f"{event['event_type']}, {event['user']}, {event['created_at']}\n")
    print(f"{repo_name} Saved the event data for: {file_path}")


def main():
    input_file = "./Repository_list/PyPi_repo_list_50000_v2.txt"
    output_dir = "./Repository_event_origin"

    start_index = START_INDEX
    end_index = END_INDEX

    # Check repositories already processed
    processed_repos = set(
        filename.split("_", 1)[-1].replace("_events.txt", "").replace("_", "/")  
        for filename in os.listdir(output_dir)
        if filename.endswith("_events.txt")  
    )

    with open(input_file, "r") as file:
        repositories = [line.split(",")[0].replace("https://github.com/", "").strip() for line in file]

    total_repos = len(repositories)

    # Check index range
    if start_index < 1 or end_index > total_repos or start_index > end_index:
        print(f"Error: ({start_index} ~ {end_index}) is not valid (1 ~ {total_repos})")
        return

    # Processing repositories within a range
    for target_index in range(start_index, end_index + 1):
        idx = target_index - 1  
        repo_path = repositories[idx]

        if repo_path in processed_repos:
            print(f"[{target_index}/{total_repos}] {repo_path}: Already processed. Skip.")
            continue

        try:
            owner, repo = repo_path.split("/")
        except ValueError:
            print(f"[{target_index}/{total_repos}] {repo_path}: The URL format is incorrect. Skipping.")
            continue

        print(f"[{target_index}/{total_repos}] {owner}/{repo} Retrieving event data...")
        try:
            events = fetch_events(owner, repo, target_index, total_repos)
            if events:
                save_events_to_file(events, output_dir, repo_path, target_index)
        except Exception as e:
            print(f"{owner}/{repo} An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
