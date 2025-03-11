import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("data_preprocessing/data/b1_extract_largest_component_edge/v2020_lc.csv")  # 파일 경로를 적절히 변경하세요.

# 중복을 제외한 repo_index 개수 출력
unique_repo_count = df["repo_index"].nunique()

print(f"✅ 중복을 제외한 repo_index 개수: {unique_repo_count}")