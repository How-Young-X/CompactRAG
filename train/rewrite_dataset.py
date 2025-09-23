import jsonlines
import json
import random
import re



result = []
idx = 0
processed_count = 0
failed_count = 0

with jsonlines.open("data/rewrite_synthesis.jsonl", "r") as f:
    for line in f:
        json_data = line["parsed_json"]
        if json_data and json_data.get("input") and json_data.get("output"):
            answer = json_data.get("input").split("prev_answer:")[-1].strip()
            if answer.upper() in json_data.get("output").upper():
                result.append({
                    "id": str(idx),
                    "input": json_data.get("input"),
                    "output": json_data.get("output"),
                })
                idx += 1

total = len(result)
train_size = int(total * 0.7)
valid_size = int(total * 0.2)

# 打乱数据
random.shuffle(result)

train_data = result[:train_size]
valid_data = result[train_size:train_size + valid_size]
test_data  = result[train_size + valid_size:]


with open("data/rewrite/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("data/rewrite/valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)
with open("data/rewrite/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)