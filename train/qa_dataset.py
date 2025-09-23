import jsonlines
import json
import random
import re

def find_answer_start(context, answer_text):
    """在context中查找答案的起始位置"""
    # 精确匹配
    start_pos = context.find(answer_text)
    if start_pos != -1:
        return start_pos
    
    # 忽略大小写
    lower_context = context.lower()
    lower_answer = answer_text.lower()
    start_pos = lower_context.find(lower_answer)
    if start_pos != -1:
        return start_pos
    
    # 去掉标点符号
    clean_answer = re.sub(r'[^\w\s]', '', answer_text).strip()
    clean_context = re.sub(r'[^\w\s]', '', context)
    start_pos = clean_context.find(clean_answer)
    if start_pos != -1:
        return context.find(clean_answer)
    
    # 匹配答案的前几个词
    answer_words = clean_answer.split()
    if len(answer_words) > 1:
        for i in range(len(answer_words), 0, -1):
            partial_answer = ' '.join(answer_words[:i])
            start_pos = clean_context.find(partial_answer)
            if start_pos != -1:
                return context.find(partial_answer)
    
    # 单词匹配
    if len(answer_words) == 1:
        word = answer_words[0]
        start_pos = clean_context.find(word)
        if start_pos != -1:
            return context.find(word)
    
    return -1


result = []
idx = 0
processed_count = 0
failed_count = 0

with jsonlines.open("data/qa_synthesis.jsonl", "r") as f:
    for line in f:
        json_data = line["parsed_json"]
        if json_data and json_data.get("qa_list"):
            processed_count += 1
            
            # 构建 context
            qa_pairs = []
            for qa in json_data["qa_list"]:
                q = qa.get("q", "")
                a = qa.get("a", "")
                if q and a:
                    qa_pairs.append(f"{q}\n{a}")
            
            if not qa_pairs:
                print(f"Warning: No valid qa pairs for question: {json_data.get('question', 'Unknown')}")
                failed_count += 1
                continue
            
            context = "\n\n".join(qa_pairs)
            
            question = json_data.get("question", "")
            answer = json_data.get("answer", "")
            answer_from = json_data.get("answer_from", "")
            
            if not question or not answer or not answer_from:
                print(f"Warning: Missing fields for question: {question}")
                failed_count += 1
                continue
            
            # 找到答案来源的 qa
            answer_qa = None
            for qa in json_data["qa_list"]:
                if qa["id"] == answer_from:
                    answer_qa = qa
                    break
            
            if answer_qa is None:
                print(f"Warning: Could not find qa id '{answer_from}' for question: {question}")
                failed_count += 1
                continue
            
            answer_text = answer_qa["a"]
            answer_start_in_qa = find_answer_start(answer_text, answer)
            if answer_start_in_qa == -1:
                print(f"Warning: Could not find answer '{answer}' in '{answer_text}' for question: {question}")
                failed_count += 1
                continue
            
            # 计算答案在整个 context 中的位置
            for qa in json_data["qa_list"]:
                if qa["id"] == answer_from:
                    qa_text = f"{qa.get('q','')}\n{qa.get('a','')}"
                    answer_start = context.find(qa_text) + len(qa.get("q","")) + 1 + answer_start_in_qa
                    break
            
            result.append({
                "id": str(idx),
                "question": question,
                "context": context,
                "answers": {
                    "text": [answer],
                    "answer_start": [answer_start]
                }
            })
            idx += 1

# -------------------------
#  7:2:1 划分 train/valid/train
# -------------------------
total = len(result)
train_size = int(total * 0.7)
valid_size = int(total * 0.2)

# 打乱数据
random.shuffle(result)

train_data = result[:train_size]
valid_data = result[train_size:train_size + valid_size]
test_data  = result[train_size + valid_size:]

with open("data/qa/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("data/qa/valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open("data/qa/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("转换完成！")
print(f"总共处理了 {processed_count} 条有效数据")
print(f"成功转换了 {len(result)} 条数据")
print(f"失败 {failed_count} 条数据")
print(f"训练集: {len(train_data)} 条, 验证集: {len(valid_data)} 条, 测试集: {len(test_data)} 条")
print("结果已保存到: data/train.json, data/valid.json, data/test.json")