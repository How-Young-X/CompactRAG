import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering

# ========= 1. 加载模型和数据 =========
model_path = "models/roberta-qa-finetuned"
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForQuestionAnswering.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

dataset = load_dataset("json", data_files={"test": "train/data/qa/test.json"})

# ========= 2. 推理函数 =========
def answer_question(question, context):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax()
    return tokenizer.decode(inputs["input_ids"][0][start:end+1])

# ========= 3. 在验证集上测试 =========
correct = 0
total = len(dataset["test"])

for sample in dataset["test"]:
    pred = answer_question(sample["question"], sample["context"]).strip()
    gold = sample["answers"]["text"][0].strip()
    
    # print("*"*10+"pred"+"*"*10)
    # print(pred)
    # print("*"*25)
    # print("*"*10+"gold"+"*"*10)
    # print(gold)
    # print("*"*25)

    
    if pred.upper().strip() == gold.upper().strip():
        correct += 1

acc = correct / total
print(f"验证集准确率: {acc:.2%} ({correct}/{total})")
