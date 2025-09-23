import json
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

# ========= 1. 加载数据 =========
dataset = load_dataset(
    "json",
    data_files={"test": "data/rewrite/test.json"}
)

# ========= 2. 模型和 tokenizer =========
model_name = "flan-t5-small-rewrite/final"  # 也可换成 flan-t5-base/mini
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========= 3. 数据预处理 =========
max_input = 128
max_output = 64

def preprocess(batch):
    inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=max_input,
    )
    labels = tokenizer(
        batch["output"],
        truncation=True,
        padding="max_length",
        max_length=max_output,
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

# ========= 4. 批量推理函数 =========
def batch_rewrite_subquestions(input_texts, model, tokenizer, max_input=128, max_output=64, batch_size=16):
    model.eval()
    all_predictions = []
    
    # 分批处理
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        
        # 编码输入
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_output,
                num_beams=4,
                early_stopping=True
            )
        
        # 解码输出
        predictions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        all_predictions.extend(predictions)
    
    return all_predictions

# ========= 5. 计算评估指标 =========
def calculate_metrics(references, predictions):
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 计算ROUGE分数
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # 计算BLEU分数
    # 准备BLEU计算所需的格式
    refs_bleu = [[ref.split()] for ref in references]
    preds_bleu = [pred.split() for pred in predictions]
    
    # 计算语料级BLEU
    smooth_fn = SmoothingFunction().method1
    bleu_score = corpus_bleu(refs_bleu, preds_bleu, smoothing_function=smooth_fn)
    
    # 计算句子级BLEU并取平均
    sentence_bleu_scores = []
    for ref, pred in zip(refs_bleu, preds_bleu):
        try:
            score = sentence_bleu(ref, pred, smoothing_function=smooth_fn)
            sentence_bleu_scores.append(score)
        except:
            # 如果计算失败（如空字符串），跳过
            continue
    
    # 返回结果
    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
        "corpus_bleu": bleu_score,
        "sentence_bleu": np.mean(sentence_bleu_scores) if sentence_bleu_scores else 0
    }

# ========= 6. 批量推理验证并计算指标 =========
# 获取验证集的前100个样本
test_samples = dataset["test"]
inputs = test_samples["input"]
expected_outputs = test_samples["output"]

# 批量推理
predictions = batch_rewrite_subquestions(inputs, model, tokenizer)

# 计算评估指标
metrics = calculate_metrics(expected_outputs, predictions)



# 打印部分样本的详细结果
print("\n部分样本详细结果:")
for i, (input_text, expected, predicted) in enumerate(zip(inputs[:5], expected_outputs[:5], predictions[:5])):
    print("*" * 15)
    print(f"样本 {i+1}:")
    print("输入:", input_text)
    print("期望输出:", expected)
    print("预测输出:", predicted)
    print("*" * 15)

# 打印指标结果
print("=" * 50)
print("评估指标结果:")
print(f"ROUGE-1: {metrics['rouge1']:.4f}")
print(f"ROUGE-2: {metrics['rouge2']:.4f}")
print(f"ROUGE-L: {metrics['rougeL']:.4f}")
print(f"语料级BLEU: {metrics['corpus_bleu']:.4f}")
print(f"句子级BLEU平均: {metrics['sentence_bleu']:.4f}")
print("=" * 50)