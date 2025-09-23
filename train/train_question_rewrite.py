import json
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch

# ========= 1. 加载数据 =========
dataset = load_dataset(
    "json",
    data_files={"train": "train/data/rewrite/train.json", "validation": "train/data/rewrite/valid.json"}
)

# ========= 2. 模型和 tokenizer =========
model_name = "models/google/flan-t5-small"  # 也可换成 flan-t5-base/mini
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

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

# ========= 4. 训练参数 =========
training_args = Seq2SeqTrainingArguments(
    output_dir="models/flan-t5-small-rewrite",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=5e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_steps=100,
    push_to_hub=False,
    report_to="none",  # 关闭 wandb 等日志
)

# ========= 5. Trainer =========
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
)

# ========= 6. 开始训练 =========
trainer.train()

# ========= 7. 保存模型 =========
trainer.save_model("models/flan-t5-small-rewrite/final")
tokenizer.save_pretrained("models/flan-t5-small-rewrite/final")
