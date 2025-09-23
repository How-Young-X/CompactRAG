import torch
import json
import random
import inspect
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    TrainingArguments,
    Trainer
)

# ========= 1. 加载数据 =========
dataset = load_dataset("json", data_files={"train": "train/data/qa/train.json", "validation": "train/data/qa/valid.json"})

# ========= 2. 加载Tokenizer和模型 =========
model_name = "models/FacebookAI/roberta-base/"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForQuestionAnswering.from_pretrained(model_name)

# ========= 3. 数据预处理 =========
max_length = 384   # 最大输入长度
doc_stride = 128   # 滑动窗口大小

def prepare_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


tokenized_datasets = dataset.map(
    prepare_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ========= 4. 训练配置 =========
def make_training_args(output_dir="models/roberta-qa-checkpoints"):
    base_kwargs = dict(
        output_dir=output_dir,
        eval_steps=500,
        save_steps=500,
        learning_rate=3e-5,
        eval_strategy = "steps",
        num_train_epochs=3,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=2,
        logging_steps=100,
        bf16=True
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",   # 或者 "f1"/"exact_match"，取决于你定义的 compute_metrics
        # greater_is_better=False,
    )
    # params = inspect.signature(TrainingArguments.__init__).parameters
    # if "evaluation_strategy" in params:
    #     base_kwargs["evaluation_strategy"] = "steps"
    # if "fp16" in params:
    #     base_kwargs["fp16"] = True
    return TrainingArguments(**base_kwargs)


training_args = make_training_args()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# ========= 5. 开始训练 =========
trainer.train()

# ========= 6. 保存模型 =========
save_dir = "models/roberta-qa-finetuned"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"模型已保存到 {save_dir}")
