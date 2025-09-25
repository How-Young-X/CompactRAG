import os
from tqdm import tqdm
import jsonlines
from utils.JudgeAnswer import is_answer_correct
from core.QASearch import FaissRetriever
from utils.json_parser import extract_model_cot_answer
from utils.JudgeAnswer import is_answer_correct
from prompt.DecomposeQuestion import DECOMPOSE_QUESTION
from prompt.GET_ANSWER import LLM_REASON_ANSWER_FROM_SUBQUESTIONS
from prompt.QA import QA_PROMPT
import time
from utils.list_parser import parse_model_output
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import json
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import jsonlines


extractor_model_path = "models/roberta-qa-finetuned"
extractor_tokenizer = None
extractor_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extractor_load_model():
    global extractor_tokenizer, extractor_model
    if extractor_tokenizer is None or extractor_model is None:
        extractor_tokenizer = RobertaTokenizerFast.from_pretrained(extractor_model_path)
        extractor_model = RobertaForQuestionAnswering.from_pretrained(extractor_model_path)
        extractor_model.to(_device)
        extractor_model.eval()

extractor_load_model()

def extractor(input_data):
    extractor_load_model()
    question = input_data.get("question", "")
    context = input_data.get("context", "")
    
    if not question or not context:
        return ""
    inputs = extractor_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    ).to(_device)
    
    with torch.no_grad():
        outputs = extractor_model(**inputs)
    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax()
    answer = extractor_tokenizer.decode(inputs["input_ids"][0][start:end+1])
    
    return answer.strip()

# 子问题重写模型
rewriter_model_name = "models/flan-t5-small-rewrite/final"
rewriter_tokenizer = None
rewriter_model = None
rewriter_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rewriter_max_input = 128
rewriter_max_output = 64

def rewriter_load_model():
    """内部函数，加载模型和tokenizer"""
    global rewriter_tokenizer, rewriter_model
    if rewriter_tokenizer is None or rewriter_model is None:
        rewriter_tokenizer = T5TokenizerFast.from_pretrained(rewriter_model_name)
        rewriter_model = T5ForConditionalGeneration.from_pretrained(rewriter_model_name)
        rewriter_model.to(_device)
        rewriter_model.eval()
rewriter_load_model()
def rewriter(input_text):
    rewriter_load_model()
    
    if not input_text or not input_text.strip():
        return ""
    encoded = rewriter_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=rewriter_max_input
    )
    encoded = {k: v.to(_device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = rewriter_model.generate(
            **encoded,
            max_length=rewriter_max_output,
            num_beams=4,
            early_stopping=True
        )
    return rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)



def get_qa_ablation_extractor_rewritor_test(input_path, output_path,benchmark, model,corpusfrom,backend,topk=5):
    with jsonlines.open(input_path, "r") as f:
        lines = list(f)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total = len(lines)
    correct = 0
    failed_questions = 0
    
    retriever = FaissRetriever()
    
    retriever.load(f"data/index/QA/{corpusfrom}/{benchmark}/corpus.index",
                       f"data/index/QA/{corpusfrom}/{benchmark}/corpus_meta.pkl",
                       f"data/index/QA/{corpusfrom}/{benchmark}/corpus_meta.db")
    pbar = tqdm(lines, desc="deal test", unit="question")
    with jsonlines.open(output_path, "w") as outfile:
        for i, line in enumerate(pbar):
            # 开始计时
            
            question = line["question"]
            gold_answer = line["answer"]
            reason = None
            reason_with_stats = None
            if backend == "vllm":
                from utils.VLLM import reason, reason_with_stats
                reason = reason
                reason_with_stats = reason_with_stats
            elif backend == "dashscope":
                from utils.Qwen import reason
                reason = reason
            
            # 统计信息
            total_input_tokens = 0
            total_output_tokens = 0
            step_stats = []
           
            decompose_input = DECOMPOSE_QUESTION.format(question=question)
            decompose_response = None
            sample_start_time = time.time()
            for attempt in range(3):
                try:
                    if backend == "vllm" and reason_with_stats:
                        # 使用带统计信息的函数
                        stats_result = reason_with_stats(model=model, prompt_=decompose_input, temperature=0)
                        decompose_response = stats_result["response"]
                        total_input_tokens += stats_result["input_tokens"]
                        total_output_tokens += stats_result["output_tokens"]
                        step_stats.append({
                            "step": "decompose",
                            "input_tokens": stats_result["input_tokens"],
                            "output_tokens": stats_result["output_tokens"]
                        })
                    else:
                        decompose_response = reason(model=model, prompt_=decompose_input,temperature=0)
                    break
                except Exception as e:
                    if attempt < 2:  
                        time.sleep(2)
                    else:
                        print(f"\nquestion {i+1} failed: {str(e)}")
                        failed_questions += 1
            subqestions = parse_model_output(decompose_response) if decompose_response else None
            recalls = []
            if subqestions and type(subqestions) is list:
                recall = {}
                for sub_index, subqestion in enumerate(subqestions):
                    sub_q = ""
                    try:
                        sub_q = subqestion["q"]
                    except:
                        break
                
                    results = retriever.search(f"{sub_q}", topk=topk)
                    recall["subquestion"] = sub_q
                    recall["retrieved"] = results
                    context = "\n\n".join([item["question"]+"\n"+item["answer"] for item in results])
                    context = context.strip()
                    subqestion["retrieved"] = context
                    recalls.append(recall)
            else:
                pass
            multi_question_sub_qestion_retrieved = question
            try:
                multi_question_sub_qestion_retrieved = question+"\n"+"\n".join([f"Sub-question {idx+1}: "+item["q"].strip()+"\n"+"Retrieved knowledge: \n"+item["retrieved"]  for idx, item in enumerate(subqestions)])
            except:
                pass
            prompt_multi_question_sub_qestion_retrieved = LLM_REASON_ANSWER_FROM_SUBQUESTIONS.format(input=multi_question_sub_qestion_retrieved)

            llm_response = None
            for attempt in range(3):
                try:
                    if backend == "vllm" and reason_with_stats:
                        # 使用带统计信息的函数
                        stats_result = reason_with_stats(model=model, prompt_=prompt_multi_question_sub_qestion_retrieved, temperature=0)
                        llm_response = stats_result["response"]
                        total_input_tokens += stats_result["input_tokens"]
                        total_output_tokens += stats_result["output_tokens"]
                        step_stats.append({
                            "step": "final_reasoning",
                            "input_tokens": stats_result["input_tokens"],
                            "output_tokens": stats_result["output_tokens"]
                        })
                    else:
                        # 使用原始函数（兼容其他backend）
                        llm_response = reason(model=model, prompt_=prompt_multi_question_sub_qestion_retrieved,temperature=0)
                    break 
                except Exception as e:
                    if attempt < 2:  
                        time.sleep(2)
                    else:
                        print(f"\nquestion {i+1} failed: {str(e)}")
                        failed_questions += 1
            
            # 结束计时
            sample_end_time = time.time()
            total_time_consumed = sample_end_time - sample_start_time

            pred_answer = extract_model_cot_answer(llm_response) if llm_response else None
            
            is_correct = is_answer_correct(pred_answer, gold_answer) if pred_answer else False
            if is_correct:
                correct += 1
            
            
            current_acc = correct / (i + 1 - failed_questions) * 100 if (i + 1 - failed_questions) > 0 else 0
            pbar.set_description(f"[acc: {current_acc:.2f}%]")
            
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "sub":subqestions,
                "recalls":recalls,
                "model_input": prompt_multi_question_sub_qestion_retrieved,
                "model_response": llm_response,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_time_consumed": total_time_consumed,
                "step_stats": step_stats
            }
            outfile.write(result)
            if not is_correct:
                with jsonlines.open(f"data/results/failed/{benchmark}.jsonl","a") as fa:
                    fa.write(result)
    
    final_acc = correct / (total - failed_questions) * 100 if (total - failed_questions) > 0 else 0
    
    print(f"\ndone!")
    print(f"model: {model}")
    print(f"total: {total}")
    print(f"success: {total - failed_questions}")
    print(f"fail: {failed_questions}")
    print(f"right: {correct}")
    print(f"acc: {final_acc:.2f}%")

    r = f"""
    model: {model}
    total: {total}
    success: {total - failed_questions}
    fail: {failed_questions}
    right: {correct}
    acc: {final_acc}
    """
    with open("data/results/result","a",encoding="utf8") as f:
        f.write(f"\n***ablation****{model}****{benchmark}**corpusfrom:{corpusfrom}*QA**top{topk}********\n")
        f.write(r)
    
    return final_acc


