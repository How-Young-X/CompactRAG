"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) 推理方法

基于IRCOT论文实现，交替进行检索和推理，逐步构建答案。
严格按照原始IRCOT项目的实现方式，包括prompt格式、推理流程和答案提取。
"""

import os
import re
import json
import time
from tqdm import tqdm
import jsonlines
from utils.JudgeAnswer import is_answer_correct
from core.PassageIndexSearch import FaissRetriever
from utils.json_parser import extract_model_cot_answer, extract_model_cot_thought
from prompt.IRCoT import IRCOT_HOTPOTQA_PROMPT, IRCOT_MUSIQUE_PROMPT, IRCOT_2WIKI_PROMPT
import random
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# 设置随机种子
random.seed(100)

def normalize_knowledge_list(kl):
    norm = []
    for x in kl or []:
        if isinstance(x, dict):
            norm.append(str(x.get("text", "")))
        else:
            norm.append(str(x))
    return [t for t in norm if t.strip()]

def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True
    
    regex = re.compile(r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    return bool(re.match(regex, sentence))

def remove_reasoning_sentences(sentences):
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]

def para_to_text(title: str, para: str, max_num_words: int) -> str:

    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para


def get_ircot_test(input_path, output_path, benchmark, model, backend, max_iter=5, retrieval_count=5):
    """
    IRCoT推理测试函数 - 严格按照原始IRCOT实现
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        benchmark: 数据集名称 (musique, 2wiki, hotpotqa)
        model: 模型名称
        backend: 后端类型 (vllm, dashscope)
        max_iter: 最大迭代次数
        retrieval_count: 每次检索的文档数量
    """
    
    # 加载数据
    with jsonlines.open(input_path, "r") as f:
        lines = list(f)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total = len(lines)
    correct = 0
    failed_questions = 0
    
    # 初始化检索器
    retriever = FaissRetriever()
    retriever.load(f"data/index/passage/{benchmark}/corpus.index",
                   f"data/index/passage/{benchmark}/corpus_meta.pkl",
                   f"data/index/passage/{benchmark}/corpus_meta.db")
    
    # 初始化模型调用函数
    reason = None
    reason_with_stats = None
    if backend == "vllm":
        from utils.VLLM import reason, reason_with_stats
        reason = reason
        reason_with_stats = reason_with_stats
    elif backend == "dashscope":
        from utils.Qwen import reason
        reason = reason
    
    # 选择IRCOT prompt模板
    if benchmark == "hotpotqa":
        prompt_template = IRCOT_HOTPOTQA_PROMPT
    elif benchmark == "musique":
        prompt_template = IRCOT_MUSIQUE_PROMPT
    elif benchmark == "2wiki":
        prompt_template = IRCOT_2WIKI_PROMPT
    else:
        # 默认使用HotpotQA的prompt
        prompt_template = IRCOT_HOTPOTQA_PROMPT
    
    # 初始化spacy用于句子分割
    nlp = None
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: en_core_web_sm not found. Using simple sentence splitting.")
            nlp = None
    else:
        print("Warning: spacy not available. Using simple sentence splitting.")
    
    pbar = tqdm(lines, desc="deal test", unit="question")
    with jsonlines.open(output_path, "w") as outfile:
        for i, line in enumerate(pbar):
            # 开始计时
            sample_start_time = time.time()
            
            question = line["question"]
            gold_answer = line["answer"]
            
            # 初始化IRCOT状态
            collected_titles = []
            collected_passages = []
            generated_sentences = []
            total_input_tokens = 0
            total_output_tokens = 0
            step_stats = []
            
            try:
                # IRCoT主循环：交替检索和推理
                for iteration in range(max_iter):
                    # 步骤1：检索相关文档
                    if iteration == 0:
                        # 第一次检索使用原始问题
                        query = question
                    else:
                        # 后续检索使用生成的句子（去除推理句子）
                        filtered_sentences = remove_reasoning_sentences(generated_sentences)
                        if filtered_sentences:
                            query = filtered_sentences[-1]
                        else:
                            query = question
                    
                    # 执行检索
                    passage_list = retriever.search(query, topk=retrieval_count)
                    passage_list = normalize_knowledge_list(passage_list)
                    
                    # 去重并添加到收集的文档中
                    for passage in passage_list:
                        if passage not in collected_passages:
                            collected_passages.append(passage)
                            # 提取标题（取第一行作为标题）
                            title = passage.split('\n')[0] if '\n' in passage else passage[:50]
                            collected_titles.append(title)
                    
                    # 步骤2：基于收集的文档进行推理
                    # 构建context，使用IRCOT的格式
                    context_parts = []
                    for title, passage in zip(collected_titles, collected_passages):
                        context_parts.append(para_to_text(title, passage, 350))
                    context = "\n\n".join(context_parts)
                    
                    # 构建当前生成的句子
                    generation_so_far = " ".join(generated_sentences)
                    
                    # 构建prompt
                    prompt = prompt_template.format(
                        context=context,
                        question=question,
                        generation_so_far=generation_so_far
                    )
                    
                    # 调用模型进行推理
                    if backend == "vllm" and reason_with_stats:
                        # 使用带统计信息的函数
                        stats_result = reason_with_stats(model=model, prompt_=prompt, temperature=0)
                        response = stats_result["response"]
                        total_input_tokens += stats_result["input_tokens"]
                        total_output_tokens += stats_result["output_tokens"]
                        step_stats.append({
                            "iteration": iteration + 1,
                            "type": "reasoning",
                            "input_tokens": stats_result["input_tokens"],
                            "output_tokens": stats_result["output_tokens"]
                        })
                    else:
                        # 使用原始函数（兼容其他backend）
                        response = reason(model=model, prompt_=prompt, temperature=0)
                    
                    # 解析响应，提取新生成的句子
                    if response:
                        # 首先尝试从JSON中提取thought作为新句子
                        thought = extract_model_cot_thought(response)
                        if thought and thought not in generated_sentences:
                            generated_sentences.append(thought)
                        else:
                            # 如果无法提取thought，使用spacy进行句子分割
                            if nlp:
                                doc = nlp(response)
                                new_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                            else:
                                # 简单的句子分割
                                new_sentences = re.split(r'[.!?]+', response)
                                new_sentences = [s.strip() for s in new_sentences if s.strip()]
                            
                            # 只取第一个新句子（按照IRCOT的实现）
                            if new_sentences:
                                new_sentence = new_sentences[0]
                                if new_sentence and new_sentence not in generated_sentences:
                                    generated_sentences.append(new_sentence)
                    
                # 检查是否已经得到最终答案
                final_answer = None
                
                # 使用CoT的解析函数提取答案
                if 'response' in locals() and response:
                    final_answer = extract_model_cot_answer(response)
                
                # 如果从response中没提取到，尝试从generated_sentences中提取
                if not final_answer and generated_sentences:
                    for sentence in reversed(generated_sentences):
                        final_answer = extract_model_cot_answer(sentence)
                        if final_answer:
                            break
                
                pred_answer = final_answer
                
            except Exception as e:
                print(f"\nquestion {i+1} failed: {str(e)}")
                failed_questions += 1
                pred_answer = None
            
            # 结束计时
            sample_end_time = time.time()
            total_time_consumed = sample_end_time - sample_start_time
            
            # 评估答案
            is_correct = is_answer_correct(pred_answer, gold_answer) if pred_answer else False
            if is_correct:
                correct += 1
            
            current_acc = correct / (i + 1 - failed_questions) * 100 if (i + 1 - failed_questions) > 0 else 0
            pbar.set_description(f"[acc: {current_acc:.2f}%]")
            
            # 保存结果
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "collected_titles": collected_titles,
                "collected_passages": collected_passages,
                "generated_sentences": generated_sentences,
                "model_input": prompt if 'prompt' in locals() else "",
                "model_response": response if 'response' in locals() else "",
                # 添加统计信息
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_time_consumed": total_time_consumed,
                "step_stats": step_stats
            }
            outfile.write(result)
    
    final_acc = correct / (total - failed_questions) * 100 if (total - failed_questions) > 0 else 0
    
    print(f"\ndone!")
    print(f"model: {model}")
    print(f"total: {total}")
    print(f"success: {total - failed_questions}")
    print(f"fail: {failed_questions}")
    print(f"right: {correct}")
    print(f"acc: {final_acc:.2f}%")
    
    # 保存结果到汇总文件
    r = f"""
    model: {model}
    total: {total}
    success: {total - failed_questions}
    fail: {failed_questions}
    right: {correct}
    acc: {final_acc}
    """
    with open("data/results/result", "a", encoding="utf8") as f:
        f.write(f"\n*******{model}****{benchmark}***ircot**********\n")
        f.write(r)
    
    return final_acc