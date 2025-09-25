from prompt.SelfAsk import (
    SELF_ASK_PROMPT_HOTPOTQA, SELF_ASK_PROMPT_MUSIEUQ, SELF_ASK_PROMPT_WIKIMQA,
    DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA, DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE, DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA
)
import os
from tqdm import tqdm
import jsonlines
from utils.JudgeAnswer import is_answer_correct
from core.PassageIndexSearch import FaissRetriever
import re
import time


def extract_question(text: str) -> str:
   
    matches = re.findall(r"Follow up:\s*(.*)", text, flags=re.I)
    if matches:
        return matches[-1].strip().rstrip(".?")
    return ""


def extract_answer(text: str) -> str:
   
    matches = re.findall(r"(?:So the final answer is:|The final answer is:)\s*(.*)", text, flags=re.I)
    if matches:
        ans = matches[-1].strip()
        return ans.rstrip(".")
    
    matches = re.findall(r"Intermediate answer:\s*(.*)", text, flags=re.I)
    if matches:
        ans = matches[-1].strip()
        return ans.rstrip(".")
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        return last.rstrip(".")
    
    return ""


def get_selfask_test(
   input_path, 
   output_path,
   benchmark, 
   model,
   backend,
   topk: int = 10,
   max_iter: int = 5,
):
    retriever = FaissRetriever()
    retriever.load(f"data/index/passage/{benchmark}/corpus.index",
                   f"data/index/passage/{benchmark}/corpus_meta.pkl",
                   f"data/index/passage/{benchmark}/corpus_meta.db")

    # backend
    if backend == "vllm":
        from utils.VLLM import reason, reason_with_stats
    elif backend == "dashscope":
        from utils.Qwen import reason

    def _normalize_knowledge_list(kl):
        norm = []
        for x in kl or []:
            if isinstance(x, dict):
                norm.append(str(x.get("text", "")))
            else:
                norm.append(str(x))
        return [t for t in norm if t.strip()]

    def _parse_json_answer(text):
        import re, json
        try:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "answer" in obj:
                    return str(obj["answer"]).strip()
        except Exception:
            pass
        m = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        if m:
            return m.group(1).strip()
        last = text.strip().split("\n")[-1].strip()
        if last.endswith("."):
            last = last[:-1]
        return last

    def _call_model(prompt, collect_stats=False):
        if backend == "vllm" and reason_with_stats and collect_stats:
            stats_result = reason_with_stats(model=model, prompt_=prompt, temperature=0)
            return stats_result["response"] or "", stats_result
        else:
            return reason(model=model, prompt_=prompt, temperature=0) or "", None

    def _retrieve_direct_answer(fu_question, collect_stats=False):
        try:
            knowledge_list = retriever.search(fu_question, topk) or []
        except Exception as e:
            print(f"[retriever error] {e}")
            knowledge_list = []
        knowledges_texts = _normalize_knowledge_list(knowledge_list)
        knowledge_block = "\n".join(knowledges_texts)

        if benchmark == "musique":
            template = DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE
        elif benchmark == "2wiki":
            template = DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA
        elif benchmark == "hotpotqa":
            template = DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA
        else:
            template = "{knowledge}\n\nQuestion: {question}\nAnswer:"

        retrieve_prompt = template.format(
            knowledge=knowledge_block, question=fu_question
        )
        retrieve_resp, stats = _call_model(retrieve_prompt, collect_stats=collect_stats)
        mid_answer = _parse_json_answer(retrieve_resp) or "unknown"
        return mid_answer, knowledge_list, stats

    with jsonlines.open(input_path, "r") as f:
        lines = list(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total, correct, failed_questions = len(lines), 0, 0
    pbar = tqdm(lines, desc="deal question", unit="question")

    with jsonlines.open(output_path, "w") as outfile:
        for i, line in enumerate(pbar):
            # 开始计时
            
            question, gold_answer = line["question"], line["answer"]

            # init prompt
            cur_prompt = ""
            if benchmark == "musique":
                cur_prompt = SELF_ASK_PROMPT_MUSIEUQ.format(question=question)
            elif benchmark == "2wiki":
                cur_prompt = SELF_ASK_PROMPT_WIKIMQA.format(question=question)
            elif benchmark == "hotpotqa":
                cur_prompt = SELF_ASK_PROMPT_HOTPOTQA.format(question=question)
            else:
                cur_prompt = f"Solve the question with self-ask:\nQuestion: {question}"

            trace_parts, internal_questions, collected_knowledges = [], [], []
            final_answer = None
            
            # 统计信息
            total_input_tokens = 0
            total_output_tokens = 0
            step_stats = []
            sample_start_time = time.time()
            try:
                for step in range(max_iter):
                    resp, stats = _call_model(cur_prompt, collect_stats=True)
                    resp = resp.strip()
                    
                    # 收集统计信息
                    if stats:
                        total_input_tokens += stats["input_tokens"]
                        total_output_tokens += stats["output_tokens"]
                        step_stats.append({
                            "step": step + 1,
                            "type": "main_reasoning",
                            "input_tokens": stats["input_tokens"],
                            "output_tokens": stats["output_tokens"]
                        })
                    
                    if step == 0:
                        resp = resp.strip().split("\n")[1]
                    else:
                        resp = resp.strip().split("\n")[0]
                    # print("*"*15)
                    # print("cur_prompt:")
                    # print(cur_prompt)
                    # print("resp:\n")
                    # print(resp)
                    # print("*"*15)
                    # exit()

                    trace_parts.append(resp)

                    # case 1: final answer
                    if resp.startswith("So the final answer is:") or resp.startswith("The final answer is:"):
                        final_answer = extract_answer(resp)
                        break

                    # case 2: follow up
                    if  resp.startswith("Follow up"):
                        fu_q = extract_question(resp)
                        internal_questions.append(fu_q)

                        mid_ans, knowledge_list, retrieve_stats = _retrieve_direct_answer(fu_q, collect_stats=True)
                        collected_knowledges.extend(knowledge_list)
                        
                        # 收集检索统计信息
                        if retrieve_stats:
                            total_input_tokens += retrieve_stats["input_tokens"]
                            total_output_tokens += retrieve_stats["output_tokens"]
                            step_stats.append({
                                "step": step + 1,
                                "type": "retrieval",
                                "input_tokens": retrieve_stats["input_tokens"],
                                "output_tokens": retrieve_stats["output_tokens"]
                            })

                        cur_prompt += (
                            f"\n{resp}\nIntermediate answer: {mid_ans}."
                        )
                        continue

                    # case 3: model indecisive → force final answer
                    if "Are follow up questions needed here:" in resp and "No" in resp:
                        cur_prompt += "\nSo the final answer is:"
                        continue

                    # fallback
                    if step == max_iter - 1 and final_answer is None:
                        cur_prompt += "\nSo the final answer is:"
                if final_answer is None:
                    last_resp, last_stats = _call_model(cur_prompt, collect_stats=True)
                    trace_parts.append(last_resp)
                    final_answer = extract_answer(last_resp)
                    
                    # 收集最后一步的统计信息
                    if last_stats:
                        total_input_tokens += last_stats["input_tokens"]
                        total_output_tokens += last_stats["output_tokens"]
                        step_stats.append({
                            "step": len(step_stats) + 1,
                            "type": "final_answer",
                            "input_tokens": last_stats["input_tokens"],
                            "output_tokens": last_stats["output_tokens"]
                        })
                
                # 结束计时
                sample_end_time = time.time()
                total_time_consumed = sample_end_time - sample_start_time

                pred_answer = final_answer

            except Exception as e:
                print(f"\nquestion {i+1} failed: {str(e)}")
                failed_questions += 1
                pred_answer = None

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
                "model_input": cur_prompt,
                "model_response": trace_parts[-1] if trace_parts else "",
                "trace": "\n".join(trace_parts),
                "internal_questions": internal_questions,
                "knowledges": collected_knowledges,
                # 添加统计信息
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_time_consumed": total_time_consumed,
                "step_stats": step_stats
            }
            outfile.write(result)

    final_acc = correct / (total - failed_questions) * 100 if (total - failed_questions) > 0 else 0
    print(f"\ndone! acc={final_acc:.2f}%")
    r = f"""
    model: {model}
    total: {total}
    success: {total - failed_questions}
    fail: {failed_questions}
    right: {correct}
    acc: {final_acc}
    """
    with open("data/results/result","a",encoding="utf8") as f:
        f.write(f"\n*******{model}****{benchmark}***selfask**********\n")
        f.write(r)
    
    
    return final_acc
