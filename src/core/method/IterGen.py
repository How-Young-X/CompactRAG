from prompt.IterRetGen import  ITER_RETGEN_MUSIQUE_PROMPT, ITER_RETGEN_HOTPOTQA_PROMPT, ITER_RETGEN_WIKIMQA_PROMPT
import os
from tqdm import tqdm
import jsonlines
from utils.JudgeAnswer import is_answer_correct
from core.PassageIndexSearch import FaissRetriever
from utils.json_parser import extract_model_cot_answer, extract_model_cot_thought
from utils.JudgeAnswer import is_answer_correct
import time

def normalize_knowledge_list(kl):
        norm = []
        for x in kl or []:
            if isinstance(x, dict):
                norm.append(str(x.get("text", "")))
            else:
                norm.append(str(x))
        return [t for t in norm if t.strip()]


def get_itergen_test(input_path, output_path, benchmark, model,backend, iter_=2):
    with jsonlines.open(input_path, "r") as f:
        lines = list(f)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total = len(lines)
    correct = 0
    failed_questions = 0
    
    retriever = FaissRetriever()

    retriever.load(f"data/index/passage/{benchmark}/corpus.index",
                       f"data/index/passage/{benchmark}/corpus_meta.pkl",
                       f"data/index/passage/{benchmark}/corpus_meta.db")


    pbar = tqdm(lines, desc="deal test", unit="question")
    with jsonlines.open(output_path, "w") as outfile:
        for i, line in enumerate(pbar):
            question = line["question"]
            gold_answer = line["answer"]
            # passage_list = retriever.search(question,topk=10)
            # passage_list = normalize_knowledge_list(passage_list)
            # passages = "\n\n".join(passage_list)
            # print(passage)

            prompt_template = None
            if benchmark == "musique":
                prompt_template = ITER_RETGEN_MUSIQUE_PROMPT
            elif benchmark == "2wiki":
                prompt_template = ITER_RETGEN_WIKIMQA_PROMPT
            elif benchmark == "hotpotqa":
                prompt_template = ITER_RETGEN_HOTPOTQA_PROMPT
           
            
            llm_response = None
            reason = None
            if backend == "vllm":
                from utils.VLLM import reason
                reason = reason
            elif backend == "dashscope":
                from utils.Qwen import reason
                reason = reason

            response_str = ""
            model_iter = []
            for call in range(iter_):
                retrieve_content = question + response_str
                passage_list = retriever.search(retrieve_content,topk=10)
                passage_list = normalize_knowledge_list(passage_list)
                passages = "\n\n".join(passage_list)
                input_ = prompt_template.format(documents=passages, question = question)
                for attempt in range(3):
                    try:
                        llm_response = reason(model=model, prompt_=input_,temperature=0)
                        pred_answer = extract_model_cot_answer(llm_response) if llm_response else ""
                        pred_thought = extract_model_cot_thought(llm_response) if llm_response else ""
                        response_str = str(pred_thought) + "So the answer is: " +str(pred_answer)
                        
                        model_iter.append(response_str)
                        break 
                    except Exception as e:
                        if attempt < 2:  
                            time.sleep(2)
                        else:
                            print(f"\nquestion {i+1} failed: {str(e)}")
                            failed_questions += 1
            
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
                "model_iter": model_iter,
                "model_input": input_,
                "model_response": llm_response
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

    r = f"""
    model: {model}
    total: {total}
    success: {total - failed_questions}
    fail: {failed_questions}
    right: {correct}
    acc: {final_acc}
    """
    with open("data/results/result","a",encoding="utf8") as f:
        f.write(f"\n*******{model}****{benchmark}***iter-gen{iter_}**********\n")
        f.write(r)
    
    return final_acc


