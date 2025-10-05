import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.LLM import reason
from prompt.GET_ANSWER import LLM_EVAL_PROMPT
from argparse import ArgumentParser
import jsonlines
import json
from tqdm import tqdm

def llm_eval(question, prediction, answer):
    prompt = LLM_EVAL_PROMPT.format(question=question, prediction=prediction, answer=answer)
    response = reason(model="qwen3-32b",prompt_=prompt,temperature=0)
    return response

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--corpusfrom", type=str)
    parser.add_argument("--topk", type=int)
    
    args = parser.parse_args()
    if args.method == "qa":
        input_path = f"data/results/{args.benchmark}_{args.model}_{args.method}_corpus_{args.corpusfrom}_top{args.topk}_evaluation_results.jsonl"
    else:
        input_path = f"data/results/{args.benchmark}_{args.model}_{args.method}_evaluation_results.jsonl"
    
    output_path = input_path.replace("data/results","data/llm_eval")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    total_samples = 0
    correct_samples = 0
    results = []
    
    with jsonlines.open(input_path) as reader:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in reader)
    
    with jsonlines.open(input_path) as reader:
        for line in tqdm(reader, total=total_lines, desc="Evaluating"):
            question = line["question"]
            prediction = line["pred_answer"]
            answer = line["gold_answer"]
            response = llm_eval(question, prediction, answer)
            
            # Parse response to determine if correct (assuming response contains "Correct" or "Incorrect")
            is_correct = "yes" in response.lower()
            if is_correct:
                correct_samples += 1
            total_samples += 1
            
            # Store result
            result = {
                "question": question,
                "prediction": prediction,
                "answer": answer,
                "llm_eval_response": response,
                "is_correct": is_correct
            }
            results.append(result)
    
    # Calculate accuracy
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    
    # Save results to jsonl
    with jsonlines.open(output_path, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Results saved to: {output_path}")

    with open(output_path.replace(".jsonl","_result.txt"),"a") as f:
        f.write(f"\n{args.benchmark}_{args.model}_{args.method}_corpus_{args.corpusfrom}_top{args.topk}********\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Correct samples: {correct_samples}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Results saved to: {output_path}\n")