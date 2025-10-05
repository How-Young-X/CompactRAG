import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
import jsonlines
import json
from tqdm import tqdm
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""
    
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    # Handle None values
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score between prediction and ground truth."""
    # Handle None values
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    
    return normalize_answer(prediction) == normalize_answer(ground_truth)

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
    
    output_path = input_path.replace("data/results","data/f1_eval")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    total_samples = 0
    total_f1 = 0.0
    total_em = 0.0
    results = []
    
    with jsonlines.open(input_path) as reader:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in reader)
    
    with jsonlines.open(input_path) as reader:
        for line in tqdm(reader, total=total_lines, desc="Calculating F1 scores"):
            question = line["question"]
            prediction = line["pred_answer"]
            answer = line["gold_answer"]
            
            # Calculate F1 and EM scores
            f1 = f1_score(prediction, answer)
            em = exact_match_score(prediction, answer)
            
            total_f1 += f1
            total_em += em
            total_samples += 1
            
            # Store result
            result = {
                "question": question,
                "prediction": prediction,
                "answer": answer,
                "f1_score": f1,
                "exact_match": em
            }
            results.append(result)
    
    # Calculate average scores
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    avg_em = total_em / total_samples if total_samples > 0 else 0.0
    
    # Save results to jsonl
    with jsonlines.open(output_path, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    # Print summary
    print(f"\nF1 Evaluation completed!")
    print(f"Total samples: {total_samples}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Exact Match: {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"Results saved to: {output_path}")

    with open(output_path.replace(".jsonl","_result.txt"),"a") as f:
        f.write(f"\n{args.benchmark}_{args.model}_{args.method}_corpus_{args.corpusfrom}_top{args.topk}********\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average Exact Match: {avg_em:.4f} ({avg_em*100:.2f}%)\n")
        f.write(f"Results saved to: {output_path}\n")
