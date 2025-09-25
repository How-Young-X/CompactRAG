import argparse
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.method.IterGen import get_itergen_test
from core.method.QARag import get_qa_test
from core.method.SelfAsk import get_selfask_test
from core.method.IRCoT import get_ircot_test
from core.method.QARagAblation import get_qa_ablation_rewritor_test
from core.method.QARagAblationOnlyDecompose import get_qa_ablation_extractor_rewritor_test

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="musique",choices=["musique","2wiki","hotpotqa"])
    parser.add_argument("--model", type=str, default="llama3b", help="Model name for vLLM")
    parser.add_argument("--method",type=str, default="itergen",choices=["direct","cot","selfask","naive","itergen","qa","ircot","qa_ablation_rewritor","qa_ablation_extractor_rewritor"], help="RAG method to use")
    parser.add_argument("--itergen",type=int, default=2, help="Number of iterations for itergen method")
    parser.add_argument("--corpusfrom",type=str, default="gpt-4", help="QA Corpus from which model")
    parser.add_argument("--topk",type=int, default=5, help="Number of top-k passages for retrieval")
    parser.add_argument("--backend",type=str, default="vllm",choices=["vllm"], help="Backend to use (only vllm supported)")

    
    args = parser.parse_args()
    
    # 使用示例
    model = args.model
    benchmark = args.benchmark
    method = args.method
    corpusfrom = args.corpusfrom
    backend = args.backend

    # 设置输入和输出路径
    input_path = f"data/sampled/{benchmark}_test.jsonl"
    
    # 根据方法设置不同的输出路径
    if method == "itergen":
        output_path = f"data/results/{benchmark}_{model}_{method}_iter{args.itergen}_evaluation_results.jsonl"
    elif method == "qa":
        output_path = f"data/results/{benchmark}_{model}_{method}_corpus_{corpusfrom}_evaluation_results.jsonl"
    elif method == "qa_ablation_rewritor":
        output_path = f"data/results/{benchmark}_{model}_{method}_corpus_{corpusfrom}_evaluation_results.jsonl"
    elif method == "qa_ablation_extractor_rewritor":
        output_path = f"data/results/{benchmark}_{model}_{method}_corpus_{corpusfrom}_evaluation_results.jsonl"
    elif method == "ircot":
        output_path = f"data/results/{benchmark}_{model}_{method}_evaluation_results.jsonl"
    else:
        output_path = f"data/results/{benchmark}_{model}_{method}_evaluation_results.jsonl"
           
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Running {method} method with {model} model on {benchmark} benchmark")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Backend: {backend}")
    if method == "itergen":
        print(f"Iterations: {args.itergen}")
    elif method == "qa":
        print(f"Top-k: {args.topk}")
    
    # 根据方法调用相应的函数
    if method == "itergen":
        final_accuracy = get_itergen_test(
            input_path=input_path, 
            output_path=output_path,
            benchmark=benchmark, 
            model=model,
            backend=backend,
            iter_=args.itergen
        )
        print(f"Final accuracy: {final_accuracy:.2f}%")
    elif method == "qa":
        final_accuracy = get_qa_test(
            input_path=input_path,
            output_path=output_path,
            benchmark=benchmark,
            model=model,
            corpusfrom=corpusfrom,
            backend=backend,
            topk=args.topk
        )
        print(f"Final accuracy: {final_accuracy:.2f}%")
    elif method == "selfask":
        final_accuracy = get_selfask_test(
            input_path=input_path,
            output_path=output_path,
            benchmark=benchmark,
            model=model,
            backend=backend
        )
    elif method == "ircot":
        final_accuracy = get_ircot_test(
            input_path=input_path,
            output_path=output_path,
            benchmark=benchmark,
            model=model,
            backend=backend
        )
    elif method == "qa_ablation_rewritor":
        final_accuracy = get_qa_ablation_rewritor_test(
            input_path=input_path,
            output_path=output_path,
            benchmark=benchmark,
            model=model,
            corpusfrom=corpusfrom,
            backend=backend,
            topk=args.topk
        )
    elif method == "qa_ablation_extractor_rewritor":
        final_accuracy = get_qa_ablation_extractor_rewritor_test(
            input_path=input_path,
            output_path=output_path,
            benchmark=benchmark,
            model=model,
            corpusfrom=corpusfrom,
            backend=backend,
            topk=args.topk
        )

    else:
        print(f"Error: Method '{method}' is not implemented yet. Only 'itergen' is currently supported.")
        print("Available methods: direct, cot, selfask, naive, itergen, qa, ircot")
        print("Please implement the missing methods or use 'itergen' method.")
        sys.exit(1)
