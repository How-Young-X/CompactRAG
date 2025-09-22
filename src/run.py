import argparse
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.method.IterGen import get_itergen_test

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="musique",choices=["musique","2wiki","hotpotqa"])
    parser.add_argument("--model", type=str, default="llama8b", help="Model name for vLLM")
    parser.add_argument("--method",type=str, default="itergen",choices=["direct","cot","selfask","naive","itergen","qa"], help="RAG method to use")
    parser.add_argument("--itergen",type=int, default=2, help="Number of iterations for itergen method")
    parser.add_argument("--topk",type=int, default=5, help="Number of top-k passages for retrieval")
    parser.add_argument("--backend",type=str, default="vllm",choices=["vllm"], help="Backend to use (only vllm supported)")

    
    args = parser.parse_args()
    
    # 使用示例
    model = args.model
    benchmark = args.benchmark
    method = args.method
    backend = args.backend

    # 设置输入和输出路径
    input_path = f"data/sampled/{benchmark}_test.jsonl"
    
    # 根据方法设置不同的输出路径
    if method == "itergen":
        output_path = f"data/results/{benchmark}_{model}_{method}_iter{args.itergen}_evaluation_results.jsonl"
    elif method == "qa":
        output_path = f"data/results/{benchmark}_{model}_{method}_top{args.topk}_evaluation_results.jsonl"
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
    else:
        print(f"Error: Method '{method}' is not implemented yet. Only 'itergen' is currently supported.")
        print("Available methods: direct, cot, selfask, naive, itergen, qa")
        print("Please implement the missing methods or use 'itergen' method.")
        sys.exit(1)
