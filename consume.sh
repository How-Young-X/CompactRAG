#!/bin/bash

# 设置错误处理
set -e

# 定义参数数组
benchmarks=("2wiki" "hotpotqa" "musique")
models=("llama8b")
methods=("selfask" "ircot" "itergen_iter4" "qa_corpus_llama8b")

# 计数器
total_runs=0
successful_runs=0
failed_runs=0

echo "开始执行 TokenConsume 测试..."
echo "=================================="

# 循环执行不同的参数组合
for benchmark in "${benchmarks[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            total_runs=$((total_runs + 1))
            
            echo "执行第 $total_runs 次测试:"
            echo "  Benchmark: $benchmark"
            echo "  Model: $model"
            echo "  Method: $method"
            echo "--------------------------------"
            
            # 执行 Python 脚本
            if python src/metrics/TokenConsume.py --benchmark "$benchmark" --model "$model" --method "$method"; then
                echo "✅ 测试成功完成"
                successful_runs=$((successful_runs + 1))
            else
                echo "❌ 测试失败"
                failed_runs=$((failed_runs + 1))
            fi
            
            echo ""
            
            # 可选：添加延迟避免系统过载
            sleep 2
        done
    done
done

# 输出总结
echo "=================================="
echo "测试完成！"
echo "总测试次数: $total_runs"
echo "成功次数: $successful_runs"
echo "失败次数: $failed_runs"
echo "成功率: $(( successful_runs * 100 / total_runs ))%"