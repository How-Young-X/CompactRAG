# ReadingCorpus 运行脚本说明

本项目提供了两个运行脚本来执行 RAG 评估实验：

## 1. run_simple.sh - 简单运行脚本

用于快速运行单个实验。

### 使用方法：
```bash
# 使用默认参数 (musique + itergen + 2 iterations)
./run_simple.sh

# 指定 benchmark
./run_simple.sh 2wiki

# 指定 benchmark 和 method
./run_simple.sh hotpotqa itergen

# 指定所有参数
./run_simple.sh musique itergen 3
```

### 参数说明：
- 第1个参数：benchmark (musique, 2wiki, hotpotqa)
- 第2个参数：method (目前只支持 itergen)
- 第3个参数：iterations (默认 2)

## 2. run.sh - 完整运行脚本

支持多个 benchmark 和 method 的批量运行，包括并行执行。

### 基本用法：
```bash
# 运行所有 benchmark 和 method
./run.sh

# 只运行特定 benchmark
./run.sh -b musique

# 运行多个 benchmark
./run.sh -b musique,2wiki

# 指定 iterations
./run.sh --iterations 3

# 并行运行 (2个任务同时执行)
./run.sh -j 2
```

### 完整参数说明：
```bash
./run.sh [OPTIONS]

Options:
  -b, --benchmarks BENCHMARKS    逗号分隔的 benchmark 列表 (默认: musique,2wiki,hotpotqa)
  -m, --methods METHODS          逗号分隔的 method 列表 (默认: itergen)
  --model MODEL                  模型名称 (默认: llama8b)
  --backend BACKEND              后端 (默认: vllm)
  --iterations ITERATIONS        迭代次数 (默认: 2)
  -j, --parallel-jobs JOBS       并行任务数 (默认: 1)
  -o, --output-dir DIR           输出目录 (默认: data/results)
  --log-dir DIR                  日志目录 (默认: logs)
  -h, --help                     显示帮助信息
```

### 使用示例：

```bash
# 运行所有 benchmark，使用 itergen 方法
./run.sh

# 只运行 musique benchmark
./run.sh -b musique

# 运行 musique 和 2wiki，使用 3 次迭代
./run.sh -b musique,2wiki --iterations 3

# 并行运行 2 个任务
./run.sh -j 2

# 自定义输出和日志目录
./run.sh -o my_results --log-dir my_logs
```

## 运行前准备

1. **启动 vLLM 服务**：
   ```bash
   bash start_vllm_server.sh
   ```

2. **确保数据文件存在**：
   - `data/sampled/musique_test.jsonl`
   - `data/sampled/2wiki_test.jsonl`
   - `data/sampled/hotpotqa_test.jsonl`

3. **确保索引文件存在**：
   - `data/index/passage/musique/corpus.index`
   - `data/index/passage/2wiki/corpus.index`
   - `data/index/passage/hotpotqa/corpus.index`

## 输出文件

- **结果文件**：`data/results/{benchmark}_{model}_{method}_iter{iterations}_evaluation_results.jsonl`
- **日志文件**：`logs/{benchmark}_{method}_{model}_iter{iterations}_{timestamp}.log`

## 注意事项

1. 确保 vLLM 服务在 `http://localhost:8000` 上运行
2. 确保有足够的磁盘空间存储结果和日志
3. 并行运行时注意内存使用情况
4. 如果遇到错误，检查日志文件获取详细信息

## 故障排除

1. **vLLM 服务未运行**：
   ```bash
   bash start_vllm_server.sh
   ```

2. **缺少依赖包**：
   ```bash
   pip install -r requirements.txt
   ```

3. **数据文件不存在**：
   确保测试数据文件在 `data/sampled/` 目录下

4. **索引文件不存在**：
   需要先构建索引文件，参考 `PassageIndexSearch.py` 的使用说明
