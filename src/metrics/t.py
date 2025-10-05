import matplotlib.pyplot as plt
import numpy as np
import jsonlines
from argparse import ArgumentParser
import os


def plot_token_consumption(method_token_consumes, benchmark, model, save_path="figures"):
    """
    绘制论文风格的 token 消耗图（单次 + 累积）
    """
    os.makedirs(save_path, exist_ok=True)

    num_calls = len(list(method_token_consumes.values())[0])
    time_points = np.arange(1, num_calls + 1)

    cumulative_data = {m: np.cumsum(v) for m, v in method_token_consumes.items()}

    # 论文风格参数
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    linestyles = ['-', '--', '-.', ':']

    # ---- 子图 1：单次消耗 ----
    plt.figure(figsize=(6, 4.5))
    for i, (method, consumes) in enumerate(method_token_consumes.items()):
        plt.plot(
            time_points, consumes,
            label=method.replace("_", " ").title(),
            linewidth=1.8,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)]
        )
    plt.title(f"{benchmark} ({model}) - Per Call", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Calls")
    plt.ylabel("Token Consumption (per call)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    file_png = os.path.join(save_path, f"{benchmark}_{model}_percall.png")
    file_pdf = os.path.join(save_path, f"{benchmark}_{model}_percall.pdf")
    plt.savefig(file_png, dpi=600, bbox_inches="tight")
    plt.savefig(file_pdf, bbox_inches="tight")
    print(f"单次消耗图已保存: {file_png}, {file_pdf}")
    plt.close()

    # ---- 子图 2：累积消耗 ----
    plt.figure(figsize=(6, 4.5))
    for i, (method, cumulative_consumes) in enumerate(cumulative_data.items()):
        plt.plot(
            time_points, cumulative_consumes,
            label=method.replace("_", " ").title(),
            linewidth=1.8,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)]
        )
    plt.title(f"{benchmark} ({model}) - Cumulative", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Calls")
    plt.ylabel("Cumulative Token Consumption")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    file_png = os.path.join(save_path, f"{benchmark}_{model}_cumulative.png")
    file_pdf = os.path.join(save_path, f"{benchmark}_{model}_cumulative.pdf")
    plt.savefig(file_png, dpi=600, bbox_inches="tight")
    plt.savefig(file_pdf, bbox_inches="tight")
    print(f"累积消耗图已保存: {file_png}, {file_pdf}")
    plt.close()


def expand_or_trim(records, num_calls):
    """
    保持顺序扩展或裁剪 records，避免随机采样
    """
    if not records:
        return [0] * num_calls
    if len(records) >= num_calls:
        return records[:num_calls]
    # 循环扩展
    repeats = (num_calls // len(records)) + 1
    expanded = (records * repeats)[:num_calls]
    return expanded


def load_token_consumption_data(benchmark, model, num_calls=500):
    """
    加载各种方法的token消耗数据
    """
    qa_corpus_path = f"data/QA/{benchmark}_qa.jsonl"
    try:
        with jsonlines.open(qa_corpus_path, "r") as reader:
            records = [record for record in reader]
        qa_corpus_token_consumes = [record["total_tokens"] for record in records]
        qa_corpus_initial_tokens = sum(qa_corpus_token_consumes)
    except FileNotFoundError:
        print(f"警告: 找不到文件 {qa_corpus_path}")
        qa_corpus_initial_tokens = 0

    methods = ["selfask", "ircot", "itergen_iter4", "qa_corpus_llama8b"]
    method_token_consumes = {method: [] for method in methods}

    for method in methods:
        if method == "qa_corpus_llama8b":
            path = f"data/results/{benchmark}_{model}_qa_corpus_llama8b_evaluation_results.jsonl"
            try:
                with jsonlines.open(path, "r") as reader:
                    records = [record for record in reader]
                tokens = [r["total_tokens"] for r in records]
                tokens = expand_or_trim(tokens, num_calls)
                method_token_consumes[method] = [tokens[0] + qa_corpus_initial_tokens] + tokens[1:]
                method_token_consumes[method] = [tokens[0]] + tokens[1:]
            
            except FileNotFoundError:
                print(f"警告: 找不到文件 {path}")
                method_token_consumes[method] = [qa_corpus_initial_tokens] + [0] * (num_calls - 1)
        else:
            path = f"data/results/{benchmark}_{model}_{method}_evaluation_results.jsonl"
            try:
                with jsonlines.open(path, "r") as reader:
                    records = [record for record in reader]
                tokens = [r["total_tokens"] for r in records]
                method_token_consumes[method] = expand_or_trim(tokens, num_calls)
            except FileNotFoundError:
                print(f"警告: 找不到文件 {path}")
                method_token_consumes[method] = [0] * num_calls

    return method_token_consumes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="benchmark name")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--num_calls", type=int, default=500, help="number of calls to simulate")
    parser.add_argument("--save_path", type=str, default="figures", help="path to save the figure")
    args = parser.parse_args()

    print(f"正在加载 {args.benchmark} 基准测试下 {args.model} 模型的token消耗数据...")

    method_token_consumes = load_token_consumption_data(args.benchmark, args.model, args.num_calls)

    print("\n数据摘要:")
    for method, consumes in method_token_consumes.items():
        total_consumes = sum(consumes)
        avg_consumes = total_consumes / len(consumes) if consumes else 0
        print(f"{method}: 总计 {total_consumes} tokens, 平均 {avg_consumes:.1f} tokens/调用, {len(consumes)} 条记录")

    print(f"\n正在生成论文风格的token消耗图...")
    plot_token_consumption(method_token_consumes, args.benchmark, args.model, args.save_path)
    print("完成!")
