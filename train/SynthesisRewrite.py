from prompt.GenerateTrainData import Synthesis_question_rewrite
from utils.Qwen import reason
import jsonlines
from utils.json_parser import extract_json_from_llm_response
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

model = "qwen3-32b"
data_source = ["2wiki", "hotpotqa","musique"]

# 单条数据的处理逻辑
def process_line(line_idx, line):
    try:
        title = line.get("title", "")
        content = line.get("passage", "")
        passage = title + "\n" + content

        # 生成 prompt
        input_ = Synthesis_question_rewrite.format(passage=passage)
        
        # 调用模型
        try:
            # print("llm start")
            llm_response = reason(model=model, prompt_=input_, temperature=0.6)
            # print(llm_response)
            # print("llm end")
        except Exception as e_model:
            print(f"[Line {line_idx}] Model call failed: {e_model}")
            llm_response = None

        # 解析 JSON
        result = None
        if llm_response:
            try:
                result = extract_json_from_llm_response(llm_response)
                # print("*"*20)
                # print(result)
                # print("*"*20)

            except Exception as e_parse:
                print(f"[Line {line_idx}] JSON parsing failed: {e_parse}")
                traceback.print_exc()

        # 返回结果
        return {
            "title": title,
            "passage": content,
            "llm_response": llm_response,
            "parsed_json": result,
            "line_idx": line_idx
        }

    except Exception as e_outer:
        print(f"[Line {line_idx}] Unexpected error: {e_outer}")
        traceback.print_exc()
        return None


for source in data_source:
    input_file = f"data/train/corpus/{source}_sample_corpus.jsonl"
    output_file = f"train/data/rewrite_synthesis.jsonl"

    with jsonlines.open(input_file, "r") as reader, \
         jsonlines.open(output_file, "a") as writer:

        lines = list(reader)

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            futures = {executor.submit(process_line, idx + 1, line): idx for idx, line in enumerate(lines)}
            
            # 处理完成的任务
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    # 原始逻辑：只有解析失败（result为None）时才写入
                    if result.get("parsed_json"):
                        writer.write(result)
                        print(f"[Line {result['line_idx']}] Processed successfully.")
                    else:
                        print(f"[Line {result['line_idx']}] Processed failed.")