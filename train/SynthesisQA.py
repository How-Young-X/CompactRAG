from prompt.GenerateTrainData import Synthesis_Answer_Extract
from utils.Qwen import reason
import jsonlines
from utils.json_parser import extract_json_from_llm_response
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

model = "qwen3-next-80b-a3b-instruct"
data_source = ["musique","hotpotqa", "2wiki"]

# 单条数据的处理逻辑
def process_line(line_idx, line):
    try:
        title = line.get("title", "")
        content = line.get("passage", "")
        passage = title + "\n" + content

        # prompt
        idx_ = random.randint(1, 5)
        input_ = Synthesis_Answer_Extract.format(idx = idx_,passage=passage)
        # print(input_)
        # 调用模型
        try:
            llm_response = reason(model=model, prompt_=input_, temperature=0.6)
        except Exception as e_model:
            print(f"[Line {line_idx}] Model call failed: {e_model}")
            llm_response = None

        # 解析 JSON
        result = None
        if llm_response:
            try:
                result = extract_json_from_llm_response(llm_response)
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
    output_file = f"train/data/qa_synthesis.jsonl"

    with jsonlines.open(input_file, "r") as reader, \
         jsonlines.open(output_file, "a") as writer:

        lines = list(reader)

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=8) as executor:  # 你可以调大或调小 max_workers
            futures = {executor.submit(process_line, idx + 1, line): idx for idx, line in enumerate(lines)}

            for future in as_completed(futures):
                result = future.result()
                if result and result.get("parsed_json"):
                    writer.write(result)
                    print(f"[Line {result['line_idx']}] Processed successfully.")
                else:
                    print(f"[Line {result['line_idx']}] Processed failed.")
