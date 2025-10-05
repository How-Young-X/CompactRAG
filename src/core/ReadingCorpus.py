import json
import re
import os
from openai import OpenAI
from tqdm import tqdm

# --------- vLLM OpenAI API 客户端配置 ---------
client = OpenAI(
    api_key="EMPTY",  # vLLM不需要真实的API key
    base_url="http://localhost:8000/v1"
)
model_name = "llama8b"  # 对应start_vllm_server.sh中设置的served-model-name

# --------- Prompt 模板 ---------
STAGE1_PROMPT = """
You are an extractor. Input: the chunk delimited by triple backticks.
Task: Return a JSON array (only the array, nothing else) of exact strings that must later appear verbatim in at least one question.
Include:
  - every named entity (person, organization, location, title, product, book, etc.) exactly as written;
  - every descriptive phrase that identifies an object (e.g., "a wide river", "an ancient temple") exactly as written;
  - any short atomic factual phrases (phrases that express a single fact) that the next stage must cover.
Do NOT add, paraphrase, split, or invent. Preserve case and punctuation exactly. If none, return [].
Example input:
```Alexander Fleming discovered penicillin in 1928 in London.```
Output:
["Alexander Fleming","penicillin","1928","London"]

Now extract REQUIRED_PHRASES for:
```{chunk}```
"""

STAGE2_PROMPT = """
You are a question generation assistant. Generate questions and answers based on the given text.

Task: Create a JSON array where each element is an object with "question" and "answer" fields.

Rules:
1. Each question must include exactly ONE phrase from REQUIRED_PHRASES
2. Avoid pronouns like "this", "that", "it", "they", "he", "she", etc.
3. Use specific names and phrases from REQUIRED_PHRASES
4. Each answer must be copied exactly from the text
5. Cover all REQUIRED_PHRASES in your questions
6. Output only the JSON array in a code block

Example:
Text: "Alexander Fleming discovered penicillin in 1928 in London."
REQUIRED_PHRASES: ["Alexander Fleming", "penicillin", "1928", "London"]

Output:
```json
[
  {{"question": "Who discovered penicillin?", "answer": "Alexander Fleming discovered penicillin"}},
  {{"question": "When was penicillin discovered?", "answer": "Penicillin discovered in1928"}},
  {{"question": "Where was penicillin discovered?", "answer": "Penicillin was discovered in London"}}
]
```

Now generate questions for:
Text: ```{chunk}```
REQUIRED_PHRASES: {required_phrases}
"""

# --------- 辅助函数 ---------
BAD_WORDS = [
"this","that","it","they","those","these",
 "he","she","his","her","him","their","them","we","us","I",
 "the film","this film","the director","the book","the article","the passage","the story","the movie",
 "the person","the place","the country","the year",
 "the director","this director","the book","this book","the article","this article","the passage","this passage","the story","this story","the movie","this movie",
 "the person","this person","the place","this place","the country","this country","the year","this year"
]

def has_bad_word(s):
    s_low = s.lower()
    return any(w in s_low for w in BAD_WORDS)

def coverage_check(required_phrases, qa_list):
    questions = [q["question"] for q in qa_list]
    missing = [p for p in required_phrases if not any(p in q for q in questions)]
    return missing

def call_llm(prompt, max_new_tokens=512, max_retries=3):
    """
    调用vLLM API生成文本，返回生成结果和token统计信息
    包含重试机制确保稳定性
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            return {
                "text": generated_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        except Exception as e:
            print(f"⚠️  API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"🔄 等待2秒后重试...")
                import time
                time.sleep(2)
            else:
                print(f"❌ 达到最大重试次数，返回空结果")
                return {
                    "text": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }

def extract_json_block(text):
    match = re.search(r"```json(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

# --------- 主流程 ---------
def generate_qa(chunk):
    """
    生成QA对，返回QA列表和token统计信息
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    # Stage 1
    stage1_prompt = STAGE1_PROMPT.format(chunk=chunk)
    stage1_result = call_llm(stage1_prompt, max_new_tokens=256)
    total_input_tokens += stage1_result["input_tokens"]
    total_output_tokens += stage1_result["output_tokens"]
    total_tokens += stage1_result["total_tokens"]
    
    try:
        required_phrases = json.loads(extract_json_block(stage1_result["text"]))
    except:
        required_phrases = []
    print("Stage1 REQUIRED_PHRASES:", required_phrases)

    # Stage 2
    stage2_prompt = STAGE2_PROMPT.format(chunk=chunk, required_phrases=json.dumps(required_phrases))
    stage2_result = call_llm(stage2_prompt, max_new_tokens=1024)
    total_input_tokens += stage2_result["input_tokens"]
    total_output_tokens += stage2_result["output_tokens"]
    total_tokens += stage2_result["total_tokens"]
    
    qa_text = extract_json_block(stage2_result["text"])
    try:
        qa_list = json.loads(qa_text)
    except:
        qa_list = []

    # 本地校验
    invalid = []
    for qa in qa_list:
        if has_bad_word(qa["question"]):
            invalid.append(qa)
    missing = coverage_check(required_phrases, qa_list)

    if invalid or missing:
        print("校验失败，重生缺失部分...")
        print("Invalid questions:", invalid)
        print("Missing phrases:", missing)

    return {
        "qa_list": qa_list,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "validation_info": {
            "invalid_questions": invalid,
            "missing_phrases": missing,
            "required_phrases": required_phrases
        }
    }

# --------- 语料处理函数 ---------
def process_corpus_file(input_file, output_file, dataset_name):
    """
    处理单个语料文件，生成QA并增量保存
    """
    print(f"开始处理 {dataset_name} 数据集...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 创建失败样本保存文件
    failed_file = output_file.replace('.jsonl', '_failed.jsonl')
    print(f"失败样本将保存到: {failed_file}")
    
    # 读取语料文件
    corpus_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus_data.append(json.loads(line.strip()))
    
    print(f"总共读取 {len(corpus_data)} 条语料")
    
    # 检查已处理的语料数量（断点续传）
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f if line.strip())
        print(f"发现已处理 {processed_count} 条语料，将从第 {processed_count + 1} 条开始")
    
    # 统计变量
    total_input_tokens = 0
    total_output_tokens = 0
    total_total_tokens = 0
    total_qa_count = 0
    failed_count = 0
    
    # 从断点开始处理
    for i in range(processed_count, len(corpus_data)):
        item = corpus_data[i]
        title = item.get("title", "")
        passage = item.get("passage", "")
        
        print(f"\n🔄 处理第 {i + 1}/{len(corpus_data)} 条语料: {title[:50]}...")
        
        try:
            # 使用passage作为chunk生成QA
            result = generate_qa(passage)
            
            # 检查生成结果是否有效
            if not result["qa_list"] or len(result["qa_list"]) == 0:
                print(f"⚠️  第 {i + 1} 条语料未生成有效QA对，保存为失败样本")
                failed_count += 1
                
                # 保存失败样本
                failed_entry = {
                    "index": i + 1,
                    "title": title,
                    "passage": passage,
                    "reason": "no_valid_qa_generated",
                    "qa_list": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "validation_info": result.get("validation_info", {})
                }
                
                with open(failed_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
                continue
            
            # 检查是否有验证问题但仍有QA对的情况
            validation_info = result.get("validation_info", {})
            has_validation_issues = (validation_info.get("invalid_questions") or 
                                   validation_info.get("missing_phrases"))
            
            if has_validation_issues:
                print(f"⚠️  第 {i + 1} 条语料有验证问题，但仍保存QA对")
                # 在成功样本中也记录验证信息
                corpus_entry = {
                    "title": title,
                    "passage": passage,
                    "qa": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "validation_warnings": validation_info
                }
            else:
                # 正常的成功样本
                corpus_entry = {
                    "title": title,
                    "passage": passage,
                    "qa": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"]
                }
            
            
            # 立即保存到文件（增量保存）
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
            
            # 累计统计
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_total_tokens += result["total_tokens"]
            total_qa_count += len(result["qa_list"])
            
            print(f"✅ 成功生成 {len(result['qa_list'])} 个QA对，已保存")
            print(f"📊 当前统计: 总QA对={total_qa_count}, 总token={total_total_tokens}")
            
        except Exception as e:
            print(f"❌ 处理第 {i + 1} 条语料时出错: {e}")
            print(f"保存为失败样本，继续处理下一条...")
            failed_count += 1
            
            # 保存失败样本
            failed_entry = {
                "index": i + 1,
                "title": title,
                "passage": passage,
                "reason": f"exception: {str(e)}",
                "qa_list": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            
            with open(failed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
            
            # 记录错误到日志文件
            error_log_file = output_file.replace('.jsonl', '_errors.log')
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error processing item {i+1} (title: {title[:50]}): {e}\n")
            continue
        
        # 每处理10条语料打印一次详细进度
        if (i + 1) % 10 == 0:
            print(f"\n📈 进度报告:")
            print(f"   已处理: {i + 1}/{len(corpus_data)} 条语料")
            print(f"   成功: {i + 1 - processed_count - failed_count} 条")
            print(f"   失败: {failed_count} 条")
            print(f"   生成QA对: {total_qa_count} 个")
            print(f"   Token使用: 输入={total_input_tokens}, 输出={total_output_tokens}, 总计={total_total_tokens}")
            if i + 1 - processed_count - failed_count > 0:
                print(f"   平均每条成功语料: {total_qa_count/(i+1-processed_count-failed_count):.1f} 个QA对")
    
    print(f"\n🎉 {dataset_name} 数据集处理完成!")
    print(f"📊 最终统计:")
    print(f"   总语料数: {len(corpus_data)} 条")
    print(f"   成功处理: {len(corpus_data) - processed_count - failed_count} 条")
    print(f"   失败样本: {failed_count} 条")
    print(f"   生成QA对: {total_qa_count} 个")
    print(f"   Token使用: 输入={total_input_tokens}, 输出={total_output_tokens}, 总计={total_total_tokens}")
    if len(corpus_data) - processed_count - failed_count > 0:
        print(f"   平均每条成功语料: {total_qa_count/(len(corpus_data)-processed_count-failed_count):.1f} 个QA对")
    print(f"   成功结果已保存到: {output_file}")
    print(f"   失败样本已保存到: {failed_file}")

def process_all_corpora():
    """
    处理所有三个语料文件
    """
    base_dir = "/root/autodl-tmp/ReadingCorpus/data/sampled"
    output_dir = "/root/autodl-tmp/ReadingCorpus/data/QA"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义数据集配置
    datasets = [
        {
            "name": "2wiki",
            "input_file": os.path.join(base_dir, "2wiki_sample_corpus.jsonl"),
            "output_file": os.path.join(output_dir, "2wiki_qa.jsonl")
        },
        # {
        #     "name": "hotpotqa", 
        #     "input_file": os.path.join(base_dir, "hotpotqa_sample_corpus.jsonl"),
        #     "output_file": os.path.join(output_dir, "hotpotqa_qa.jsonl")
        # },
        # {
        #     "name": "musique",
        #     "input_file": os.path.join(base_dir, "musique_sample_corpus.jsonl"),
        #     "output_file": os.path.join(output_dir, "musique_qa.jsonl")
        # },
        # {
        #     "name": "hotpotqa", 
        #     "input_file": os.path.join(base_dir, "hotpotqa_sample_corpus.jsonl"),
        #     "output_file": os.path.join(output_dir, "hotpotqa_qa.jsonl")
        # }
    ]
    
    # 处理每个数据集
    for dataset in datasets:
        if os.path.exists(dataset["input_file"]):
            process_corpus_file(
                dataset["input_file"], 
                dataset["output_file"], 
                dataset["name"]
            )
        else:
            print(f"❌ 输入文件不存在: {dataset['input_file']}")

# --------- 测试和主程序 ---------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        chunk = "Alexander Fleming discovered penicillin in 1928 in London when he observed mold killing bacteria."
        result = generate_qa(chunk)
        print("测试结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 处理所有语料
        process_all_corpora()
