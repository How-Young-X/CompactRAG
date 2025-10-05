#!/usr/bin/env python3
"""
演示失败样本保存功能
"""
import json
import os

def create_demo_failed_samples():
    """
    创建一些演示用的失败样本
    """
    print("🎭 创建演示失败样本...")
    
    # 创建输出目录
    output_dir = "/root/autodl-tmp/ReadingCorpus/data/QA"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建失败样本文件
    failed_file = os.path.join(output_dir, "demo_qa_failed.jsonl")
    
    # 创建不同类型的失败样本
    failed_samples = [
        {
            "index": 1,
            "title": "Empty Passage Sample",
            "passage": "",
            "reason": "no_valid_qa_generated",
            "qa_list": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "validation_info": {
                "invalid_questions": [],
                "missing_phrases": [],
                "required_phrases": []
            }
        },
        {
            "index": 2,
            "title": "API Error Sample",
            "passage": "This is a sample passage that caused an API error during processing.",
            "reason": "exception: Connection timeout to vLLM server",
            "qa_list": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        },
        {
            "index": 3,
            "title": "Validation Issues Sample",
            "passage": "Alexander Fleming discovered penicillin in 1928 in London.",
            "reason": "no_valid_qa_generated",
            "qa_list": [
                {"question": "What did this person discover?", "answer": "penicillin"},
                {"question": "When did this happen?", "answer": "1928"}
            ],
            "input_tokens": 150,
            "output_tokens": 50,
            "total_tokens": 200,
            "validation_info": {
                "invalid_questions": [
                    {"question": "What did this person discover?", "answer": "penicillin"}
                ],
                "missing_phrases": ["Alexander Fleming", "London"],
                "required_phrases": ["Alexander Fleming", "penicillin", "1928", "London"]
            }
        }
    ]
    
    # 保存失败样本
    with open(failed_file, 'w', encoding='utf-8') as f:
        for sample in failed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 已创建 {len(failed_samples)} 个演示失败样本")
    print(f"📁 保存位置: {failed_file}")
    
    # 创建错误日志文件
    error_log_file = os.path.join(output_dir, "demo_qa_errors.log")
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write("Error processing item 2 (title: API Error Sample): Connection timeout to vLLM server\n")
        f.write("Error processing item 5 (title: Another Sample): JSON parsing error\n")
    
    print(f"📝 已创建错误日志文件: {error_log_file}")
    
    return failed_file, error_log_file

def analyze_failed_samples(failed_file):
    """
    分析失败样本
    """
    print(f"\n🔍 分析失败样本: {failed_file}")
    
    if not os.path.exists(failed_file):
        print("❌ 失败样本文件不存在")
        return
    
    with open(failed_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"📊 总共 {len(samples)} 个失败样本")
    
    # 按失败原因分类
    reasons = {}
    for sample in samples:
        reason = sample.get('reason', 'unknown')
        if reason not in reasons:
            reasons[reason] = 0
        reasons[reason] += 1
    
    print(f"\n📋 失败原因统计:")
    for reason, count in reasons.items():
        print(f"   {reason}: {count} 个")
    
    # 显示验证问题详情
    validation_issues = 0
    for sample in samples:
        if 'validation_info' in sample:
            validation_info = sample['validation_info']
            if validation_info.get('invalid_questions') or validation_info.get('missing_phrases'):
                validation_issues += 1
    
    print(f"\n⚠️  有验证问题的样本: {validation_issues} 个")

if __name__ == "__main__":
    # 创建演示样本
    failed_file, error_log_file = create_demo_failed_samples()
    
    # 分析失败样本
    analyze_failed_samples(failed_file)
    
    print(f"\n🎉 演示完成！")
    print(f"现在可以运行 test_failed_samples.py 来查看失败样本统计")


