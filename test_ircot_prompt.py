#!/usr/bin/env python3
"""
测试IRCoT prompt格式的详细脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ircot_prompt_format():
    """测试IRCoT prompt格式"""
    try:
        from prompt.IRCoT import IRCOT_HOTPOTQA_PROMPT, IRCOT_MUSIQUE_PROMPT, IRCOT_2WIKI_PROMPT
        print("✓ IRCoT prompts imported successfully")
        
        # 测试prompt格式
        test_context = """Wikipedia Title: Test Title
Test content here with some information about the topic.

Wikipedia Title: Another Title
Another paragraph with more information."""
        
        test_question = "What is the test question?"
        test_generation = "Let me think step by step."
        
        # 测试HotpotQA prompt
        formatted_prompt = IRCOT_HOTPOTQA_PROMPT.format(
            context=test_context,
            question=test_question,
            generation_so_far=test_generation
        )
        
        print("✓ HotpotQA prompt formatting works correctly")
        print(f"Formatted prompt length: {len(formatted_prompt)} characters")
        
        # 检查prompt是否包含示例
        if "Nobody Loves You" in formatted_prompt:
            print("✓ HotpotQA prompt contains correct examples")
        else:
            print("✗ HotpotQA prompt missing examples")
        
        # 检查prompt是否包含当前问题
        if test_question in formatted_prompt:
            print("✓ Current question is included in prompt")
        else:
            print("✗ Current question not found in prompt")
        
        # 检查prompt是否包含当前推理
        if test_generation in formatted_prompt:
            print("✓ Current generation is included in prompt")
        else:
            print("✗ Current generation not found in prompt")
        
        # 测试其他数据集的prompt
        formatted_prompt_2wiki = IRCOT_2WIKI_PROMPT.format(
            context=test_context,
            question=test_question,
            generation_so_far=test_generation
        )
        
        if "Blind Shaft" in formatted_prompt_2wiki:
            print("✓ 2Wiki prompt contains correct examples")
        else:
            print("✗ 2Wiki prompt missing examples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing IRCoT prompt format: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ircot_prompt_structure():
    """测试prompt结构"""
    try:
        from prompt.IRCoT import IRCOT_HOTPOTQA_PROMPT
        
        # 检查prompt是否包含正确的结构
        prompt_lines = IRCOT_HOTPOTQA_PROMPT.split('\n')
        
        # 检查是否包含Wikipedia Title
        has_wikipedia_titles = any("Wikipedia Title:" in line for line in prompt_lines)
        if has_wikipedia_titles:
            print("✓ Prompt contains Wikipedia titles")
        else:
            print("✗ Prompt missing Wikipedia titles")
        
        # 检查是否包含Q:和A:格式
        has_qa_format = any("Q:" in line for line in prompt_lines) and any("A:" in line for line in prompt_lines)
        if has_qa_format:
            print("✓ Prompt contains Q: and A: format")
        else:
            print("✗ Prompt missing Q: and A: format")
        
        # 检查是否包含答案格式
        has_answer_format = any("So the answer is:" in line for line in prompt_lines)
        if has_answer_format:
            print("✓ Prompt contains answer format")
        else:
            print("✗ Prompt missing answer format")
        
        # 检查是否包含占位符
        has_placeholders = "{context}" in IRCOT_HOTPOTQA_PROMPT and "{question}" in IRCOT_HOTPOTQA_PROMPT and "{generation_so_far}" in IRCOT_HOTPOTQA_PROMPT
        if has_placeholders:
            print("✓ Prompt contains required placeholders")
        else:
            print("✗ Prompt missing required placeholders")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing IRCoT prompt structure: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing IRCoT prompt format and structure...")
    print("=" * 60)
    
    # 测试prompt格式
    format_ok = test_ircot_prompt_format()
    print()
    
    # 测试prompt结构
    structure_ok = test_ircot_prompt_structure()
    print()
    
    if format_ok and structure_ok:
        print("✓ All IRCoT prompt tests passed!")
    else:
        print("✗ Some IRCoT prompt tests failed.")
