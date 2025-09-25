#!/usr/bin/env python3
"""
测试IRCoT方法的简单脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ircot_prompt():
    """测试IRCoT prompt导入"""
    try:
        from prompt.IRCoT import IRCOT_HOTPOTQA_PROMPT, IRCOT_MUSIQUE_PROMPT, IRCOT_2WIKI_PROMPT
        print("✓ IRCoT prompts imported successfully")
        
        # 测试prompt格式
        test_context = "Wikipedia Title: Test Title\nTest content here."
        test_question = "What is the test question?"
        test_generation = "Let me think step by step."
        
        formatted_prompt = IRCOT_HOTPOTQA_PROMPT.format(
            context=test_context,
            question=test_question,
            generation_so_far=test_generation
        )
        
        print("✓ Prompt formatting works correctly")
        print(f"Formatted prompt length: {len(formatted_prompt)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing IRCoT prompts: {str(e)}")
        return False

def test_ircot_method():
    """测试IRCoT方法导入"""
    try:
        from core.method.IRCoT import get_ircot_test
        print("✓ IRCoT method imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error importing IRCoT method: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing IRCoT implementation...")
    print("=" * 50)
    
    # 测试prompt导入
    prompt_ok = test_ircot_prompt()
    print()
    
    # 测试方法导入
    method_ok = test_ircot_method()
    print()
    
    if prompt_ok and method_ok:
        print("✓ All tests passed! IRCoT implementation is ready.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
