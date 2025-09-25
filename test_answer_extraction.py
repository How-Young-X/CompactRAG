#!/usr/bin/env python3
"""
测试IRCoT答案提取逻辑
"""

import re

def test_answer_extraction():
    """测试答案提取逻辑"""
    
    # 测试用例
    test_cases = [
        {
            "response": "The director of the film is John Smith. So the answer is: John Smith.",
            "expected": "John Smith"
        },
        {
            "response": "Based on the information, the answer is: Netherlands.",
            "expected": "Netherlands"
        },
        {
            "response": "The answer: Romania",
            "expected": "Romania"
        },
        {
            "response": "After analyzing the data, answer is: 2001.",
            "expected": "2001"
        }
    ]
    
    # 答案提取模式
    answer_patterns = [
        r"answer is:? (.+?)(?:\.|$)",
        r"answer: (.+?)(?:\.|$)",
        r"the answer is:? (.+?)(?:\.|$)",
        r"so the answer is:? (.+?)(?:\.|$)",
    ]
    
    print("Testing answer extraction logic...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        response = test_case["response"]
        expected = test_case["expected"]
        
        final_answer = None
        
        # 尝试多种答案格式
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                if final_answer.endswith('.'):
                    final_answer = final_answer[:-1]
                break
        
        print(f"Test {i}:")
        print(f"  Response: {response}")
        print(f"  Expected: {expected}")
        print(f"  Extracted: {final_answer}")
        print(f"  Result: {'✓ PASS' if final_answer == expected else '✗ FAIL'}")
        print()

if __name__ == "__main__":
    test_answer_extraction()
