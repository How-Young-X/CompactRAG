#!/usr/bin/env python3
"""
测试修复后的答案提取逻辑
"""

import re

def test_answer_extraction_fix():
    """测试修复后的答案提取逻辑"""
    
    # 测试用例 - 基于实际的结果文件
    test_cases = [
        {
            "question": "Which country the director of film Lilli'S Marriage is from?",
            "response": """Here are the answers to the questions:

1. Q: Which film came out first, Blind Shaft or The Mask Of Fu Manchu? 
A: The Mask Of Fu Manchu.

2. Q: When did John V, Prince Of Anhalt-Zerbst's father die? 
A: 12 June 1516.

3. Q: Which film has the director who was born later, El Extrano Viaje or Love In Pawn? 
A: El Extrano Viaje.

4. Q: Which film has the director who was born later, Lilli's Marriage or The Wedding of Lilli Marlene? 
A: There is no information about the director of Lilli's Marriage, but Jaap Speyer, the director of Lilli's Marriage, was born in 1891, and the director of The Wedding of Lilli Marlene is Arthur Crabtree, who was born in 1903, so Arthur Crabtree was born later.

5. Q: Which film has the director who was born later, The Police Serve the Citizens? or My Blue Heaven? 
A: There is no information about the director of The Police Serve the Citizens, but Charles Saunders, the director of My Blue Heaven, was born in 1904, and Fernando Fernan Gomez, the director of The Police Serve the Citizens, was born in 1921, so Fernando Fernan Gomez was born later.

6. Q: Which film has the director who was born later, The Dance of Death or My Blue Heaven? 
A: There is no information about the director of The Dance of Death, but Michael Verhoeven, the director of The Dance of Death, was born in 1941, and Ronald Beer, the director of My Blue Heaven, was born in 1938, so Ronald Beer was born later.

7. Q: Which country the director of film Lilli'S Marriage is from? 
A: The Netherlands.""",
            "expected": "The Netherlands"
        },
        {
            "question": "What is the place of birth of the director of film I Love, You Love?",
            "response": """Here are the answers to the questions:

1. Q: Which film came out first, Blind Shaft or The Mask Of Fu Manchu? 
A: The Mask Of Fu Manchu.

2. Q: When did John V, Prince Of Anhalt-Zerbst's father die? 
A: 12 June 1516.

3. Q: Which film has the director who was born later, El Extrano Viaje or Love In Pawn? 
A: El Extrano Viaje.

4. Q: Where was the director of film "I Love, You Love" born? 
A: Bratislava.""",
            "expected": "Bratislava"
        },
        {
            "question": "Where was the place of death of the director of film Gold And The Woman?",
            "response": "The Mask Of Fu Manchu.",
            "expected": "The Mask Of Fu Manchu"
        }
    ]
    
    print("Testing fixed answer extraction logic...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        response = test_case["response"]
        expected = test_case["expected"]
        
        final_answer = None
        
        # 使用修复后的逻辑
        if response:
            # 首先尝试找到当前问题的答案
            question_lower = question.lower()
            response_lines = response.split('\n')
            
            # 寻找最后一个包含当前问题关键词的答案
            last_matching_answer = None
            for i, line in enumerate(response_lines):
                line_lower = line.lower()
                if 'A:' in line:
                    # 检查这一行或前一行是否包含当前问题的关键词
                    # 提取问题中的关键词（长度大于3的词）
                    question_words = [word.strip() for word in question_lower.split() if len(word) > 3]
                    
                    current_line_has_question = any(word in line_lower for word in question_words)
                    prev_line_has_question = False
                    if i > 0:
                        prev_line_lower = response_lines[i-1].lower()
                        prev_line_has_question = any(word in prev_line_lower for word in question_words)
                    
                    if current_line_has_question or prev_line_has_question:
                        # 找到包含当前问题的行，提取答案
                        answer_part = line.split('A:')[-1].strip()
                        if answer_part:
                            last_matching_answer = answer_part.rstrip('.')
            
            if last_matching_answer:
                final_answer = last_matching_answer
            
            # 如果还没找到，尝试多种答案格式
            if not final_answer:
                answer_patterns = [
                    r"answer is:? (.+?)(?:\.|$)",
                    r"answer: (.+?)(?:\.|$)",
                    r"the answer is:? (.+?)(?:\.|$)",
                    r"so the answer is:? (.+?)(?:\.|$)",
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        final_answer = match.group(1).strip()
                        if final_answer.endswith('.'):
                            final_answer = final_answer[:-1]
                        break
                
                # 如果还是没有找到，且响应很短，直接使用响应内容
                if not final_answer and len(response.strip()) < 100:
                    final_answer = response.strip().rstrip('.')
        
        print(f"Test {i}:")
        print(f"  Question: {question}")
        print(f"  Expected: {expected}")
        print(f"  Extracted: {final_answer}")
        print(f"  Result: {'✓ PASS' if final_answer == expected else '✗ FAIL'}")
        print()

if __name__ == "__main__":
    test_answer_extraction_fix()
