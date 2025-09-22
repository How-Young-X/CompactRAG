def is_answer_correct(pred, gold):
    
    if not pred or not gold:
        return False
    
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    
    # if pred_norm == gold_norm:
    #     return True
    if pred_norm == gold_norm:
        return True
    # if gold_norm in pred_norm:
    #     return True
    return False