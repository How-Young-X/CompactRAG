# python src/run.py --benchmark musique --model llama8b --method qa_ablation_rewritor  --corpusfrom test --topk 5
# python src/run.py --benchmark musique --model llama8b --method qa_ablation_extractor_rewritor  --corpusfrom test --topk 5

# python src/run.py --benchmark 2wiki --model llama8b --method qa_ablation_rewritor  --corpusfrom test --topk 5
# python src/run.py --benchmark 2wiki --model llama8b --method qa_ablation_extractor_rewritor  --corpusfrom test --topk 5

# python src/run.py --benchmark hotpotqa --model llama8b --method qa_ablation_rewritor  --corpusfrom test --topk 5
python src/run.py --benchmark hotpotqa --model llama8b --method qa_ablation_extractor_rewritor  --corpusfrom test --topk 5