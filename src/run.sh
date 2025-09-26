#!/bin/bash

MODEL="gemma4b"

BENCHMARKS=("2wiki" "hotpotqa" "musique")
METHODS=("qa" "ircot" "itergen" "selfask")
ITERGENS=(2 4) 
CORPUSFROM=("qwen3-32b" "llama8b")

for BENCH in "${BENCHMARKS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    
    if [ "$METHOD" == "itergen" ]; then
      # itergen 特殊处理
      for ITERS in "${ITERGENS[@]}"; do
        echo "Running: benchmark=$BENCH, model=$MODEL, method=$METHOD, itergen=$ITERS"
        python src/run.py \
          --benchmark "$BENCH" \
          --model "$MODEL" \
          --method "$METHOD" \
          --itergen "$ITERS"
      done
    
    elif [ "$METHOD" == "qa" ]; then
      # qa 特殊处理
      for CORPUS in "${CORPUSFROM[@]}"; do
        echo "Running: benchmark=$BENCH, model=$MODEL, method=$METHOD, corpusfrom=$CORPUS"
        python src/run.py \
          --benchmark "$BENCH" \
          --model "$MODEL" \
          --method "$METHOD" \
          --corpusfrom "$CORPUS"
      done
    
    else
      echo "Running: benchmark=$BENCH, model=$MODEL, method=$METHOD"
      python src/run.py \
        --benchmark "$BENCH" \
        --model "$MODEL" \
        --method "$METHOD"
    fi
  done
done
