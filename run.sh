#!/bin/bash

# ReadingCorpus RAG Evaluation Script
# Supports multiple benchmarks and methods

# example:
# ./run.sh -b musique,2wiki -m itergen,qa --iterations 2 --topk 5
# m represents method
# b represents benchmark
# --iterations 2 --topk 5 are the arguments for the method
# if you want to run all benchmarks and methods, you can use:
# ./run.sh
# if you want to run all benchmarks and methods with parallel jobs, you can use:
# ./run.sh -j 4
# if you want to run all benchmarks and methods with parallel jobs and output directory, you can use:
# ./run.sh -j 4 -o data/results
# if you want to run all benchmarks and methods with parallel jobs and output directory and log directory, you can use:
# ./run.sh -j 4 -o data/results -l logs
# if you want to run all benchmarks and methods with parallel jobs and output directory and log directory and help, you can use:
# ./run.sh -j 4 -o data/results -l logs -h

set -e  # Exit on any error

# Default values
BENCHMARKS=("hotpotqa" "2wiki" "musique")
METHODS=("qa")
MODEL="llama8b"
BACKEND="vllm"
ITERATIONS=2
TOPK=5
PARALLEL_JOBS=1
OUTPUT_DIR="data/results"
LOG_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --benchmarks BENCHMARKS    Comma-separated list of benchmarks (default: musique,2wiki,hotpotqa)"
    echo "  -m, --methods METHODS          Comma-separated list of methods (default: itergen)"
    echo "  --model MODEL                  Model name (default: llama8b)"
    echo "  --backend BACKEND              Backend to use (default: vllm)"
    echo "  --iterations ITERATIONS        Number of iterations for itergen (default: 2)"
    echo "  --topk TOPK                    Number of top-k passages for retrieval (default: 5)"
    echo "  -j, --parallel-jobs JOBS       Number of parallel jobs (default: 1)"
    echo "  -o, --output-dir DIR           Output directory (default: data/results)"
    echo "  --log-dir DIR                  Log directory (default: logs)"
    echo "  -h, --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all benchmarks with all methods"
    echo "  $0 -b musique -m direct              # Run only musique benchmark with direct method"
    echo "  $0 -b musique,2wiki -m cot,selfask   # Run musique and 2wiki with cot and selfask methods"
    echo "  $0 -m itergen --iterations 3         # Run itergen method with 3 iterations"
    echo "  $0 -m qa --topk 10                   # Run qa method with top-10 retrieval"
    echo "  $0 -j 2                              # Run with 2 parallel jobs"
}

# Function to check if vLLM service is running
check_vllm_service() {
    print_info "Checking vLLM service..."
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_success "vLLM service is running"
        return 0
    else
        print_error "vLLM service is not running. Please start the vLLM server first."
        print_info "You can start it with: bash start_vllm_server.sh"
        return 1
    fi
}

# Function to run a single experiment
run_experiment() {
    local benchmark=$1
    local method=$2
    local model=$3
    local backend=$4
    local iterations=$5
    local topk=$6
    local output_dir=$7
    local log_dir=$8
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${log_dir}/${benchmark}_${method}_${model}_${timestamp}.log"
    
    print_info "Starting experiment: ${benchmark} + ${method} + ${model}"
    
    # Create log directory if it doesn't exist
    mkdir -p "$log_dir"
    
    # Build command arguments
    local cmd_args="--benchmark $benchmark --method $method --model $model --backend $backend"
    
    # Add method-specific arguments
    case "$method" in
        "itergen")
            cmd_args="$cmd_args --itergen $iterations"
            ;;
        "qa")
            cmd_args="$cmd_args --topk $topk"
            ;;
    esac
    
    # Run the experiment
    if python src/run.py $cmd_args > "$log_file" 2>&1; then
        print_success "Completed: ${benchmark} + ${method} + ${model}"
        return 0
    else
        print_error "Failed: ${benchmark} + ${method} + ${model}"
        print_error "Check log file: $log_file"
        return 1
    fi
}

# Function to run experiments in parallel
run_parallel_experiments() {
    local jobs=0
    local max_jobs=$1
    local pids=()
    
    for benchmark in "${BENCHMARKS[@]}"; do
        for method in "${METHODS[@]}"; do
            # Wait for a slot if we've reached the maximum number of parallel jobs
            while [ $jobs -ge $max_jobs ]; do
                for i in "${!pids[@]}"; do
                    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                        wait "${pids[$i]}"
                        unset pids[$i]
                        ((jobs--))
                    fi
                done
                sleep 1
            done
            
            # Start a new job
            run_experiment "$benchmark" "$method" "$MODEL" "$BACKEND" "$ITERATIONS" "$TOPK" "$OUTPUT_DIR" "$LOG_DIR" &
            pids+=($!)
            ((jobs++))
            
            print_info "Started job ${jobs}/${max_jobs}: ${benchmark} + ${method}"
        done
    done
    
    # Wait for all remaining jobs to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--benchmarks)
            IFS=',' read -ra BENCHMARKS <<< "$2"
            shift 2
            ;;
        -m|--methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --topk)
            TOPK="$2"
            shift 2
            ;;
        -j|--parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "ReadingCorpus RAG Evaluation"
    print_info "=============================="
    print_info "Benchmarks: ${BENCHMARKS[*]}"
    print_info "Methods: ${METHODS[*]}"
    print_info "Model: $MODEL"
    print_info "Backend: $BACKEND"
    print_info "Iterations: $ITERATIONS"
    print_info "Top-k: $TOPK"
    print_info "Parallel jobs: $PARALLEL_JOBS"
    print_info "Output directory: $OUTPUT_DIR"
    print_info "Log directory: $LOG_DIR"
    print_info "=============================="
    
    # Check if vLLM service is running
    if ! check_vllm_service; then
        exit 1
    fi
    
    # Create output and log directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    # Calculate total number of experiments
    total_experiments=$((${#BENCHMARKS[@]} * ${#METHODS[@]}))
    print_info "Total experiments to run: $total_experiments"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run experiments
    if [ $PARALLEL_JOBS -eq 1 ]; then
        # Sequential execution
        success_count=0
        for benchmark in "${BENCHMARKS[@]}"; do
            for method in "${METHODS[@]}"; do
                if run_experiment "$benchmark" "$method" "$MODEL" "$BACKEND" "$ITERATIONS" "$TOPK" "$OUTPUT_DIR" "$LOG_DIR"; then
                    ((success_count++))
                fi
            done
        done
    else
        # Parallel execution
        run_parallel_experiments "$PARALLEL_JOBS"
        success_count=$total_experiments  # Assume all succeeded for parallel execution
    fi
    
    # Calculate and display execution time
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    hours=$((execution_time / 3600))
    minutes=$(((execution_time % 3600) / 60))
    seconds=$((execution_time % 60))
    
    print_info "=============================="
    print_info "Execution completed!"
    print_info "Total time: ${hours}h ${minutes}m ${seconds}s"
    print_info "Results saved in: $OUTPUT_DIR"
    print_info "Logs saved in: $LOG_DIR"
    
    # Show summary of results
    print_info "=============================="
    print_info "Results summary:"
    for benchmark in "${BENCHMARKS[@]}"; do
        for method in "${METHODS[@]}"; do
            # Build result file name based on method
            case "$method" in
                "itergen")
                    result_file="${OUTPUT_DIR}/${benchmark}_${MODEL}_${method}_iter${ITERATIONS}_evaluation_results.jsonl"
                    ;;
                "qa")
                    result_file="${OUTPUT_DIR}/${benchmark}_${MODEL}_${method}_top${TOPK}_evaluation_results.jsonl"
                    ;;
                *)
                    result_file="${OUTPUT_DIR}/${benchmark}_${MODEL}_${method}_evaluation_results.jsonl"
                    ;;
            esac
            
            if [ -f "$result_file" ]; then
                total_lines=$(wc -l < "$result_file")
                print_success "${benchmark} + ${method}: $total_lines results"
            else
                print_warning "${benchmark} + ${method}: No results file found"
            fi
        done
    done
}

# Run main function
main "$@"
