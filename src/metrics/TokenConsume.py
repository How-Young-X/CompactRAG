from argparse import ArgumentParser
import jsonlines




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="benchark name")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--method", type=str, required=True, help="method name")
    args = parser.parse_args()

    benchark = args.benchmark
    model = args.model  
    method = args.method

    input_path = f"data/results/{benchark}_{model}_{method}_evaluation_results.jsonl"

    with jsonlines.open(input_path, "r") as reader:
        records = [record for record in reader]
    token_consumes = [record["total_tokens"] for record in records]
    avg_token_consume = sum(token_consumes) / len(token_consumes)
    print(f"{method} Token Consume: {avg_token_consume}")