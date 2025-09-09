import json


def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to the JSONL result file")
    args = parser.parse_args()

    results_file = args.result_path
    print(results_file)

    infilling_results = {
        "count": 0,
        "exact_matches": 0,
    }

    with open(results_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            completion = data.get("completion")
            ground_truth = data.get("ground_truth_middle")

            # Process model generation
            # https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/base/benchmarks/fim-bench/hm_fim/humaneval_fim.py#L37
            if completion.startswith("\n"):
                response = completion.split("\n")[1]
            elif completion.startswith(" \n"):
                response = completion.split("\n")[1]
            else:
                response = completion.split("\n")[0]

            response = response.strip()
            canonical_solution = ground_truth.strip()

            # Update statistics
            infilling_results["count"] += 1
            if response == canonical_solution:
                infilling_results["exact_matches"] += 1

    exact_match_rate = (infilling_results["exact_matches"] / infilling_results["count"]) * 100

    final_results = {
        "count": infilling_results["count"],
        "exact_match_rate": exact_match_rate,
    }
    print(final_results)
