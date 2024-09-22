import json
import re
from vllm import LLM, SamplingParams
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness
import argparse

parser = argparse.ArgumentParser(
    prog="Human Evaluation",
)

parser.add_argument(
    "--checkpoint", type=str, default="./runs/mistral_SORSA_r64/checkpoint"
)
parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--fp16", action="store_true")

args = parser.parse_args()

model = LLM(
    model_name=args.checkpoint,
    dtype="bfloat16" if args.bf16 else "float16" if args.fp16 else "float32",
    tokenizer_name=args.tokenizer,
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=2048,
)


def run_human_eval():
    problems = read_problems()

    # Prepare prompts for batch inference
    prompts = [
        # You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n
        f"@@ Instruction\nHere is the given code to do completion:\n```python\n{problem['prompt']}\n```\n\nPlease continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.\n\nPlease return all completed codes in one code block.\nThis code block should be in the following format:\n'''python\n# Your codes here\n'''\n\n@@ Response\n"
        # f"@@ Instruction\n{problem['prompt']}\n\n@@ Response\n "
        for problem in problems.values()
    ]
    task_ids = list(problems.keys())

    # Generate completions in batch
    outputs = model.generate(prompts, sampling_params)

    # Process the outputs
    completions = {}
    for task_id, output in zip(task_ids, outputs):
        completion = output.outputs[0].text
        pattern = r"```python\s*([\s\S]*?)\s```"
        match = re.search(pattern, completion)
        if match is not None:
            match = match.group(1)
        else:
            match = completion
        completions[task_id] = match

    # Save completions to a file
    with open("completions.jsonl", "w") as f:
        for task_id, completion in completions.items():
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

    # Evaluate the completions
    results = evaluate_functional_correctness("completions.jsonl")

    print(f"Pass@1: {results['pass@1']:.4f}")


if __name__ == "__main__":
    run_human_eval()
