import json
from vllm import LLM, SamplingParams
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

model = LLM("./runs/Llama2_SORSA_r128/checkpoint", dtype="bfloat16", tokenizer="meta-llama/Llama-2-7b-hf")

sampling_params = SamplingParams(
    top_p=0.9,
    max_tokens=2048,
)

def run_human_eval():
    problems = read_problems()
    
    # Prepare prompts for batch inference
    prompts = [f"{problem['prompt']}" for problem in problems.values()]
    print(prompts[0])
    task_ids = list(problems.keys())

    # Generate completions in batch
    outputs = model.generate(prompts, sampling_params)

    # Process the outputs
    completions = {}
    for task_id, output in zip(task_ids, outputs):
        completion = output.outputs[0].text
        completions[task_id] = completion

    # Save completions to a file
    with open("completions.jsonl", "w") as f:
        for task_id, completion in completions.items():
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

    # Evaluate the completions
    results = evaluate_functional_correctness("completions.jsonl")
    
    print(f"Pass@1: {results['pass@1']:.4f}")

if __name__ == "__main__":
    run_human_eval()
