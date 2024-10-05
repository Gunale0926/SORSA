#  ------------------------------------------------------------------------------------------
#  SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------

import pandas as pd
from dataset import extract_answer_number, extract_answer, remove_boxed
from inference.util import is_equiv, last_boxed_only_string
from tqdm import tqdm


def test_gsm(model, tokenizer, precision):
    from vllm import LLM, SamplingParams

    correct = 0
    dataset = pd.read_parquet("datasets/test-00000-of-00001.parquet").to_dict(
        orient="records"
    )
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024)
    llm = LLM(model=model, tokenizer=tokenizer, dtype=precision)
    query = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
        for data in dataset
    ]
    outputs = llm.generate(query, sampling_params=sampling_params)
    for i in range(len(outputs)):
        ans = extract_answer(dataset[i]["answer"])
        extract = extract_answer_number(outputs[i].outputs[0].text)
        if extract is not None and ans is not None:
            if float(extract) == float(ans) or extract == ans:
                correct += 1
    print(f"Final GSM-8K Result: {correct / 1319 * 100 :.4f}% Correct!")


def extract_ans_for_math(completion):
    split_ans = completion.split("The answer is: ")
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split(".\n")[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        return extract_ans
    else:
        return None


def test_math(model, tokenizer, precision):
    from vllm import LLM, SamplingParams

    correct = 0
    dataset = pd.read_json("datasets/MATH_test.jsonl", lines=True).to_dict(
        orient="records"
    )
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=2048)
    llm = LLM(model=model, tokenizer=tokenizer, dtype=precision)
    query = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n### Response: Let's think step by step."
        for data in dataset
    ]
    outputs = llm.generate(query, sampling_params=sampling_params)
    for i in range(len(outputs)):
        answer = remove_boxed(last_boxed_only_string(dataset[i]["output"]))
        extract = extract_ans_for_math(outputs[i].outputs[0].text)
        if is_equiv(str(answer), str(extract)):
            correct += 1
    print(f"Final MATH Result: {correct / 5000 * 100 :.4f}% Correct!")


def test_gsm_rwkv(model, precision):
    import os

    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS

    model = RWKV(model=model, strategy="cuda " + precision)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    args = PIPELINE_ARGS(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        token_stop=[0],  # stop generation whenever you see any token here
        chunk_len=4096,
    )
    dataset = pd.read_parquet("datasets/test-00000-of-00001.parquet").to_dict(
        orient="records"
    )
    query = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
        for data in dataset
    ]
    correct = 0
    progress = tqdm(total=len(query))
    for i, q in enumerate(query):
        output = pipeline.generate(q, token_count=1024, args=args)
        ans = extract_answer(dataset[i]["answer"])
        extract = extract_answer_number(output)
        if extract is not None and ans is not None:
            if float(extract) == float(ans) or extract == ans:
                correct += 1
        progress.update(1)
        progress.set_description(f"Correct: {correct / (i + 1) * 100 :.4f}%")
    progress.close()

    print(f"Final GSM-8K Result: {correct / 1319 * 100 :.4f}% Correct!")


def test_math_rwkv(model, precision):
    import os

    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS

    model = RWKV(model=model, strategy="cuda " + precision)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    args = PIPELINE_ARGS(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        token_stop=[0],  # stop generation whenever you see any token here
        chunk_len=4096,
    )
    dataset = pd.read_json("datasets/MATH_test.jsonl", lines=True).to_dict(
        orient="records"
    )
    query = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n### Response: Let's think step by step."
        for data in dataset
    ]
    correct = 0
    progress = tqdm(total=len(query))
    for i, q in enumerate(query):
        output = pipeline.generate(q, token_count=2048, args=args)
        answer = remove_boxed(last_boxed_only_string(dataset[i]["output"]))
        extract = extract_ans_for_math(output)
        if is_equiv(str(answer), str(extract)):
            correct += 1
        progress.update(1)
        progress.set_description(f"Correct: {correct / (i + 1) * 100 :.4f}%")
    progress.close()

    print(f"Final MATH Result: {correct / 5000 * 100 :.4f}% Correct!")


def test_human_eval(model, tokenizer, precision):
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness
    from vllm import LLM, SamplingParams
    import re
    import json

    model = LLM(
        model,
        dtype=precision,
        tokenizer=tokenizer,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=2048,
    )

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


def test_human_eval_rwkv(model, precision):
    import os

    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    import re
    import json

    model = RWKV(model=model, strategy="cuda " + precision)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    args = PIPELINE_ARGS(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        token_stop=[0],  # stop generation whenever you see any token here
        chunk_len=4096,
    )
    problems = read_problems()

    # Prepare prompts for batch inference
    prompts = [
        # You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n
        f"@@ Instruction\nHere is the given code to do completion:\n```python\n{problem['prompt']}\n```\n\nPlease continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.\n\nPlease return all completed codes in one code block.\nThis code block should be in the following format:\n'''python\n# Your codes here\n'''\n\n@@ Response\n"
        # f"@@ Instruction\n{problem['prompt']}\n\n@@ Response\n "
        for problem in problems.values()
    ]
    task_ids = list(problems.keys())

    # Process the outputs
    completions = {}
    progress = tqdm(total=len(task_ids))
    for task_id, prompt in zip(task_ids, prompts):
        output = pipeline.generate(prompt, token_count=2048, args=args)
        print(output)
        completion = output
        pattern = r"```python\s*([\s\S]*?)\s```"
        match = re.search(pattern, completion)
        if match is not None:
            match = match.group(1)
        else:
            match = completion
        completions[task_id] = match
        progress.update(1)
    progress.close

    # Save completions to a file
    with open("completions.jsonl", "w") as f:
        for task_id, completion in completions.items():
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

    # Evaluate the completions
    results = evaluate_functional_correctness("completions.jsonl")

    print(f"Pass@1: {results['pass@1']:.4f}")
