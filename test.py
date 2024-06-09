import pandas as pd
from dataset import extract_answer_number, extract_answer, remove_boxed
from inference.util import is_equiv, last_boxed_only_string
from vllm import LLM, SamplingParams


def test_gsm(
    model,
    tokenizer,
    dataset,
    precision
):
    correct = 0
    dataset = pd.read_parquet(dataset).to_dict(orient="records")
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


def test_math(
    model,
    tokenizer,
    dataset,
    precision
):
    correct = 0
    dataset = pd.read_json(dataset, lines=True).to_dict(orient="records")
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
