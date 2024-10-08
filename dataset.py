#  ------------------------------------------------------------------------------------------
#  SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from fraction import Fraction
import inference.util as util


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    """
    Extract answer from GSM-8K format completion.
    """
    text = completion.split("The answer is: ")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None


def extract_answer(completion):
    match = re.compile(r"#### (\-?[0-9\.\,]+)").search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        match_str = re.sub(r",(?=\d)", "", match_str)
        return match_str
    else:
        return None


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def preprocess_metamathqa(item, tokenizer, max_length):
    # Identical replica with PiSSA
    question = item["query"]
    completion = item["response"]
    text = f"{tokenizer.bos_token}Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"
    target_text = f"{completion}{tokenizer.eos_token}"
    query = tokenizer.encode_plus(
        text=text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    ans = tokenizer.encode_plus(
        text=f"{target_text}",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    length = query["input_ids"].size(-1)

    input_ids = torch.concat(
        (query["input_ids"].squeeze(0), ans["input_ids"].squeeze(0))
    )
    attention_mask = torch.concat(
        (query["attention_mask"].squeeze(0), ans["attention_mask"].squeeze(0))
    )

    labels = torch.full_like(input_ids, fill_value=-100)
    labels[length:] = input_ids[length:]

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return batch


def collate_fn(batch, tokenizer):

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_length = max(x.size(0) for x in input_ids)

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    batch = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }

    return batch


def preprocess_codefeedback(example, tokenizer, max_length=512):
    input_ids = []
    labels = []

    for message in example["messages"]:
        role = message["role"]
        content = message["content"]

        if role == "user":
            # For user messages, add to input_ids and set labels to -100
            user_ids = tokenizer.encode(
                f"@@ Instruction\n{content}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids.extend(user_ids)
            labels.extend([-100] * len(user_ids))
        elif role == "assistant":
            # For assistant messages, add to both input_ids and labels, append EOS token
            assistant_ids = tokenizer.encode(
                f"@@ Response\n{content}{tokenizer.eos_token}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids.extend(assistant_ids)
            labels.extend(assistant_ids)

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def preprocess_codefeedback_instructed(example, tokenizer, max_length=512):
    # Self implementation
    input_ids = []
    labels = []

    user_ids = tokenizer.encode(
        f"@@ Instruction\n{example['query']}\n\n@@ Response\n",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids.extend(user_ids)
    labels.extend([-100] * len(user_ids))

    assistant_ids = tokenizer.encode(
        f"{example['answer']}{tokenizer.eos_token}",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids.extend(assistant_ids)
    labels.extend(assistant_ids)

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
