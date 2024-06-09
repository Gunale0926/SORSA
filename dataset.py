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


class MetaMathQADataset(Dataset):
    # This dataset is used for training ONLY!
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_json(file_path).to_dict(orient="records")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["query"]
        completion = item["response"]
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"
        target_text = f"{completion}{self.tokenizer.eos_token}"
        query = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        full = self.tokenizer.encode_plus(
            text=f"{text}{target_text}",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, fill_value=-100)
        length = query["input_ids"].size(-1)
        labels[length:] = input_ids[length:]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_length = max(x.size(0) for x in input_ids)

        if self.tokenizer.padding_side == "right":
            # Pad sequences to the right
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            labels_padded = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
        else:
            # Pad sequences to the left
            input_ids_padded = []
            attention_mask_padded = []
            labels_padded = []

            for input_id, attn_mask, label in zip(input_ids, attention_mask, labels):
                padding_length = max_length - input_id.size(0)

                input_ids_padded.append(
                    torch.cat(
                        (
                            torch.full(
                                (padding_length,),
                                self.tokenizer.pad_token_id,
                                dtype=torch.long,
                            ),
                            input_id,
                        ),
                        dim=0,
                    )
                )
                attention_mask_padded.append(
                    torch.cat(
                        (torch.zeros(padding_length, dtype=torch.long), attn_mask),
                        dim=0,
                    )
                )
                labels_padded.append(
                    torch.cat(
                        (torch.full((padding_length,), -100, dtype=torch.long), label),
                        dim=0,
                    )
                )

            input_ids_padded = torch.stack(input_ids_padded, dim=0)
            attention_mask_padded = torch.stack(attention_mask_padded, dim=0)
            labels_padded = torch.stack(labels_padded, dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
