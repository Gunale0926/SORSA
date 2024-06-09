from torch._C import device
import torch
import subprocess
from dataset import extract_answer_number
from torch.utils.data import DataLoader
from tqdm import tqdm
from inference.util import is_equiv


def test_gsm(
    model,
    tokenizer,
    device: device,
    dataset: DataLoader,
):
    model.eval()
    correct = 0
    skip = 0
    progress_bar = tqdm(enumerate(dataset), total=len(dataset), initial=skip)
    for idx, batch in enumerate(dataset):
        if idx < skip:
            continue
        bsz = batch["query"].size(0)
        query = batch["query"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids=query,
                attention_mask=attention_mask,
                max_length=1024,
                eos_token_id=tokenizer.eos_token_id,
                # do_sample=True,
                # temperature=0,
                top_p=1,
                num_return_sequences=1,
            )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(outputs)):
            ans = batch["answer"][i]
            extract = extract_answer_number(outputs[i].strip())
            if extract is not None and ans is not None:
                if float(extract) == float(ans) or extract == ans:
                    correct += 1
        progress_bar.set_description(f"{correct / (bsz * (idx+1)) * 100}")
        progress_bar.update(1)
    print(f"{correct / 1320 * 100 :.4f}% Correct!")


def test_math(
    model,
    tokenizer,
    device: device,
    dataset: DataLoader,
):
    model.eval()
    correct = 0
    skip = 0
    progress_bar = tqdm(enumerate(dataset), total=len(dataset), initial=skip)
    for idx, batch in enumerate(dataset):
        if idx < skip:
            continue
        bsz = batch["query"].size(0)
        query = batch["query"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids=query,
                attention_mask=attention_mask,
                max_length=2048,
                eos_token_id=tokenizer.eos_token_id,
                # do_sample=True,
                # temperature=0,
                top_p=1,
                num_return_sequences=1,
            )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(outputs)):
            ans = batch["answer"][i]
            extract = extract_answer_number(outputs[i].strip())
            print(ans, extract)
            if is_equiv(str(ans), str(extract)):
                correct += 1
        progress_bar.set_description(f"{correct / (bsz * (idx+1)) * 100}")
        progress_bar.update(1)
    print(f"{correct / 5000 * 100 :.4f}% Correct!")
