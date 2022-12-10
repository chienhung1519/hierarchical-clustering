from typing import List
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

def bert_embedding(model, tokenizer, x: List[str], y: List[str]) -> pd.DataFrame:
    """Embbeding text to hidden vectors."""
    
    model.to("cuda")
    model.eval()

    # Initialize dataset and dataloader
    def process_text(examples):
        model_inputs = tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)
        return model_inputs

    dataset = Dataset.from_dict({"text": x})
    dataset = dataset.map(process_text, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=dataset.column_names)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)

    # Collect bert outputs
    hiddens = []
    for batch in tqdm(dataloader, desc="Generating hiddens"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        hiddens.extend(outputs.pooler_output.detach().cpu().numpy().tolist())

    # Get represent hidden. 
    # If y is larger than the y categories, collect and average the hiddens with the same category.
    rtn = []
    y_cat = sorted(list(set(y)))
    if len(y) == len(y_cat):
        rtn = [hiddens[y.index(y_i)] for y_i in y_cat]
    else:
        for y_i in y_cat:
            collected_hiddens = [hiddens[i] for i, y_j in enumerate(y) if y_i == y_j]
            represent_hidden = np.mean(collected_hiddens, axis=0)
            rtn.append(represent_hidden.tolist())

    return pd.DataFrame(rtn, index=y_cat)