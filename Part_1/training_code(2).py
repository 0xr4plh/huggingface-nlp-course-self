# Decided the base-transformer - bert-base-uncased
# Decided the tokenizer - same used for bert-base-uncased
# Fine-tuned for mrpc dataset.
# Removed the columns which have strings in them and renamed label to labels , as the model accepts labels not label
# Loaded the data in Pytorch dataloader in train dataloader and test dataloader with dynamic padding using collate function
# Defined head AutoModelForSequenceClassification with 2 labels
# Defined optimizer and loss function with learning rate and other hyper-parameters.
# Set the number of epochs and wrote the training and testing loop 
# Evaluated the fine-tuned model from the metrices like accuracy and f1-score.
# Saved the model and tokenizer in a directory to be later uploaded on HF_HUB.

import torch
import os
import numpy as np
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from torch.utils.data import DataLoader
import evaluate
from datasets import load_dataset

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function_1(example):
    return tokenizer(example["sentence"], truncation=True)

def tokenize_function_2(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

glue_dataset = input("Please enter on which glue dataset you want to perform pre-processing(for now the entire code is ""mrpc"" centric so enter mrpc ) : ")

def glue_preprocess(checkpoint, tokenizer, glue_dataset):
    raw_datasets = load_dataset("glue", glue_dataset)
    if glue_dataset in ("sst2", "cola"):
        tokenized_datasets = raw_datasets.map(tokenize_function_1, batched=True)
    else:
        tokenized_datasets = raw_datasets.map(tokenize_function_2, batched=True)
    return tokenized_datasets    

tokenized_datasets = glue_preprocess(checkpoint, tokenizer, glue_dataset)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=True, batch_size=8, collate_fn=data_collator)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

save_directory = "bert-finetuned-mrpc"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
