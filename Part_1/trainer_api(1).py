#  The description will be according to the fine tuning task I performed 
#  Fine tuning the bert model over mrpc data , it takes 2 input or a pair of sentence and dteremine whether the two of them are equivalent in meaning or not. 

import torch
import os
import numpy as np
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
from datasets import load_dataset

# This code follows the fine-tuning done by the high-level TrainerAPI of Hugging Face hub that means for most of the things behind the scene , the HF does things for us and we really don't write much code.
# This ofc comes with a disadvantage we have less things in our control to fine tune the model.

checkpoint = "bert-base-uncased" # Base-Transformer
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Same checkpoint for tokenizer.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function_1(example):
    return tokenizer(example["sentence"] , truncation=True)

def tokenize_function_2(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True) # This is executed for mrpc

glue_dataset = input("Please enter on which glue dataset you want to perform pre-processing : ") # Enter the dataset , which have sequence classification head and 2 labels , in the later part of the code I have fixed that.

def glue_preprocess(checkpoint,tokenizer,glue_dataset):

    raw_datasets = load_dataset("glue", glue_dataset)

    if glue_dataset in ("sst2", "cola"):
        tokenized_datasets = raw_datasets.map(tokenize_function_1, batched=True)
    else:
        tokenized_datasets = raw_datasets.map(tokenize_function_2, batched=True)

    return tokenized_datasets    

tokenized_datasets =  glue_preprocess(checkpoint,tokenizer,glue_dataset)

def compute_metrics(eval_preds): # Matrices for evaluation
    metric = evaluate.load("glue", glue_dataset)
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) # Defining the head , the inputs to this will be the outputs given by base transformer model which is of shape - (batch_size,max seqence_length,hidden_size=768(for bert))

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
) # High level Trainer API which will do the fine tuning for us.

trainer.train() # Start Training.
# The file tarining_code(2) , we don't use Trainer API and writes the code for fine tuning by ourselves setting epcohs and training and evaluation loop.


