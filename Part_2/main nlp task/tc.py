from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import numpy as np
import torch

raw_datasets = load_dataset("conll2003" , trust_remote_code=True)

# print(raw_datasets["train"][0]["tokens"])
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']

# print(raw_datasets["train"][0]["ner_tags"])
# [3, 0, 7, 0, 0, 0, 7, 0, 0]

ner_feature = raw_datasets["train"].features["ner_tags"]
# print(ner_feature)
# Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], id=None), length=-1, id=None) 

label_names = ner_feature.feature.names
# print(label_names)
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
# print(inputs.tokens())
# ['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']

# print(inputs.word_ids())
# [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]

## Self written function , this function has a flaw that gave me error
def align_labels_with_tokens(labels, word_ids):
    word_ids = word_ids[1:len(word_ids)-1]
    new_labels = []
    new_labels.append(-100)
    for i,word_id in enumerate(word_ids):
        if (word_ids[i] == word_ids[i-1] and i!=0 and labels[word_id]!=0):
            new_labels.append(labels[word_id]+1)

        else:
            new_labels.append(labels[word_id])

    new_labels.append(-100)

    return new_labels

# print(align_labels_with_tokens(raw_datasets["train"][0]["ner_tags"],inputs.word_ids()))
# [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100] -> result of function made by me.
# [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100] -> correct answer

## Function in the hugging face NLP course
def align_labels_with_tokens_1(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
# print(labels)
# print(align_labels_with_tokens_1(labels, word_ids))
# print(align_labels_with_tokens(labels, word_ids))  

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens_1(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_dataset = raw_datasets.map(tokenize_and_align_labels,batched=True,remove_columns=raw_datasets["train"].column_names)

# print(tokenized_dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 14041
#     })
#     validation: Dataset({
#         features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 3250
#     })
#     test: Dataset({
#         features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 3453
#     })
# })

metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    # for computing evaluation metrices we don't need to consider special tokens hence ignoring -100 , and tracking -100 from labels of tokenized dataset
    # writing the below function because seqeval takes input the string not integer to compute classification metrics used , so converting int labels to string labels from prediction as well as from true labels

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,id2label=id2label,label2id=label2id)

# Check if CUDA is available and move the model to GPU
if torch.cuda.is_available():
    model.cuda()
else:
    raise SystemError("CUDA is not available. Make sure you've configured PyTorch with CUDA support.")

# This time training from the high level api 

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=True, # keeping push to hub true to upload it to hub
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

trainer.push_to_hub(commit_message="Training complete")

# The training was completed on GPU 
# Results after 5 epochs

# Epoch	Training Loss	Validation Loss	Precision	Recall	F1	Accuracy
# 1	0.077500	0.069376	0.891153	0.927297	0.908866	0.981677
# 2	0.037700	0.070697	0.924547	0.944463	0.934399	0.984959
# 3	0.024300	0.067120	0.928053	0.946483	0.937177	0.985459
# 4	0.014500	0.073398	0.935265	0.950690	0.942914	0.985901
# 5	0.006000	0.074080	0.934115	0.952036	0.942990	0.986652




        






