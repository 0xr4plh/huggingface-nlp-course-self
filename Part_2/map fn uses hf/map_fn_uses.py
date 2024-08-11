from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_from_disk
import html

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# print(drug_dataset)

# DatasetDict({
#     train: Dataset({
#         features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
#         num_rows: 161297
#     })
#     test: Dataset({
#         features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
#         num_rows: 53766
#     })
# })

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))

# print(drug_sample[:3])

drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", new_column_name="patient_id")

drug_dataset = drug_dataset.filter(lambda x : x["condition"] is not None)

def lower_case(example):
    return {"condition" : example["condition"].lower()}

def review_length(example):
    return {"review_length" : len(example["review"])}

drug_dataset = drug_dataset.map(lower_case)
drug_dataset = drug_dataset.map(review_length)

# print(drug_dataset["train"]["condition"][:3])---> ['left ventricular dysfunction', 'adhd', 'birth control']
# print(drug_dataset["train"]["review_length"][-1]) ---> 347

# print(len(drug_dataset["train"]))---> 160398
# print(len(drug_dataset["test"]))---> 53471

drug_dataset = drug_dataset.filter(lambda x : x["review_length"] > 30)

# print(len(drug_dataset["train"]))---> 158429
# print(len(drug_dataset["test"]))---> 52825

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset.set_format("pandas")

# print(tokenized_dataset["train"][:10])
# print(tokenized_dataset["train"]["review_length"][:10])

drug_dataset.set_format("pandas")
train_df = drug_dataset["train"][:]
frequencies = (train_df["condition"].value_counts().to_frame().reset_index().rename(columns={"index": "condition", "condition": "frequency"}))
print(frequencies.head())

#        frequency  count
# 0  birth control  28755
# 1     depression   8964
# 2           pain   5979
# 3        anxiety   5785
# 4           acne   5571

freq_dataset = Dataset.from_pandas(frequencies)
# print(freq_dataset)

# Dataset({
#     features: ['frequency', 'count'],
#     num_rows: 872
# })


drug_name_dataset = (
    train_df
    .groupby("drugName"))

# print(drug_name_dataset.head())

drug_dataset.reset_format()
tokenized_dataset.reset_format()

drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)

drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")

drug_dataset_clean["test"] = drug_dataset["test"]
# print(drug_dataset_clean)

drug_dataset_clean.save_to_disk("drug-reviews")

# Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████| 126743/126743 [00:02<00:00, 61858.91 examples/s]
# Saving the dataset (1/1 shards): 100%|███████████████████████████████████████████████████████████████| 31686/31686 [00:00<00:00, 83857.60 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████████████████████████████████████████████████████████| 52825/52825 [00:00<00:00, 366019.00 examples/s]

drug_dataset_reloaded = load_from_disk("drug-reviews")
# print(drug_dataset_reloaded)

for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

data_files_1 = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}

drug_dataset_reloaded_1 = load_dataset("json", data_files=data_files_1)   
# print(drug_dataset_reloaded_1) 
