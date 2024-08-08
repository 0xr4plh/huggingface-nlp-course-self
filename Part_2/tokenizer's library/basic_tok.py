from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("code_search_net", "python",trust_remote_code=True)

# # print(raw_datasets["train"])

# Dataset({
#     features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
#     num_rows: 412178
# })

# print(raw_datasets["train"][0: 0 + 10]["whole_func_string"])

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
print(tokens)

