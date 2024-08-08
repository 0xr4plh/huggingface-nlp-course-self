from huggingface_hub import Repository
from transformers import AutoModelForSequenceClassification,AutoTokenizer

# The local-folder is the directory in my local machine which is the link between hub and my local machine , I have cloned from my account namespace/directory into local folder.
repo = Repository("local-folder",clone_from="amannagrawall002/bert-finetued-mrpc")

repo.git_pull()

checkpoint_directory = "bert-finetuned-mrpc" # The directory in which all the files of model(model architecture - config.json and model weights - in .bin file or model safetensors it's a big file.) and tokenizer are saved. This directory was used for saving the fine-tuned model and tokeniser used in the file training_code(2).py
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_directory) # Calling model from saved dir . from_pretrained searches for config and model wieghts file so proper path to model directory should be given.
tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory) # Similar to model , all important files of a tokenizer were saved in this directory.

model.save_pretrained(repo.local_dir) 
tokenizer.save_pretrained(repo.local_dir)

# model.save_pretrained(repo.local_dir): Saves the model’s configuration and weights to the local directory managed by the repo object.
# tokenizer.save_pretrained(repo.local_dir): Saves the tokenizer’s configuration and vocabulary to the same local directory.

repo.git_add()

# repo.git_add(): Stages all changes in the local repository directory (repo.local_dir) for commit. This includes the newly saved model and tokenizer files.

repo.git_commit("Added model and tokenizer")

# repo.git_commit("Added model and tokenizer"): Commits the staged changes with a commit message ("Added model and tokenizer").

repo.git_push()

# repo.git_push(): Pushes the committed changes to the remote repository on the Hugging Face Hub. This updates the remote repository with the latest changes from your local repository.