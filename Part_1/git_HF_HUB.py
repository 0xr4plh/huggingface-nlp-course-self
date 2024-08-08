# This code , I run this on python console to add , commit and push to Hugging face Hub just like we perform these actions on github.
# In GitHub we write these commands on command line and here to perform those same actions to HF_Hub , writing code in python console is more convinient
# The Python console is in the virtual environmental setup of Hugging face transformers , so executing these commands become easy , we can also commit to hugging face from command line too , some additional commands have to be passed then.
# I haven't used git and git-lfs (large file storage) , not needed also and seperate set-up and installation was required for testing them.

from huggingface_hub import Repository

repo = Repository("local-folder")

# Commit and push changes
repo.git_add()
repo.git_commit("Updated Auto-generated model card.")
repo.git_push()
