import os
from huggingface_hub._login import _login

# Read the token from the file
with open("/.huggingface/token/hf_token.txt") as f:
    token = f.read().strip()

# Authenticate with Hugging Face
_login(token=token, add_to_git_credential=False)
