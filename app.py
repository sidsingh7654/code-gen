# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qiiTlO6J3fZhbm5bUW5ZBPEgeTpszFie
"""

import os
from dotenv import load_dotenv
from github import Github
import re
from datasets import Dataset

# Load environment variables from the .env file
load_dotenv()

# Get the GitHub token from the environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Create a GitHub client instance
g = Github(GITHUB_TOKEN)

# Access the repository (public or private depending on token permissions)
repo = g.get_repo("openai/gym")

# Example: print the repository name
print(f"Repository: {repo.name}")



# Initialize GitHub API (Ensure you have authentication)
g = Github("your_github_access_token")

# Specify the repository
repo = g.get_repo("openai/gym")

# Function to extract Python functions from a script
def extract_functions_from_code(code):
    if not isinstance(code, str):  # Ensure input is a string
        print("Error: Code is not a string")
        return []
    
    pattern = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\):")  # Valid function name regex
    functions = pattern.findall(code)

    if not functions:
        print("Warning: No functions found in the code.")

    return functions

# Fetch Python files from the repository
python_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    elif file_content.path.endswith(".py"):
        python_files.append(file_content)

# Extract functions and create dataset
data = {"code": [], "function_name": []}
for file in python_files:
    code = file.decoded_content.decode("utf-8")

    # Debugging print statement
    print(f"\nProcessing file: {file.path}\nFirst 200 characters:\n{code[:200]}\n")

    functions = extract_functions_from_code(code)
    for function in functions:
        data["code"].append(code)
        data["function_name"].append(function)

# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Save the dataset to disk
dataset.save_to_disk("code_generation_dataset")

print("Dataset created and saved to disk.")



from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# set the pad_token to eos_token or add a new pad token
tokenizer.pad_token = tokenizer.eos_token

# load the dataset
dataset = load_from_disk("code_generation_dataset")

# split the dataset into training and test sets
dataset = dataset.train_test_split(test_size=0.3)

# preprocess the dataset
def preprocess_function(examples):
       return tokenizer(examples['code'], truncation=True, padding='max_length', max_length=128) # Reduced max_length

# preprocess the dataset
def preprocess_function(examples):
    # Tokenize the code
    tokenized_examples = tokenizer(examples['code'], truncation=True, padding='max_length', max_length=128)
    # Create labels - shifted input_ids
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples

Ftokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=0.5, # This line was improperly indented and has been corrected
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=2, # Accumulate gradients over 2 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()

# define a function to generate code using the fine-tuned model
def generate_code(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# test the model with a code generation prompt
prompt = "def merge_sort(arr):"
generated_code = generate_code(prompt)

print("Generated Code:")
print(generated_code)


import streamlit as st
from transformers import pipeline

st.title("AI Code Generator")
# Replace "Salesforce/codegen-350M-mono" with the actual model you fine-tuned
# or the path to the folder containing the saved model
# Example: model = pipeline("text-generation", model="./results")
model = pipeline("text-generation", model="Salesforce/codegen-350M-mono")

prompt = st.text_area("Enter prompt")
if st.button("Generate"):
    result = model(prompt, max_length=200)
    st.code(result[0]['generated_text'])

