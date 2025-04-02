# -*- coding: utf-8 -*-
import re
import os
from dotenv import load_dotenv
from github import Github
from datasets import Dataset

# Load environment variables from the .env file
load_dotenv()

# Get the GitHub token from the environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Check if the token exists
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set. Please add it to your .env file.")

# Create a GitHub client instance
g = Github(GITHUB_TOKEN)

# Access the repository (ensure token has correct permissions)
repo = g.get_repo("openai/gym")
print(f"✅ Connected to Repository: {repo.name}")

# Function to extract Python function names from code
def extract_functions_from_code(code):
    if not isinstance(code, str):
        print("❌ Error: `code` is not a string, received:", type(code))
        return []
    
    pattern = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\):")
    functions = pattern.findall(code)
    
    if not functions:
        print("⚠️ Warning: No functions found in the code.")
    
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
    try:
        if not hasattr(file, "decoded_content"):
            print(f"⚠️ Warning: No content found for {file.path}")
            continue

        code = file.decoded_content.decode("utf-8")

        if not code.strip():
            print(f"⚠️ Skipping empty file: {file.path}")
            continue

        print(f"\n✅ Processing file: {file.path}\nFirst 200 characters:\n{code[:200]}\n")

        functions = extract_functions_from_code(code)
        for function in functions:
            data["code"].append(code)
            data["function_name"].append(function)

    except Exception as e:
        print(f"❌ Error processing file {file.path}: {e}")

# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Save the dataset to disk
dataset.save_to_disk("code_generation_dataset")

print("✅ Dataset created and saved to disk.")

# Load and fine-tune the model
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_from_disk("code_generation_dataset")

# Split the dataset into training and test sets
dataset = dataset.train_test_split(test_size=0.3)

# Preprocess the dataset
def preprocess_function(examples):
    tokenized_examples = tokenizer(examples['code'], truncation=True, padding='max_length', max_length=128)
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=0.5, 
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()

# Define a function to generate code using the fine-tuned model
def generate_code(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# Test the model with a code generation prompt
prompt = "def merge_sort(arr):"
generated_code = generate_code(prompt)

print("📝 Generated Code:")
print(generated_code)

# Deploy with Streamlit
import streamlit as st
from transformers import pipeline

st.title("AI Code Generator")

model = pipeline("text-generation", model="Salesforce/codegen-350M-mono")

prompt = st.text_area("Enter prompt")
if st.button("Generate"):
    result = model(prompt, max_length=200)
    st.code(result[0]['generated_text'])
