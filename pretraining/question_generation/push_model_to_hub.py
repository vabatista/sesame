
## Need to login in the CLI first
## huggingface-cli login

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define your model and tokenizer
model_name = "vabatista/t5-small-answer-extraction-en"  # Replace with your desired model name
model_folder = "../../saved-models/t5-small-ans-ext/"  # Replace with the path to your model folder

# Initialize the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)
tokenizer = AutoTokenizer.from_pretrained(model_folder)

# Push the model to the repository
model.push_to_hub(repo_id=model_name)
tokenizer.push_to_hub(repo_id=model_name)
print(f"Model '{model_name}' has been pushed to the Hugging Face Model Hub.")
