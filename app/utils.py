# app/utils.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_text_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model

def generate_description(tokenizer, model, features):
    # Placeholder for generating text based on image features
    input_ids = tokenizer.encode("Describe the X-ray image:", return_tensors='pt')
    output = model.generate(input_ids, max_length=50)
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    return description
