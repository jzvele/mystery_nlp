from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import torch
    
def read_file_in_chunks(filename, chunk_size=256):
        # Read the entire content of the file
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into tokens based on whitespace
    tokens = content.split()
    
    # Yield successive chunks of tokens of size chunk_size, joined as a single string
    for i in range(0, len(tokens), chunk_size):
        yield ' '.join(tokens[i:i + chunk_size])


model_name = "martincc98/srl_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Accessing the label to index mapping
label_map = tokenizer.get_vocab()

texts = []
for chunk in read_file_in_chunks("split_texts/Appointment With Death.txt"):
        texts.append(chunk)

text = texts[0]

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=-1)
predicted_indices = torch.argmax(probabilities, dim=-1)

# Convert indices to labels using the id2label mapping provided in the config
predicted_labels = [model.config.id2label[idx.item()] for idx in predicted_indices.squeeze()]

# Print token-role pairs
tokens = tokenizer.tokenize(text)
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
    print(type(label))

'''
This did not work. For whatever reason, the model consistently predicted the same label (O) for all tokens in the text. With more time and computational resources, I might have been able to train it.
Will try using GPT instead.
'''