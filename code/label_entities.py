import spacy
import json

# Load a spaCy model
nlp = spacy.load("en_core_web_sm")

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

text = read_text_file("split_texts/Appointment With Death.txt")

# Process the text with spaCy to create a Doc
doc = nlp(text)

character_names = []
for ent in doc.ents:
    if ent.label_ == "PERSON":
        character_names.append(ent.text)
character_names = list(set(character_names))
character_names = [s.replace("\n", "") for s in character_names]

characters = [{'name': name} for name in character_names]
with open('ner_results.json', 'w') as f:
    json.dump(characters, f)