import spacy
import coreferee
import json
from label_entities import read_text_file

text = read_text_file("split_texts/Appointment With Death.txt")

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('coreferee')

# Load NER results
with open('ner_results.json', 'r') as f:
    entities = json.load(f)

# Assuming we process the same text (you could store the text in JSON as well)
doc = nlp(text)


# Accessing the coreference chains
if doc._.coref_chains:
    doc._.coref_chains.print()
else:
    print("No coreference chains found.")

for chain in doc._.coref_chains:
    print(f"Chain {chain.index}: {chain.pretty_representation}")
    for mention in chain:
        # Accessing the span for each mention using the indices
        span = doc[mention[0]:mention[-1] + 1]  # Create a span from indices
        print(f"  - Mention: '{span.text}', Indices: {mention}")


'''
This worked but did not provide very complex output; it primarily revealed pronouns and their antecedents.
'''