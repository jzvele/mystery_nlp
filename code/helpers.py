import os
import pickle
from label_utils import filename_mapping

relationship_categories = ["colleague", "stepmother", "stepfather", "stepchild", "stepsibling", "neighbor", "employer", "employee", "acquaintance", "friend", "lover", "father", "mother", "daughter", "son", "sibling", "spouse", "accomplice", "rival", "client", "patient", "co-conspirator", "romantic partner", "fiancé", "fiancée", "sibling-in-law","cousin", "niece", "nephew", "uncle", "aunt", "governess", "ward", "wife", "husband", "collaborator", "manipulator",'former lover', "mother-in-law", "servant", "stepson", "stepdaughter", "ex-fiancé", "assistant", "maid", "secretary", "love interest", "solicitor", "look-alike", "alleged father", "alleged son", "father's former lover's son", "grandfather", "trustee", "pseudonym", "blackmailer", "protector", "companion", "half-sibling", "housekeeper", "business associate", "doctor", "former employer", "former employee", "donor", "advisor", "butler", "gardener", "caretaker"]

def flatten_and_prepare_relationships(relationships, label_to_int, bidirectional_relationships):
    sentences = []
    encoded_labels = []
    
    # Helper to create a standard relationship sentence
    def make_sentence(person1, person2, relationship):
        return f"{person1} is {relationship} to {person2}"

    for person_a, links in relationships.items():
        for person_b, relation_a_to_b in links.items():
            # Check for a bidirectional relationship
            if person_b in relationships and person_a in relationships[person_b] and person_a in bidirectional_relationships:
                bidirectional_info = [d for d in bidirectional_relationships[person_a] if d['PersonB'] == person_b]
                if bidirectional_info:
                    correct_rel = bidirectional_info[0]
                    direction = correct_rel['CorrectChoice']
                    if direction == 'AtoB':
                        sentence = make_sentence(person_a, person_b, relation_a_to_b)
                        relation_label = relation_a_to_b
                    elif direction == 'BtoA':
                        sentence = make_sentence(person_b, person_a, relationships[person_b][person_a])
                        relation_label = relationships[person_b][person_a]

                    encoded_label = label_to_int.get(relation_label, -1)
                    if encoded_label == -1:
                        encoded_label = label_to_int.get('sibling', -1)
                    sentences.append(sentence)
                    encoded_labels.append(encoded_label)
                continue

            # Normal single direction relationship
            sentence = make_sentence(person_a, person_b, relation_a_to_b)
            encoded_label = label_to_int.get(relation_a_to_b, -1)
            if encoded_label == -1:
                encoded_label = label_to_int.get('sibling', -1)
            sentences.append(sentence)
            encoded_labels.append(encoded_label)
    
    return sentences, encoded_labels

def create_pairs_of_scene_and_reldict():

    pickle_files = []
    rel_directory = 'code/rel_dicts.py'
    for filename in os.listdir(rel_directory):
        if filename.endswith('.pkl'):
            pickle_files.append(os.path.join(rel_directory, filename))

    relationship_dicts = {}
    for file in pickle_files:
        with open(file, 'rb') as f:
            file_name = os.path.basename(file)
            title = os.path.splitext(file_name)[0]
            relationship_dicts[title] = pickle.load(f)

    scene_directory = '/Users/josie/Natural Language Processing/nlp/Project/split_texts'
    scene_files = []
    for filename in os.listdir(scene_directory):
        if filename.endswith('.txt'):
            scene_files.append(os.path.join(scene_directory, filename))

    data_pairs = []
    for scene_file in scene_files:
        with open(scene_file, 'r', encoding='utf-8') as f:
            scene_text = f.read()
        scene_title = os.path.basename(scene_file)
        scene_title = os.path.splitext(scene_title)[0]
        scene_title = filename_mapping[scene_title]
        if scene_title in relationship_dicts:
            data_pairs.append((scene_text, relationship_dicts[scene_title]))
    return data_pairs


