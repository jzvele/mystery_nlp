import os
import pickle
import json

pickle_files = []
rel_directory = 'code/rel_dicts.py'
for filename in os.listdir(rel_directory):
    if filename.startswith('scene'):
        pickle_files.append(os.path.join(rel_directory, filename))

relationship_dicts = {}
for file in pickle_files:
    with open(file, 'rb') as f:
        file_name = os.path.basename(file)
        title = os.path.splitext(file_name)[0]
        relationship_dicts[title] = pickle.load(f)

def determine_correct_choice(person_a, person_b, relation_a_to_b, relation_b_to_a):
    # This function should implement logic to decide which relationship is more appropriate.
    # It could be based on external criteria, frequency of mentions, context analysis, etc.
    # Example placeholder logic (needs real implementation):
    if len(relation_a_to_b) > len(relation_b_to_a):
        return 'BtoA'
    else:
        return 'AtoB'

def find_bidirectional_relationships(relationship_dicts):
    bidirectional_pairs = {}
    
    for title, relationships in relationship_dicts.items():
        checked_pairs = set()  # to avoid re-checking the same pair in reverse
        for person_a, links in relationships.items():
            for person_b, relation_a_to_b in links.items():
                if (person_b, person_a) in checked_pairs:
                    continue  # Skip if this pair was already checked
                
                relation_b_to_a = relationships.get(person_b, {}).get(person_a, None)
                
                # Check if there is a bidirectional relationship
                if relation_b_to_a and relation_a_to_b != relation_b_to_a:
                    # Determine the correct choice based on additional criteria
                    correct_choice = determine_correct_choice(person_a, person_b, relation_a_to_b, relation_b_to_a)

                    if title not in bidirectional_pairs:
                        bidirectional_pairs[title] = []
                    bidirectional_pairs[title].append({
                        'PersonA': person_a,
                        'PersonB': person_b,
                        'RelationshipAtoB': relation_a_to_b,
                        'RelationshipBtoA': relation_b_to_a,
                        'CorrectChoice': correct_choice  # Include directionality in the choice
                    })
                
                checked_pairs.add((person_a, person_b))
    
    return bidirectional_pairs

bidirectional_pairs = find_bidirectional_relationships(relationship_dicts)

# Pretty print the bidirectional pairs
for title, pairs in bidirectional_pairs.items():
    print(f"{title} : ")
    for pair in pairs:
        print(json.dumps(pair, indent=4))
        print()