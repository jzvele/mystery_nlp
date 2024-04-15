import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pickle_files = []
directory = '/Users/josie/Natural Language Processing/nlp/Project/code'

for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        pickle_files.append(os.path.join(directory, filename))

data = {}
for file in pickle_files:
    with open(file, 'rb') as f:
        file_name = os.path.basename(file)
        title = os.path.splitext(file_name)[0]
        data[title] = pickle.load(f)

def normalize_relationships(relationships, resolution_function):
    """
    Normalize relationships in a dictionary where bidirectional relationships might differ.

    Parameters:
    - relationships (dict): Dictionary of relationships.
    - resolution_function (func): Function that decides which relationship to retain when discrepancies exist.

    Returns:
    - dict: Updated dictionary with normalized relationships.
    """
    # Copy the dictionary to avoid mutating the input directly
    normalized = dict(relationships)

    # Check each pair of relationships
    for personA, linksA in relationships.items():
        for personB, relationAB in linksA.items():
            relationBA = relationships.get(personB, {}).get(personA)
            if relationBA and relationAB != relationBA:
                # Discrepancy found, apply the resolution function
                chosen_relation = resolution_function(personA, personB, relationAB, relationBA)
                normalized[personA][personB] = chosen_relation
                normalized[personB][personA] = chosen_relation

    return normalized

def choose_dominant_relation(personA, personB, relationAB, relationBA):
    """
    Decide which relationship to keep based on a predefined logic.

    Returns:
    - str: The relationship to retain.
    """
    # Placeholder logic: Prefer the relationship from A to B
    return relationAB
        
def create_relationship_matrix(relationships):
    # Extract all unique characters
    characters = set()
    for key, nested_dict in relationships.items():
        characters.add(key)
        characters.update(nested_dict.keys())

    # Create an empty matrix
    characters = sorted(characters)  # sort to have a consistent order
    matrix = pd.DataFrame(index=characters, columns=characters)

    # Fill the matrix with relationships
    for char1, relations in relationships.items():
        for char2, relation in relations.items():
            matrix.loc[char1, char2] = relation

    # Replace NaN with empty strings for better readability
    matrix.fillna("", inplace=True)

    return matrix

def save_matrices_to_pdf(dict_of_dicts, filename='relationship_matrices.pdf'):
    with PdfPages(filename) as pdf:
        for title, relationships in dict_of_dicts.items():
            matrix = create_relationship_matrix(relationships)
            fig, ax = plt.subplots(figsize=(12, 8))  # Adjust size as needed
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=matrix.values, colLabels=matrix.columns, rowLabels=matrix.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)  # Adjust font size as needed
            table.scale(1.2, 1.2)  # Adjust table scaling as needed
            plt.title(title)  # Use the dictionary key as the plot title
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def save_matrices_to_csv(dict_of_dicts, folder='relationship_matrices'):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    for title, relationships in dict_of_dicts.items():
        matrix = create_relationship_matrix(relationships)
        # Create a valid filename for each matrix
        filename = f"{title.replace(' ', '_').replace('/', '_')}.csv"
        # Save the matrix to a CSV file
        matrix.to_csv(os.path.join(folder, filename))

save_matrices_to_csv(data)
save_matrices_to_pdf(data)

