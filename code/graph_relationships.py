import networkx as nx
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages


def plot_character_relationships(relationships):
    """
    Creates and displays a directed graph for character relationships where all edges point
    from the character specified in the outer dictionary to the characters in the inner dictionary.

    Parameters:
    - relationships (dict): A dictionary containing characters and their relationships.

    Example of input:
    {
        'Character A': {
            'Character B': 'relationship from A to B',
            'Character C': 'relationship from A to C'
        },
        'Character B': {
            'Character A': 'relationship from B to A'
        }
    }
    """
    # Create a directed graph from the relationships
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for person, links in relationships.items():
        G.add_node(person)  # Add each character as a node
        for related_person, relation in links.items():
            G.add_edge(person, related_person, label=relation)  # Maintain original direction

    # Using Kamada-Kawai layout for better visual spacing
    pos = nx.kamada_kawai_layout(G, scale=8)

    # Draw the graph
    plt.figure(figsize=(16, 10))
    nx.draw(G, pos, node_size=3000, node_color='lightblue', with_labels=True, arrows=True, font_size=7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=6)

    plt.title('Character Relationships')
    plt.show()


with open("code/motoe_relationships.pkl", "rb") as f:
    relationships = pickle.load(f)

plot_character_relationships(relationships)