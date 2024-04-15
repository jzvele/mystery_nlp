import torch
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from read_file import read_text_file
import pickle
import os
from model import RelationshipModel
from label_utils import relationship_categories, filename_mapping, bidirectional_pairs
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score


# ----------------------------- Create pairs of scene text and relationship dictionary -----------------------------
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

data_pairs = create_pairs_of_scene_and_reldict()

# ----------------------------- Split into train/test sets -----------------------------

train_val_pairs, test_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
label_to_int = {label: idx for idx, label in enumerate(relationship_categories)} # Relationship_categories is a list of possible labels for a relationship btw two characters
print(f'Total pairs: {len(data_pairs)}, Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}')

# ----------------------------- Encode the data -----------------------------
# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
batch_size = 24


def flatten_and_prepare_relationships(relationships, label_to_int, bidirectional_relationships):
    sentences = []
    encoded_labels = []
    
    # to create a standard relationship sentence
    def make_sentence(person1, person2, relationship):
        return f"{person1} is {relationship} to {person2}"

    for person_a, links in relationships.items():
        for person_b, relation_a_to_b in links.items():
            # check for a bidirectional relationship
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

            # single direction relationship
            sentence = make_sentence(person_a, person_b, relation_a_to_b)
            encoded_label = label_to_int.get(relation_a_to_b, -1)
            if encoded_label == -1:
                encoded_label = label_to_int.get('sibling', -1)
            sentences.append(sentence)
            encoded_labels.append(encoded_label)
    
    return sentences, encoded_labels

def preprocess_and_tokenize(pairs):
    inputs = []
    labels = []
    for scene_text, relationships in pairs:
        # Tokenize scene text
        context_inputs = tokenizer(scene_text, return_tensors="pt", truncation=True, max_length=512)
        context_outputs = bert_model(**context_inputs)
        
        # Prepare relationship sentences and labels
        relationship_sentences, encoded_labels = flatten_and_prepare_relationships(relationships, label_to_int, bidirectional_pairs)
        relation_inputs = tokenizer(relationship_sentences, padding=True, truncation=True, return_tensors="pt")
        relation_outputs = bert_model(**relation_inputs)
        
        # Repeat the context_outputs.pooler_output to match the number of relationship sentences
        context_repeated = context_outputs.pooler_output.detach().repeat(relation_outputs.pooler_output.size(0), 1)

        # Concatenate along dimension 1 (feature dimension)
        combined_input = torch.cat((context_repeated, relation_outputs.pooler_output), dim=1)

        inputs.append(combined_input)
        labels.append(torch.tensor(encoded_labels, dtype=torch.long))
    
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)  # Concatenate all inputs and labels along the batch dimension

def validate(model, val_inputs, val_labels, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(val_inputs), batch_size):  # Manually batching
            inputs = val_inputs[i:i+batch_size]
            labels = val_labels[i:i+batch_size]

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)
    return avg_loss, accuracy, precision, recall, f1

train_inputs, train_labels = preprocess_and_tokenize(train_pairs)
val_inputs, val_labels = preprocess_and_tokenize(val_pairs)
test_inputs, test_labels = preprocess_and_tokenize(test_pairs)

# Model instantiation
relationship_model = RelationshipModel(num_labels=len(relationship_categories))
optimizer = optim.Adam(relationship_model.parameters(), lr=0.0003)
loss_fn = nn.CrossEntropyLoss()

# Initialize variables for loss smoothing
smooth_loss = 0.0
smooth_factor = 0.1  # This factor determines the weight of new losses
# Variables for early stopping
best_val_loss = float('inf')
patience = 5
trials = 0
# Learning rate scheduler
optimizer = optim.Adam(relationship_model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training Loop
for epoch in range(20):
    print(f"Starting Epoch {epoch+1}")
    for i in range(0, len(train_inputs), batch_size):
        inputs = train_inputs[i:i + batch_size].detach()
        labels = train_labels[i:i + batch_size].detach()
        
        optimizer.zero_grad()
        outputs = relationship_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update smoothed loss
        # smooth_loss = (1 - smooth_factor) * smooth_loss + smooth_factor * loss.item()

    # Validation step at the end of each epoch
    val_loss, accuracy, precision, recall, f1 = validate(relationship_model, val_inputs, val_labels, loss_fn)
    print(f"End of Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy:.2f}, Validation Precision: {precision:.2f}, Validation Recall: {recall:.2f}, Validation F1: {f1:.2f}")

    # scheduler.step(val_loss)  # Adjust the learning rate

    # Implementing early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trials = 0  # reset the patience counter
        torch.save(relationship_model.state_dict(), 'best_model.pth')  # save the best model
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping triggered")
            break

# # Training loop
# num_epochs = 3
# for epoch in range(num_epochs):
#     relationship_model.train()
#     optimizer.zero_grad()
#     outputs = relationship_model(train_inputs)
#     loss = loss_fn(outputs, train_labels)
#     loss.backward()
#     optimizer.step()
    
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Saving the model
torch.save(relationship_model.state_dict(), 'relationship_model.pth')


relationship_model.load_state_dict(torch.load('relationship_model.pth'))

def evaluate_model(model, test_inputs, test_labels, label_to_int):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients for evaluation
        outputs = model(test_inputs)
        _, predicted_indices = torch.max(outputs, dim=1)  # Get the indices of the max log-probability

    # Convert indices to labels for predictions
    int_to_label = {v: k for k, v in label_to_int.items()}
    predicted_labels = [int_to_label[idx.item()] for idx in predicted_indices]

    # Convert indices to labels for actual labels
    actual_labels = [int_to_label[idx.item()] for idx in test_labels]

    # Calculate accuracy
    correct = (predicted_indices == test_labels).sum().item()
    accuracy = correct / test_labels.size(0) * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

    # Print scenes with their actual and predicted labels
    for actual, predicted in zip(actual_labels, predicted_labels):
        print(f"Actual Relationship: {actual}\nPredicted Relationship: {predicted}\n")

    return predicted_labels, actual_labels, accuracy


predicted_labels, actual_labels, accuracy = evaluate_model(relationship_model, test_inputs, test_labels, label_to_int)
