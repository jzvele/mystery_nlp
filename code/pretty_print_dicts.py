import json
import pickle
import os

pickle_files = []
directory = '/Users/josie/Natural Language Processing/nlp/Project/code/rel_dicts.py'
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        pickle_files.append(os.path.join(directory, filename))

data = []
for file in pickle_files:
    with open(file, 'rb') as f:
        data.append(pickle.load(f))

with open("code/generated_scenes.pkl", "rb") as f:
    scenes = pickle.load(f)

for relationships in data:
    printable_dict = json.dumps(relationships, indent=4)
    print(printable_dict)


for i, scene in enumerate(scenes, 1):
    with open(f"split_texts/scene_{i}.txt", "w") as f:
        f.write(f"Scene {i}:\n")
        f.write(scene)

# title = "awd_relationships"
# if title in data:
#     print(f"Title: {title}")
#     relationships = data[title]
#     printable_dict = json.dumps(relationships, indent=4)
#     print(printable_dict)
# else:
#     print(f"Dictionary with title '{title}' not found.")