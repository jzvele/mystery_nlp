from openai import OpenAI
import json
import pickle
from openaikey import api_key

client = OpenAI(api_key=api_key)

def generate_scene():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 'You are a helpful assistant and an expert on classic mystery novels.'},
            {"role": "user", "content": 'Please write an example of the classic scene in a detective story in which the detective assembles the suspects, reveals the culprit, and explains the motive and means behind the crime. Please give particular attention to the relationships between the characters and how those relationships impact the case.'},
        ]
    )
    result = response.choices[0].message.content
    return result

scenes = []
for i in range(12):
    scene = generate_scene()
    scenes.append(scene)
    print(f"Scene {i+1}:\n{scene}\n\n")

with open("code/generated_scenes.pkl", "wb") as f:
    pickle.dump(scenes, f)