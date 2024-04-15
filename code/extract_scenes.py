from openai import OpenAI
from openaikey import api_key
import os

client = OpenAI(api_key=api_key)

preamble = '''
        "The following text, delimited by triple backticks, contains the text of the final fifth of a Poirot novel. Please extract the classic scene in which Poriot reveals the culprit to the suspects and explains the case. Please return this scene to me with no additional commentary."
        '''

def extract_scene(text, preamble):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 'You are a helpful assistant and an expert on Agatha Christie novels.'},
            {"role": "user", "content": preamble + f'```{text}```'},
        ]
    )
    result = response.choices[0].message.content
    return result


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
if __name__ == '__main__':

    split_texts_dir = "Project/code/split_texts"
    text_files = os.listdir(split_texts_dir)

    text_dict = {}
    for file in text_files:
        file_path = os.path.join(split_texts_dir, file)
        text = read_text_file(file_path)
        text_dict[file] = text

    for title, text in text_dict.items():
        scene = extract_scene(text, preamble)
        with open(f"Project/code/split_texts/{title}.txt", "wb") as f:
            f.write(scene.encode('utf-8'))