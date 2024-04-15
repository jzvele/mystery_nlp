from openai import OpenAI
import json
import pickle
from openaikey import api_key

client = OpenAI(api_key=api_key)

preamble = '''
        "Identify the relationships between characters in the following text, which is delimited by triple backticks. Please return this information to me in the form of a JSON string that represents a dictionary with the following structure: {'character1': {'character2': 'relationship'}, 'character2': {'character1': 'relationship'}}. Please ensure that all values in the outer dictionary are of type dictionary. Examples of potential relationships might be "colleague," "daughter," "suspect," "killer," "childhood friend," etc. In the 'character1': {'character2': 'relationship'}} structure, the "relationship" should describe character2 from the perspective of character1. Thank you!"
        '''
def dictionary_depth(d, current_depth=1):
    """ Recursively find the depth of a nested dictionary. """
    if not isinstance(d, dict) or not d:
        return current_depth
    return max(dictionary_depth(v, current_depth + 1) for k, v in d.items())

def extract_relationships(text_to_analyze, preamble):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a helpful assistant designed to output JSON. You are also an expert on mystery novels.'},
            {"role": "user", "content": preamble + f'```{text_to_analyze}```'},
        ]
    )
    result = response.choices[0].message.content
    try:
        # response text should be a JSON string representing a dictionary
        result_dict = json.loads(result)
    except json.JSONDecodeError:
        print("Failed to decode the response as JSON.")
        return None

    if dictionary_depth(result_dict) > 4:
        key_list = list(result_dict.keys())
        return result_dict[key_list[0]]
        
    else: 
        return result_dict


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
if __name__ == '__main__':

    with open("code/generated_scenes.pkl", "rb") as f:
        scenes = pickle.load(f)
    for i, scene in enumerate(scenes):
        relationships = extract_relationships(scene, preamble)
        for key, value in relationships.items():
            print(f"{key}: {value}\n")
            for person in relationships.keys():
                if person in value.keys():
                    print(f"{person}: {value[person]}\n\n")
        filename = f"scene_{i+1}_relationships"
        with open(f"code/{filename}.pkl", "wb") as f:
            pickle.dump(relationships, f)