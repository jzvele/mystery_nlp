import re
import os

'''
Assumes Project Gutenberg header and footer already removed. Chapter titles are not already removed.
'''

def clean_text(text):
    
    # Replace multiple newline characters with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Remove or replace special characters
    text = re.sub(r'[^a-zA-Z0-9 ,.!?;:\'\n]', '', text)
    
    return text

def split_text_into_sections(input_filepath, output_dir, words_per_section=1500):
   
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and clean the novel's text
    with open(input_filepath, 'r', encoding='utf-8') as file:
        text = clean_text(file.read())
    
    # Split the text into words
    words = text.split()
    
    # Calculate the number of sections
    total_sections = len(words) // words_per_section + (1 if len(words) % words_per_section > 0 else 0)
    
    for section_number in range(total_sections):
        # Extract words for the current section
        section_words = words[section_number*words_per_section : (section_number+1)*words_per_section]
        section_text = ' '.join(section_words)
        
        # Save the section to a file
        section_filename = os.path.join(output_dir, f'section_{section_number+1}.txt')
        with open(section_filename, 'w', encoding='utf-8') as section_file:
            section_file.write(section_text)
        
        print(f'Saved section {section_number+1} to {section_filename}')


def clean_text_files(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Check if the file is a text file
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            
            # Read the content of the file
            with open(input_filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Clean the text
            cleaned_content = clean_text(content)
            
            # Save the cleaned text to a new file in the output directory
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            
            print(f'Processed and saved cleaned text to {output_filepath}')
        else:
            print(f'Skipping non-text file: {filename}')

def split_novel_into_parts(input_filepath, output_dir, num_parts=5):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the novel's text
    with open(input_filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Calculate the size of each part
    part_size = len(text) // num_parts
    
    # For each part, save a slice of the novel
    for part in range(num_parts):
        start_index = part * part_size
        # Make sure to capture any remainder in the last part
        end_index = (start_index + part_size) if (part < num_parts - 1) else None
        
        part_text = text[start_index:end_index]
        
        # Save the part to a file
        part_filename = os.path.join(output_dir, f'part_{part + 1}.txt')
        with open(part_filename, 'w', encoding='utf-8') as part_file:
            part_file.write(part_text)
        
        print(f'Saved part {part + 1} to {part_filename}')


if __name__ == '__main__':

    # Process all text files in a directory
    # clean_text_files('raw', 'cleaned_texts')

    for filename in os.listdir('cleaned_texts'):
        if filename.endswith('.txt'):  # Check if the file is a text file
            input_filepath = os.path.join('cleaned_texts', filename)
            output_filepath = os.path.join('split_texts', filename)

            split_novel_into_parts(input_filepath, output_filepath)