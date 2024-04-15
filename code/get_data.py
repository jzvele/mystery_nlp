import requests

# Can't install gutenberg library for some reason that has to do with Berkeley DB?

def download_gutenberg_text(url):
    """
    Downloads and saves the text from a Project Gutenberg URL.
    
    Parameters:
    - url (str): The URL of the Project Gutenberg text to download.
    
    Returns:
    - str: The text of the book.
    """
    response = requests.get(url)
    response.raise_for_status()  # This will raise an exception for HTTP errors
    return response.text

# URLs for Agatha Christie books
urls = {
'The Mysterious Affair at Styles' : 'https://www.gutenberg.org/cache/epub/863/pg863.txt',
'Poirot Investigates' : 'https://www.gutenberg.org/cache/epub/61262/pg61262.txt',
'The Murder on the Links' : 'https://www.gutenberg.org/cache/epub/58866/pg58866.txt',
'The mystery of the Blue Train' : 'https://www.gutenberg.org/cache/epub/72824/pg72824.txt',
'The Secret Adversary' : 'https://www.gutenberg.org/cache/epub/1155/pg1155.txt',
'The Big Four' : 'https://www.gutenberg.org/cache/epub/70114/pg70114.txt',
'The Man in the Brown Suit' : 'https://www.gutenberg.org/cache/epub/61168/pg61168.txt',
'The Secret of Chimneys' : 'https://www.gutenberg.org/cache/epub/65238/pg65238.txt',
'The Missing Will' : 'https://www.gutenberg.org/cache/epub/67173/pg67173.txt',
'The Plymouth Express Affair' : 'https://www.gutenberg.org/cache/epub/66446/pg66446.txt',
'The Hunter\'s Lodge Case' : 'https://www.gutenberg.org/cache/epub/67160/pg67160.txt'
}

for title, link in urls.items():
    # Download the text
    text = download_gutenberg_text(link)
    # Save the text to a file
    with open(f"{title}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(text)

print("Download completed.")

