import os
import re
import sys
from string import punctuation

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))       
from config.eval_config import get_config_eval

# Dictionary mapping number words to their numeric values
number_words = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15
}

# Function to replace number words with numeric values
def replace_number_words(input_string):
    # Convert input string to lowercase to handle case-insensitivity
    input_string = input_string.lower()
    
    # Use regex to find all words in the string
    words = re.findall(r'\b\w+\b', input_string)
    
    # List to store modified words
    modified_words = []

    for word in words:
        if word in number_words:
            # Replace the word with its numeric value
            modified_words.append(str(number_words[word]))
        else:
            # Keep the word as is
            modified_words.append(word)

    # Join the modified words back into a single string
    modified_string = ' '.join(modified_words)
    
    return modified_string

def is_string(obj):
    return isinstance(obj, str)

def clean_string(input_string):
    # Replace tabs with a single space
    input_string = input_string.replace('\t', ' ').replace('\n', ' ')
    # Replace multiple spaces with a single space
    cleaned_string = re.sub(' +', ' ', input_string)
    # Strip leading and trailing spaces
    cleaned_string = cleaned_string.strip()
    return cleaned_string    


