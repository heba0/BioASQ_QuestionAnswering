import json
import os
import re
import sys

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from database.faiss import FaissIndexer
from utils.utils import *

if __name__ == '__main__':
    pass

def split_paragraph_into_sentences(paragraph):
    # Split the paragraph into sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    return sentences

def format_paragraph(paragraph, max_words_per_line=512):
    sentences = split_paragraph_into_sentences(paragraph)
    formatted_lines = []
    current_line = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        if sum(len(cl_sentence.split()) for cl_sentence in current_line) - 1 + sentence_word_count <= max_words_per_line:
            current_line.append(sentence)
        else:
            if current_line:
                formatted_lines.append(" ".join(current_line))
            current_line = [sentence]
    
    # Append the last line if it has content
    if current_line:
        formatted_lines.append(" ".join(current_line))
    
    return formatted_lines




def fetch_abstract_from_url(url):
    '''
    Get the abstract from a url.
    '''
    # Set up retries
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504]  # Status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)

    try:
        response = http.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            abstract = soup.find('div', class_='abstract-content')
            if abstract:
                return abstract.text.strip()
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    
    return None


def process_abstract(cfg):
    '''
    For each question, get the abstact of each url, divide it into lines of sentences of max 512 words (BERT sentence limit)
    Save the set of modified abstracts for each question into a file.
    '''


    # Initialize the indexer
    faiss = FaissIndexer()

    with open(cfg.training_data, 'r') as f:
        data = json.load(f)['questions']

    # Check if preprocessing directory exists
    if not os.path.exists(cfg.processed_data_dir):
        # Create the directory if it does not exist
        os.makedirs(cfg.processed_data_dir)
        print(f"Directory '{cfg.processed_data_dir}' created.")
    else:
        print(f"Directory '{cfg.processed_data_dir}' already exists.")

    q_indx = cfg.start_indx 
    for entry in data[cfg.start_indx :]:
        formatted_lines = []
        d =  open(cfg.processed_data_dir+'/'+str(q_indx)+'.txt','w') 
        current_line = []
        for url in entry['documents']:
            abstract = fetch_abstract_from_url(url)
            if not abstract:
                print("Failed to fetch the abstract.")
                continue
            if not is_string(abstract):
                continue
            abstract = clean_string(abstract) #remove extra spaces and tabs 
            sentences = split_paragraph_into_sentences(abstract)
                                   
            for sentence  in sentences:
                sentence_words = sentence.split()
                sentence_word_count = len(sentence_words)
                
                if sum(len(cl_sentence.split()) for cl_sentence in current_line) - 1 + sentence_word_count <= cfg.max_words_per_line:
                    current_line.append(sentence)
                else:
                    if current_line:
                        d.write( " ".join(current_line) + "\n")
                        formatted_lines.append(" ".join(current_line))
                    current_line = [sentence]                                  
        faiss.add_to_index(formatted_lines)
        d.close() 
        if q_indx % cfg.db_save_freq == 0:
            faiss.save_to_disk()
        q_indx += 1 
