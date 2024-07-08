import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))       
from string import punctuation

from config.eval_config import get_config_eval
from database.faiss import FaissIndexer
from models.model import get_model


def get_common_phrase(input_list):
    most_common_phrase = ''
    duplicates_num = 0

    f = lambda x: x.translate(str.maketrans('','',punctuation)).lower() # removes punctuation
    phrases = f(' 000 '.join(input_list)) # adds dividers

    for i in input_list:
        phrase = f(i).split()
        for j in range(len(phrase)-1):
            for y in range(j+2,len(phrase)+1):
                phrase_comb = ' '.join(phrase[j:y])
                if (n:=phrases.count(phrase_comb)) > duplicates_num:
                    duplicates_num = n
                    most_common_phrase = phrase_comb
                    
    return most_common_phrase

def remove_word_from_list(words, words_to_remove):
    new_words = [word for word in words if word not in words_to_remove]
    return new_words




def get_k_answers(question,text,tokenizer, model, K=1):
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    top_k_answer_start = torch.topk(answer_start_scores, K).indices
    answer_end_scores = outputs.end_logits
    top_k_answer_end = torch.topk(answer_end_scores, K).indices + 1
    final_answer = ''
    for i in range(K):
      # Get the most likely end of answer with the argmax of the score
      answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[top_k_answer_start.data[0][i]:top_k_answer_end.data[0][i]]))
      final_answer += answer
    return final_answer


if __name__ == '__main__':

    faiss = FaissIndexer()

    cfg = get_config_eval()
    model , tokenizer  = get_model()

    with open(cfg.training_data, 'r') as f:
        data = json.load(f)['questions']

    for entry in data:
        text = []
        if entry['type'] == 'list':
            i=0
            question = entry['body']
            print("Question:", question)
            corpus = faiss.search(question).splitlines()
            if corpus == None:
                    continue
            # for url in entry['documents']:
            

                #text.append(fetch_abstract_from_url(url))
               # abstract = fetch_abstract_from_url(url)
                
                
                #print(f"number of lines : {len(corpus)}")
            for line in corpus:
                i+=1
                #print(f"line length: {len(line.split())}")
                answer = get_k_answers(question,line, tokenizer, model,K=1)
                answer = answer.replace("[CLS]", " ")
                answer = answer.replace("[SEP]", " ")
                answer = answer.strip()
            #if len(answer) > 0:
                print(f"Answer using url %d : %s {i,answer}")