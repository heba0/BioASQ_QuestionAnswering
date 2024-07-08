import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))       
import statistics
from string import punctuation

import evaluate
from bert_score import BERTScorer
from evaluate import evaluator

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
                    
    return most_common_phrase, duplicates_num

def remove_word_from_list(query, words_to_remove):
    querywords = query.split()

    resultwords  = [word for word in querywords if word.lower() not in words_to_remove]
    result = ' '.join(resultwords)
    return result

def get_processed_doc_content(filename):
    cfg = get_config_eval()
    file =  open(cfg.processed_data_dir+'/'+filename,'r') 
    contents = file.read()
    return contents

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

   # faiss = FaissIndexer()

    cfg = get_config_eval()
    model , tokenizer  = get_model()
    rouge = evaluate.load('rouge')

    with open(cfg.training_data, 'r') as f:
        data = json.load(f)['questions']
    entry_num = 0
    for entry in data:
        text = []
        if entry['type'] == 'list':
            i=0
            answers_list = []
            question = entry['body']
            exact_answers =  entry['exact_answer']
            print("Question:", question)
            print("Exact Answer:", exact_answers) 
           # corpus = faiss.search(question).splitlines()
            print(f"file name: {entry_num}.txt")
            corpus= get_processed_doc_content(f"{entry_num}.txt").splitlines()
           # print(f"corpus {corpus}")
            if corpus == None:
                    continue
            for line in corpus:
                i+=1
                #print(f"line length: {len(line.split())}")
                answer = get_k_answers(question,line, tokenizer, model,K=1)
                answer = answer.replace("[CLS]", " ")
                answer = answer.replace("[SEP]", " ")
                answer = answer.strip()
                if len(answer) > 0:
                    #print(f"Answer using url %d : %s {i,answer}")
                    answers_list.append(answer)

            #postprocess answers
            final_answers_list = []
            
            for i in range(10):
               # print(f"answers_list: {answers_list}")
                most_common_phrase, duplicates_num = get_common_phrase(answers_list)
                if duplicates_num > 1:
                    final_answers_list.append(most_common_phrase)
                    for indx in range(len(answers_list)):
                        #print(f"answers_list: {answers_list[indx]}, most_common_phrase {most_common_phrase}")
                        answers_list[indx] = remove_word_from_list(answers_list[indx], most_common_phrase)
                else:
                    break
            print(f"Final Answer: {final_answers_list}")

            final_F1_list, final_R_list, final_P_list, final_rouge_list, final_rouge1_list = [],[],[],[],[]
            for exact_answer in exact_answers:
                for answer  in final_answers_list:
                    F1_list, R_list, P_list, rouge_list, rouge1_list = [],[],[],[],[]
                    scorer = BERTScorer(model_type='bert-base-uncased') #scorer.score([candidate], [reference])
                    P, R, F1 = scorer.score([answer], [exact_answer]) # rouge.compute(predictions=predictions,references=references)
                    
                 #   print(f"BERTScore : Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

                    rouge_results = rouge.compute(predictions=[answer],references=exact_answer)

                    F1_list.append(F1.mean())
                    R_list.append(R.mean())
                    P_list.append(P.mean())
                    rouge_list.append(rouge_results)
                    rouge1_list.append(rouge_results['rouge1'])
                index_max = np.argmax(rouge1_list.copy())

                final_F1_list.append(F1_list[index_max])
                final_R_list.append(R_list[index_max])
                final_P_list.append(P_list[index_max])
                final_rouge_list.append(rouge_list[index_max])
                final_rouge1_list.append(rouge1_list[index_max])
                print(f"BERTScore: Precision: {P_list[index_max]:.4f}, Recall: {R_list[index_max]:.4f}, F1: {F1_list[index_max]:.4f}")
                print(f"ROUGE 1 Score: {rouge1_list[index_max]:.4f}")
           
           
            print(f"Final Evaluation Results: ")
            print('Average F1 score: ', torch.stack(final_F1_list).mean(dim=0).item())
            print('Average Recall score: ', torch.stack(final_R_list).mean(dim=0).item() )
            print('Average Precision score: ', torch.stack(final_P_list).mean(dim=0).item() )           
            print('Average ROUGE 1 score: ', statistics.mean(final_rouge1_list)   )
        entry_num += 1    