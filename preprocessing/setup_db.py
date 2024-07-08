import os
import sys

import faiss
import numpy as np
import torch
from process_absracts import process_abstract
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))       
from config.eval_config import get_config_eval
from models.model import get_model


def add_sentence_to_index(sentences):
    combined_embedding = None
    prev_hidden_state = None

    for sentence in sentences:
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]
        sentence_embedding = torch.mean(hidden_states, dim=1)

        # Compute the mean of the hidden states from both sentences
        if combined_embedding == None: #handle first senentce case
            mean_hidden_states = torch.mean(torch.stack(hidden_states), dim=0)
            combined_embedding = mean_hidden_states
            prev_hidden_state = mean_hidden_states
        else:
            mean_hidden_states = torch.mean(torch.stack([prev_hidden_state, hidden_states]), dim=0)
            combined_embedding = torch.mean(mean_hidden_states, dim=1)
    # Convert embedding to numpy array
    combined_embedding_np = combined_embedding.cpu().numpy()
    
    # Add embedding to the FAISS index
    index.add(combined_embedding_np)





if __name__ == '__main__':
    cfg = get_config_eval()
    process_abstract(cfg)

    model, tokenizer  = get_model()
    # Create the FAISS index
    d = cfg.embeddings_dimension  
    index = faiss.IndexFlatL2(d)  # L2 distance measure





