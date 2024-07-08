import os
import sys

import faiss
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))       
from config.eval_config import get_config_eval
from models.model import get_model


class FaissIndexer:
    def __init__(self, index_file="faiss_index.bin"):

        cfg = get_config_eval()
        
        
        # Create the FAISS index
        embedding_dim  = cfg.embeddings_dimension  
        append_to_db  = cfg.append_to_db  
        # Get the pre-trained model and tokenizer
        self.model , self.tokenizer  = get_model()
        
        # Initialize or load the FAISS index
        self.index_file = index_file
        if append_to_db and os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            self.current_index = self.index.ntotal
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance measure
            self.current_index = 0  # To keep track of the current index
    
    def tokenize_and_embed(self, sentence):
        """Tokenize a sentence and generate its embedding."""
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
        return hidden_states

    def add_to_index(self, sentences):
        """Add the combined embedding of two sentences to the FAISS index."""      
        combined_embedding = None

        for sentence in sentences:
            hidden_states = self.tokenize_and_embed(sentence)

            mean_hidden_states = torch.mean(hidden_states, dim=1)  # Shape [1, 768]

            if combined_embedding is None:  # Handle the first sentence case
                combined_embedding = mean_hidden_states
            else:
                # Compute the mean of the combined embeddings
                combined_embedding = torch.mean(torch.stack([combined_embedding, mean_hidden_states]), dim=0)
                # Convert embedding to numpy array
        if combined_embedding is None:
            combined_embedding =  torch.zeros(1, 768)  # Return a dummy embedding for None sentences
        combined_embedding_np = combined_embedding.cpu().numpy()
        
        # Add embedding to the FAISS index
        self.index.add(combined_embedding_np)
            
    def delete(self):
        """Delete FAISS index file."""
        # Initialize a new FAISS index
        if os.path.exists(self.index_file):
            os.remove(self.index_file)

    def search(self, query_sentence, k=1):
        """Search for the k most similar embeddings to the combined query sentences, return corresponding document content"""
        query_hidden_states = self.tokenize_and_embed(query_sentence)
        
        # Compute the mean of the combined hidden state
        query_embedding = torch.mean(query_hidden_states, dim=1)
        
        # Convert query embedding to numpy array
        query_embedding_np = query_embedding.cpu().numpy()
        
        # Perform the search
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Retrieve the corresponding documents
        #results = [(str(idx)+'.txt', distances[0][i]) for i, idx in enumerate(indices[0])]
        print(f"most relevant text is {indices[0][0]}.txt")
        return get_processed_doc_content( f"1.txt")
        
    def display_index_contents(self,limit):
        """Display the contents of the FAISS index and corresponding document names."""
        print(f"{'Index':<10}{'Document':<20}{'Embedding (first 5 elements)'}{'Document content':<20}")
        print("="*60)
        for i in range(limit):
            embedding = self.index.reconstruct(i)
            document = f"{i}.txt"
            doc_content= get_processed_doc_content(f"{i}.txt")
            print(f"{i:<10}{document:<20}{embedding[:5]}{doc_content:<20}")

    def save_to_disk(self, index_file="faiss_index.bin"):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, index_file)

    def load_from_disk(self, index_file="faiss_index.bin"):
        """Load the FAISS index from disk."""
        self.index = faiss.read_index(index_file)
        self.current_index = self.index.ntotal


# # Usage example

# # Initialize the indexer
# indexer = FaissIndexer()

# # Add initial sentence pairs to the index
# initial_sentence_pairs = [
#     ("The quick brown fox jumps over the lazy dog.", "I love natural language processing."),
#     ("The weather today is sunny and bright.", "Machine learning is fascinating."),
#     ("Artificial intelligence is the future.", "Deep learning models are powerful."),
# ]
# initial_documents = ["document_1.txt", "document_2.txt", "document_3.txt"]

# for (sentence1, sentence2), document in zip(initial_sentence_pairs, initial_documents):
#     indexer.add_combined_embeddings_to_index(sentence1, sentence2, document)


# # Display the contents of the index
# indexer.display_index_contents()

# # Save the FAISS index to disk
# indexer.save_to_disk()

# # Create a new indexer instance and load data from disk
# new_indexer = FaissIndexer()
# new_indexer.load_from_disk()

# # Display the contents of the new indexer
# new_indexer.display_index_contents()

# # Search for similar sentences
# query_sentence = "I enjoy working with NLP."
# results = new_indexer.search(query_sentence, k=3)

def get_processed_doc_content(filename):
    cfg = get_config_eval()
    file =  open(cfg.processed_data_dir+'/'+filename,'r') 
    contents = file.read()
    return contents