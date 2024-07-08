import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')



# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--task', type=str, default='list') #one of yesno, factoid, list
eval_arg.add_argument('--processed_data_dir', type=str, default='processed_data')
eval_arg.add_argument('--max_words_per_line', type=int, default='150')
eval_arg.add_argument('--training_data', type=str, default='BioASQ-training12b/training11b.json')
''' 
Possible model parameters for each task 
if yesno: 
if  factoid: biomistral
if list: biobert, biogpt
'''
eval_arg.add_argument('--model', type=str, default='biobert')
eval_arg.add_argument('--similarity_metric', type=str, default='ROUGE') #one of ROUGE
eval_arg.add_argument('--start_indx', type=int, default='201') #preprocessing data processing starting index
eval_arg.add_argument('--embeddings_dimension', type=int, default='768')
eval_arg.add_argument('--append_to_db', type=str2bool, default=True) #start a new db, or append to existing
eval_arg.add_argument('--db_save_freq', type=int, default=10) #save db every x process



def get_config_eval():
    args = parser.parse_args()
    return args