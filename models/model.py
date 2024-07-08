from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from config.eval_config import get_config_eval


def get_model(): 
    cfg = get_config_eval()
    task = cfg.task
    model = cfg.model
    if task == 'yesno':
        pass
    elif task == 'factoid':
        pass
    elif task == 'list':
        if model == 'biobert':
            list_model_name = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
        elif model == 'bluebert':
            list_model_name = 'rsml/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12-finetuned-squad'
        elif model == 'roberta':
            list_model_name = 'scite/roberta-base-squad2-nq-bioasq'
        model = AutoModelForQuestionAnswering.from_pretrained(list_model_name)
        tokenizer = AutoTokenizer.from_pretrained(list_model_name)

    return model, tokenizer