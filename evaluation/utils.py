def check_input(config):
    model_name = config.model_name 
    task = config.task 

    if task == 'yesno':
        if config.model_name in []:
            print(f"{model_name} is not a valid model.")
        else:
            raise ValueError(f"Error: {model_name} is not a valid model.")
    
    elif task == 'factoid':
        if model_name in []:
            print(f"{model_name} is not a valid model.")
        else:
            raise ValueError(f"Error: {model_name} is not a valid model.")
    
    else: # list
        if model_name in ['biobert','bluebert', 'roberta']:
            print(f"{model_name} is not a valid model.")
        else:
            raise ValueError(f"Error: {model_name} is not a valid model.")