import argparse

def get_on_fit_config_fn(batch_size, local_epochs):
    ''' Returns a functions which returns the server config file,
    to be used by the client on fit() function
    '''
    def fit_config(server_round):
        
        config = {
            "batch_size": batch_size,
            "current_round": server_round,
            "local_epochs": local_epochs,
        }
        
        return config
    
    return fit_config

def get_on_evaluate_config_fn(num_rounds):
    ''' Returns a functions which returns the server config file,
    to be used by the client on evalaute() function
    '''
    def evaluate_config(server_round):
        
        config = {
            "current_round": server_round,
            "total_rounds": num_rounds
        }
        
        return config
    
    return evaluate_config