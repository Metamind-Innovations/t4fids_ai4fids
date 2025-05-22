import flwr as fl
import argparse
import json
from t4fids.common.model import get_model
from t4fids.server.strategy import custom_strategy
from t4fids.server.utils import get_on_evaluate_config_fn, get_on_fit_config_fn
from t4fids.server.constants import STRATEGY_MAP
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    '''Parser
    '''
    parser = argparse.ArgumentParser(description='FL server')
    parser.add_argument('--config', type=str, default='conf/server/config.json', help='Default path to the config file of server')
    parser.add_argument('--server_address', type=str, help='Address of server')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds')
    parser.add_argument('--num_clients', type=int, help='Minimum number of clients')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--local_epochs', type=int, help='Number of local epochs')
    parser.add_argument('--input_shape', type=int, help='Model input shape')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--strategy', type=str, help='Aggregation strategy')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def update_config(args, conf):
    """Update config file via CLI."""
    KEYS = [
        "server_address", "num_rounds", "num_clients",
        "batch_size", "local_epochs", "input_shape",
        "num_classes", "strategy"
    ]
    for key in KEYS:
        value = getattr(args, key, None)
        if value is not None:
            conf[key] = value
    return conf
    

def main():
    
    # parser
    args = parse_arguments()
    
    # load config and update based on CLI arguments (optional)
    conf = load_config(args.config)
    conf = update_config(args, conf)
    
    # Compile model
    model = get_model(conf['input_shape'], conf['num_classes'])
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Get model weights and serialize
    weights = model.get_weights()
    parameters = fl.common.ndarrays_to_parameters(weights)

    BaseStrategy = STRATEGY_MAP.get(conf["strategy"])
    AggregateCustomMetricStrategy = custom_strategy(BaseStrategy)

    # Server's strategy
    strategy = AggregateCustomMetricStrategy(
        min_available_clients=conf["num_clients"],
        on_fit_config_fn=get_on_fit_config_fn(conf["batch_size"], conf['local_epochs']),
        on_evaluate_config_fn=get_on_evaluate_config_fn(conf['num_rounds']),
        initial_parameters=parameters,
    )
    
    # Start server
    server_address = f"{conf['server_address']}:{conf['server_port']}"
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=conf['num_rounds']),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()