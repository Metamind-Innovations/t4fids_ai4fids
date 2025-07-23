import argparse
import flwr as fl
import tensorflow as tf
from t4fids.client.client import gen_client
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def main():
    
    # Parser: takes only config file path
    parser = argparse.ArgumentParser(description='Federated learning client')
    parser.add_argument('--config', 
                        default='conf/client/config.json',
                        type=str, 
                        help='Path to the configuration file of the client'
                        )
    parser.add_argument('--client_type', 
                    default='normal',
                    type=str, 
                    help='Path to the configuration file of the client'
                    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger.info(f"Client configuration is: \n {config}")

    # (Optional): force TensorFlow to use the CPU only
    tf.config.set_visible_devices([], 'GPU')
    
    # Generate client instance
    client = gen_client(config['data_path'],
                        config['features_to_drop'],
                        config['label_keyword'],
                        config['learning_rate'],
                        config['resample_flag'],
                        args.client_type
    )
    
    # Start client
    server_address = f"{config['server_address']}:{config['server_port']}"
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

    # Save artifacts
    client.save_artifacts(config['model_save_path'])
    client.save_results(config['results_save_path'])

if __name__ == "__main__":
    main()