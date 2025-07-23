from .client import Client
import logging
from .utils import evaluation_metrics
import numpy as np
import json

# Configure logging before other imports
logger = logging.getLogger(__name__)

class StreamlitClient(Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ACC_FILE_PATH = 'tmp/streamlit/acc.json'
        self.METRICS_FILE_PATH = 'tmp/streamlit/metrics.json'
        self.AUX_FILE_PATH = 'tmp/streamlit/aux_config.json'

    def evaluate(self, parameters, config):
        try:
            self.model.set_weights(parameters)
            current_round = config['current_round']
            logger.info(f'Evaluation of the global model at round {current_round}')

            loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
            predicted_test = self.model.predict(self.X_test)
            predicted_classes = np.argmax(predicted_test,axis=1)
            eval_metrics = evaluation_metrics(self.y_test, predicted_classes)
            
            logger.info(
                f"Evaluation metrics are: \nACC: {eval_metrics[0]} \nTPR: {eval_metrics[1]}"
                f"\nFPR: {eval_metrics[2]}, \nF1 score: {eval_metrics[3]} \n"
            )

            # A list of tuples: (accuracy, TPR, FPR, f1). Saved for future use
            self.eval_metrics_lst.append(eval_metrics)

            if config['current_round']==1:
                with open(self.AUX_FILE_PATH, 'w') as f:
                    json.dump(config, f)
                
            with open(self.ACC_FILE_PATH, 'w') as f:
                 json.dump(self.eval_metrics_lst, f)
                 
            if config['current_round']==config['total_rounds']:
                with open(self.AUX_FILE_PATH, 'w') as f:
                    json.dump(eval_metrics, f)
            
            return loss, len(self.X_test), {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise