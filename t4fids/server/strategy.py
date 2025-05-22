import flwr as fl
from flwr.common import Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def custom_strategy(BaseStrategy):
    '''This function returns the custom strategy class, with the BaseStrategy as its parent class.
    '''
    class AggregateCustomMetricStrategy(BaseStrategy):
        ''' Standard FedAvg, but server also evaluates aggregated accuracy of clients
        '''
        def __init__(self,*args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_acc_lst = []
            
        def aggregate_evaluate(
            self,
            server_round: int,
            results,
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            '''Aggregate evaluation accuracy using weighted average.
            '''
            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            self.eval_acc_lst.append(aggregated_accuracy)
                
            logger.info(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}
    return AggregateCustomMetricStrategy