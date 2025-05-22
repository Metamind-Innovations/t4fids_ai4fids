import flwr as fl

STRATEGY_MAP = {
    "FedAvg": fl.server.strategy.FedAvg,
    "FedYogi": fl.server.strategy.FedYogi,
    "FedAdam": fl.server.strategy.FedAdam,
    "FedAdagrad": fl.server.strategy.FedAdagrad
}