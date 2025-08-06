from dataclasses import dataclass
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Data structures
@dataclass
class TrainingArtifacts:
    model: tf.keras.Model
    label_encoder: LabelEncoder
    feature_scaler: StandardScaler

@dataclass
class TrainingParameters:
    lr: float
    batch_size: int
    local_epochs: int

DEFAULT_DASH_CONF = {
    "paths": {
        "data": "data",
        "model_save": "models",
        "results_save": "results"
    },
    "server": {
        "address": "127.0.0.1",
        "port": "8080"
    },
    "tr_params": {
        "learning_rate": 0.1,
        "batch_size": 64,
        "local_epochs": 5
    },
    "data_kw": {
        "features_to_drop": [
            "flow_id",
            "ip_src",
            "ip_dst",
            "udp_sport",
            "udp_dport"
        ],
        "label_keyword": "label"
    },
    "misc": {
        "override_params": False,
        "resample_flag": False
    },
    "opt": {
        "optimizer": "RMSprop",
        "params": {
            "rho": 0.9,
            "momentum": 0.0
        }
    }
}