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