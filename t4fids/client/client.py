import logging
from pathlib import Path
from ..common.model import get_model
from .utils import evaluation_metrics
import flwr as fl
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import joblib
from .utils import load_data
from .constants import TrainingArtifacts
import json

# Configure logging before other imports
logger = logging.getLogger(__name__)

class Client(fl.client.NumPyClient):
    '''FL Client core class'''
    def __init__(self, 
                 train_data,
                 test_data,
                 features_to_drop:str,
                 label_keyword: str,
                 learning_rate: float,
                 resample_flag: bool
        ):

        self.train_data = train_data
        self.test_data = test_data
        self.features_to_drop = features_to_drop
        self.label_keyword = label_keyword
        self.lr = learning_rate
        self.resample_flag = resample_flag

        self._preprocess_data()
        self._initialize()
        self.eval_metrics_lst = []

    def _initialize(self):
        """Initialize client"""
        try:
            self._initialize_model()
            logger.info("Client initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _preprocess_data(self):
        """Handle all data preprocessing steps"""

        self.train_data.dropna(inplace=True)
        self.test_data.dropna(inplace=True)
        
        # Identify label column
        label_col = self.label_keyword
        
        # Drop features and label
        self.X_train = self.train_data.drop(
            self.features_to_drop + [label_col], axis=1
        )
        self.y_train = self.train_data[label_col]
        
        self.X_test = self.test_data.drop(
            self.features_to_drop + [label_col], axis=1
        )
        self.y_test = self.test_data[label_col]
        
        # To numpy
        self.X_train = self.X_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        
        # Encode labels
        self._encode_labels()

        # Scale features
        self._scale_features()

        # Resample for class imbalance
        if self.resample_flag:
            self._resample()

    def _encode_labels(self):
        """Encode labels"""
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

    def _scale_features(self):
        """Standardize feature scaling"""
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _resample(self):
        """Apply SMOTE"""
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def _initialize_model(self):
        """Initialize and compile the model, and training artifacts"""
        input_shape = self.X_train.shape[1]
        num_classes = len(self.label_encoder.classes_)
        
        self.model = get_model(input_shape, num_classes)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.artifacts = TrainingArtifacts(
            model=self.model,
            label_encoder=self.label_encoder,
            feature_scaler=self.scaler,
        )

    # Flower-specific
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        try:
            current_round = config['current_round']
            self.epochs = config['local_epochs']
            self.model.set_weights(parameters)

            # training steps
            logger.info(f'Client performs local training at round {current_round}:')
            self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=self.epochs, 
                batch_size=config['batch_size']
            )
            return self.model.get_weights(), len(self.y_train), {}
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

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
            
            return loss, len(self.X_test), {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def save_artifacts(self, 
                       model_save_path: str, 
                       suffix: str = None
        ) -> None:
        """Save all training artifacts"""
        if not self.artifacts:
            raise ValueError("No artifacts to save")

        if suffix==None:
            suffix = 'v1'

        save_dir = Path(model_save_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Model
        model_path = save_dir / f"model_{timestamp}_{suffix}.h5"
        self.artifacts.model.save(model_path)
        
        # Scaler
        scaler_path = save_dir / f"scaler_{timestamp}_{suffix}.joblib"
        joblib.dump(self.artifacts.feature_scaler, scaler_path)

        # Label encoder
        encoder_path =  save_dir / f"label_encoder_{timestamp}_{suffix}.joblib"
        joblib.dump(self.artifacts.label_encoder, encoder_path)

        logger.info(f"Saved model, scaler, and encoder to {save_dir}")

    def save_results(self,
                     results_save_path: str
                    ):
        
        METRIC_NAMES = ["ACC", "TPR", "FPR", "F1_score"]
        data = {
                f"round_{i+1}": dict(zip(METRIC_NAMES, metric_tuple))
                for i, metric_tuple in enumerate(self.eval_metrics_lst)
        }

        # Save to JSON file
        save_dir = Path(results_save_path)
        save_path = save_dir / "metrics.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"Results saved to {save_dir}.")

def gen_client(data_path,
               features_to_drop,
               label_keyword,
               lr=0.002,
               resample_flag=False
               ) -> Client:
    ''' Generate and return client instance'''
    try:

        # Load data
        train_data, test_data = load_data(data_path)

        client = Client(train_data,
                        test_data,
                        features_to_drop,
                        label_keyword,
                        lr,
                        resample_flag
               )
        logger.info('Client instance has been generated.')
    except Exception as e:
        logger.error(f"Failed to generate client instance: {e}")
        raise

    return client