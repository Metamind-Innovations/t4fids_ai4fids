import logging
import json
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

logger = logging.getLogger(__name__)

def evaluation_metrics(y_true, classes):
    '''Calculate evaluation metrics '''

    accuracy = accuracy_score(y_true, classes)

    cnf_matrix = confusion_matrix(y_true, classes)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # true positive rate - TPR
    TPR = TP/(TP+FN)

    # false positive rate - FPR
    FPR = FP/(FP+TN)

    # F1 Score
    f1 = f1_score(y_true, classes, average='macro')

    return (accuracy, np.mean(TPR), np.mean(FPR), f1)

def load_data(data_path: str) -> Tuple[DataFrame, DataFrame]:
    """Load data and return train and test sets."""
    try:
        train_path = Path(data_path) / "train.csv"
        test_path = Path(data_path) / "test.csv"
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

    return train_data, test_data