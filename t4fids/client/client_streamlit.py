from .client import Client
import tensorflow as tf
import logging

# Configure logging before other imports
logger = logging.getLogger(__name__)

class StreamlitClient(Client):
    '''Streamlit client class'''

    TMP_SAVE_PATH = 'tmp/streamlit'
    OPTIMIZERS = {
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Adam": tf.keras.optimizers.Adam,
        "Adadelta": tf.keras.optimizers.Adadelta,
        "SGD": tf.keras.optimizers.SGD
    }
        