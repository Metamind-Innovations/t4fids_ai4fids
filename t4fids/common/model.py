import tensorflow as tf

def get_model(input_shape, num_classes):
    '''Create a model with the given input shape and output/num_classes
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),  
    ])
    return model
