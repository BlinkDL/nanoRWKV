import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from tflite_convertation.keras_model import Keras_RWKV


class TfliteModel(tf.keras.Model):
    def __init__(self, keras_model, states, logits_lambda=None):
        inputs = [
            Input(shape=tuple(), name='tokens', dtype=np.int32),
            Input(shape=tuple(), name='position', dtype=np.int32),
            *[
                Input(shape=state.shape[1:], name=f'state_{i}')
                for i, state in enumerate(states)
            ],
        ]
        outputs = list(keras_model(inputs))
        if logits_lambda is not None:
            outputs[0] = logits_lambda(outputs[0])
        super().__init__(inputs=inputs, outputs=outputs)


def convert_model(keras_model: Keras_RWKV, path_to_write=None,
                  logits_lambda=None, use_dynamic_quantization=False):
    tokens = np.zeros(1, dtype=np.int32)
    positions, keras_states = keras_model.get_states(batch_size=1)
    
    tfliteModel = TfliteModel(keras_model, keras_states, logits_lambda)
    tfliteModel([tokens, positions, *keras_states])

    converter = tf.lite.TFLiteConverter.from_keras_model(tfliteModel)

    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if use_dynamic_quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converted_model = converter.convert()

    if path_to_write is not None:
        with open(path_to_write, 'wb') as file:
            file.write(converted_model)
    return converted_model
