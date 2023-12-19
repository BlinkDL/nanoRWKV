import numpy as np
import tensorflow as tf

class TfLiteRunner:
    def __init__(self, path: str = None, content: bytes = None):
        assert (path is None) != (content is None)
        if path is not None:
            interpreter = tf.lite.Interpreter(model_path=path)
        else:
            interpreter = tf.lite.Interpreter(model_content=content)

        self.batch_size = 1
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_ids = [inp['index'] for inp in interpreter.get_input_details()]
        self.output_ids = [out['index'] for out in interpreter.get_output_details()]

    def try_change_batch_size(self, inputs):
        batch_size = inputs[0].shape[0]
        if self.batch_size == batch_size:
            return
        self.batch_size = batch_size

        input_details = self.interpreter.get_input_details()
        for tensor in input_details:
            tensor_shape = tensor['shape']
            tensor_shape[0] = batch_size
            self.interpreter.resize_tensor_input(tensor['index'], tensor_shape)
        self.interpreter.allocate_tensors()
        
    def run(self, inputs: list) -> list[np.ndarray]:
        assert len(inputs) == len(self.input_ids)
        self.try_change_batch_size(inputs)

        for i in range(len(inputs)):
            self.interpreter.set_tensor(self.input_ids[i], inputs[i])

        self.interpreter.invoke()
        return [self.interpreter.get_tensor(idx) for idx in self.output_ids]

    def __call__(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        return [tf.constant(out) for out in self.run(inputs)]
