import os
import torch
import numpy as np
import tensorflow as tf
from model import GPTConfig, GPT

from tflite_convertation.convertation import convert_model
from tflite_convertation.test_rwkv_modules import test_keras_modules, maybe_sampling
from tflite_convertation.tflite_runner import TfLiteRunner

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
device = 'cpu'
logits_lambda = [None, tf.math.softmax, tf.math.log_softmax][0] # can modify tflite model head
use_dynamic_quantization = False
exec(open('configurator.py').read()) # overrides from command line or config file'
# -----------------------------------------------------------------------------

# model
assert init_from == 'resume'
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()


with torch.no_grad():
    keras_model = test_keras_modules(model)
    tflite_model_path = ckpt_path.replace('.pt', '.tflite')
    tflite_model = convert_model(keras_model, tflite_model_path, logits_lambda, use_dynamic_quantization)
    runner = TfLiteRunner(content=tflite_model)

    # test may be used only with raw logits
    if logits_lambda is None:
        for batch_size in [1, 32, 64]:
            positions, flatten_states = keras_model.get_states(batch_size)
            vocab_size, max_len = keras_model.config.vocab_size, keras_model.config.block_size
            testing_tokens = np.random.randint(0, vocab_size, size=(batch_size, max_len), dtype=np.int32)

            logits_gt, _ = model(torch.as_tensor(testing_tokens), warn=False)
            for i in range(max_len):
                logits, *flatten_states = runner([testing_tokens[:, i], positions, *flatten_states])
                assert np.allclose(logits_gt[:, i].numpy(), logits, rtol=5e-2, atol=1e-5)
                positions += 1

        maybe_sampling(checkpoint, keras_model, runner)
