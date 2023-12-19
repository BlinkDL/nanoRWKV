import torch
import numpy as np

from tflite_convertation.keras_model import KerasLinear
from tflite_convertation.keras_model import KerasGroupNorm
from tflite_convertation.keras_model import KerasLayerNorm
from tflite_convertation.keras_model import KerasLayerState

from tflite_convertation.keras_model import Keras_RWKV_ChannelMix_x051a
from tflite_convertation.keras_model import Keras_RWKV_TimeMix_x051a
from tflite_convertation.keras_model import Keras_RWKV_Block
from tflite_convertation.keras_model import Keras_RWKV

from model import LayerNorm, RWKV_TimeMix_x051a, RWKV_ChannelMix_x051a, Block
from model import GPT, GPTConfig, LayerState


TEST_BATCH_SIZE = 64


def test_linear():
    torch_linear = torch.nn.Linear(128, 256, True)
    keras_linear = KerasLinear(torch_linear=torch_linear)

    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_linear(data)
    torch_out = torch_linear(torch.as_tensor(data))
    assert np.allclose(keras_out, torch_out.numpy(), rtol=1e-3, atol=1e-6)

    data = np.random.randn(TEST_BATCH_SIZE, TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_linear(data)
    torch_out = torch_linear(torch.as_tensor(data))
    assert np.allclose(keras_out, torch_out.numpy(), rtol=1e-3, atol=1e-6)


def test_group_norm():
    torch_norm = torch.nn.GroupNorm(4, 128, eps=1e-1)
    keras_norm = KerasGroupNorm(torch_norm=torch_norm)

    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_norm(data)
    torch_out = torch_norm(torch.as_tensor(data))
    assert np.allclose(keras_out, torch_out.numpy(), rtol=1e-3, atol=1e-6)


def test_layer_norm():
    torch_norm = LayerNorm(128, bias=False)
    keras_norm = KerasLayerNorm(torch_norm=torch_norm)

    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_norm(data)
    torch_out = torch_norm(torch.as_tensor(data))
    assert np.allclose(keras_out, torch_out.numpy(), rtol=1e-3, atol=1e-6)


def test_time_mix(torch_time_mix: RWKV_TimeMix_x051a, torch_state: LayerState, keras_state: KerasLayerState):
    keras_time_mix = Keras_RWKV_TimeMix_x051a(torch_time_mix)
    
    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_time_mix(data, *keras_state[1:])
    torch_out = torch_time_mix.forward_step(
        torch.as_tensor(data), torch_state.time_mixer_state, torch_state.kv_state
    )

    for i in range(len(torch_out)):
        assert np.allclose(keras_out[i], torch_out[i].numpy(), rtol=1e-3, atol=1e-6)


def test_channel_mix(torch_channel_mix: RWKV_ChannelMix_x051a, torch_state: LayerState, keras_state: KerasLayerState):
    keras_channel_mix = Keras_RWKV_ChannelMix_x051a(torch_channel_mix)
    
    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)
    keras_out = keras_channel_mix(data, keras_state[0])
    torch_out = torch_channel_mix.forward_step(
        torch.as_tensor(data), torch_state.channel_mixer_state
    )

    for i in range(len(torch_out)):
        assert np.allclose(keras_out[i], torch_out[i].numpy(), rtol=1e-3, atol=1e-6)


def test_rwkv_module(torch_block: Block, torch_state: LayerState, keras_state: KerasLayerState):
    keras_block = Keras_RWKV_Block(torch_block)
    
    data = np.random.randn(TEST_BATCH_SIZE, 128).astype(np.float32)

    keras_out, *keras_state = keras_block(data, *keras_state)
    torch_out, torch_state = torch_block.forward_step(torch.as_tensor(data), torch_state)
    torch_state = KerasLayerState.from_state(torch_state)

    for i in range(len(torch_state)):
        assert np.allclose(keras_state[i], torch_state[i], rtol=1e-3, atol=1e-6)
    assert np.allclose(keras_out, torch_out.numpy(), rtol=1e-3, atol=1e-6)


def test_rwkv_model(model: GPT):
    keras_model = Keras_RWKV(model)
    conf: GPTConfig = model.config

    data = np.random.randint(0, conf.vocab_size, size=(TEST_BATCH_SIZE, conf.block_size))
    positions, keras_states = keras_model.get_states(TEST_BATCH_SIZE)

    logits_gt, _ = model(torch.as_tensor(data), warn=False)
    for i in range(data.shape[1]):
        logits, keras_states = keras_model([data[:, i], positions, *keras_states])
        assert np.allclose(logits_gt[:, i].numpy(), logits, rtol=1e-2, atol=1e-4), i
        positions += 1

    return keras_model


def test_keras_modules(model):
    test_linear()
    test_group_norm()
    test_layer_norm()

    torch_state = LayerState(model.config, TEST_BATCH_SIZE)
    torch_state.kv_state = torch.randn_like(torch_state.kv_state)
    torch_state.time_mixer_state = torch.randn_like(torch_state.time_mixer_state)
    torch_state.channel_mixer_state = torch.randn_like(torch_state.channel_mixer_state)

    keras_state = KerasLayerState.from_state(torch_state)

    module = model.transformer.h[0]

    test_time_mix(module.tmix, torch_state, keras_state)
    test_channel_mix(module.cmix, torch_state, keras_state)
    test_rwkv_module(module, torch_state, keras_state)
    return test_rwkv_model(model)


def maybe_sampling(checkpoint, keras_model, runner):
    import os, pickle

    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

        str_tokens = ['▁привет', ',', '▁как']
        tokens = encode(str_tokens)
        positions, flatten_states = keras_model.get_states(1)

        for token in tokens:
            token = np.array([token], dtype=np.int32)
            logits, *flatten_states = runner([token, positions, *flatten_states])
            positions += 1

        temperature = 1
        max_tokens = keras_model.config.block_size
        num_generated_tokens = max_tokens - len(tokens)
        for _ in range(num_generated_tokens):
            # softmax
            probs = logits[0] / temperature
            probs -= np.max(probs)
            probs = np.exp(probs)
            probs = probs / np.sum(probs)

            # sample
            token = np.random.choice(len(probs), p=probs)
            token = np.array([token], dtype=np.int32)
            tokens.append(token[0])

            # next prediction
            logits, *flatten_states = runner([token, positions, *flatten_states])
            positions += 1

        print(decode(tokens))


