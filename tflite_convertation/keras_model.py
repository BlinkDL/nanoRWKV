import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from model import GPTConfig, GPT, LayerState
from model import RWKV_ChannelMix_x051a, RWKV_TimeMix_x051a, LayerNorm, Block


class KerasLayerState(list):
    def __init__(self, cfg: GPTConfig, batch_size: int):
        self.extend(self.from_state(LayerState(cfg, batch_size)))

    @staticmethod
    def from_state(state: LayerState):
        return [
            state.channel_mixer_state.numpy().astype(np.float32),
            state.time_mixer_state.numpy().astype(np.float32),
            state.kv_state.numpy().astype(np.float32),
        ]


class KerasLinear(L.Dense):
    def __init__(self, torch_linear: torch.nn.Linear, **kwargs):
        out_channels, in_channels = torch_linear.weight.shape
        use_bias = torch_linear.bias is not None

        super().__init__(units=out_channels, use_bias=use_bias, **kwargs)
        self.build((None, in_channels))
        
        weights = [torch_linear.weight.transpose(1, 0).numpy()]
        if use_bias:
            weights.append(torch_linear.bias.numpy())
        self.set_weights(weights)


class KerasEmbedding(L.Embedding):
    def __init__(self, torch_embed: torch.nn.Embedding, **kwargs):
        super().__init__(
            input_dim=torch_embed.num_embeddings,
            output_dim=torch_embed.embedding_dim,
        )
        self.build((None, ))
        self.set_weights([torch_embed.weight.numpy()])


class KerasGroupNorm(L.Layer):
    def __init__(self, torch_norm: torch.nn.GroupNorm, **kwargs):
        super().__init__()
        self.eps = torch_norm.eps
        self.num_groups = torch_norm.num_groups
        self.weight = torch_norm.weight.numpy()
        self.bias = torch_norm.bias.numpy()

    def call(self, x):
        dim = x.shape[-1]
        x = tf.reshape(x, (-1, self.num_groups, dim // self.num_groups))
        centered = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.math.squared_difference(x, centered), axis=-1, keepdims=True)
        x = (x - centered) * tf.math.rsqrt(var + self.eps)
        x = tf.reshape(x, (-1, dim))
        return x * self.weight + self.bias


class KerasLayerNorm(L.LayerNormalization):
    def __init__(self, torch_norm: LayerNorm, **kwargs):
        use_bias = torch_norm.bias is not None
        super().__init__(
            epsilon=torch_norm.eps,
            center=use_bias,
        )
        self.build((None, torch_norm.weight.size(-1)))

        weights = [torch_norm.weight.numpy()]
        if use_bias:
            weights.append(torch_norm.bias.numpy())
        self.set_weights(weights)


class Keras_RWKV_ChannelMix_x051a(L.Layer):
    def __init__(self, layer: RWKV_ChannelMix_x051a):
        super().__init__()
        self.time_maa_k = layer.time_maa_k.squeeze(-2).numpy()
        self.time_maa_r = layer.time_maa_r.squeeze(-2).numpy()

        self.receptance = KerasLinear(layer.receptance, activation='sigmoid')
        self.key = KerasLinear(layer.key, activation='relu')
        self.value = KerasLinear(layer.value)

    def call(self, x, state):
        xx = state - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        out = self.key(xk) ** 2
        out = self.value(out)
        out = self.receptance(xr) * out
        return out, x


class Keras_RWKV_TimeMix_x051a(L.Layer):
    def __init__(self, layer: RWKV_TimeMix_x051a):
        super().__init__()
        self.n_head = layer.n_head
        self.head_size = layer.head_size

        self.time_maa_k = layer.time_maa_k.squeeze(-2).numpy()
        self.time_maa_v = layer.time_maa_v.squeeze(-2).numpy()
        self.time_maa_r = layer.time_maa_r.squeeze(-2).numpy()
        self.time_maa_g = layer.time_maa_g.squeeze(-2).numpy()

        self.time_decay = np.exp(-np.exp(layer.time_decay.unsqueeze(-1).numpy()))
        self.time_faaaa = layer.time_faaaa.unsqueeze(-1).numpy()

        self.ln_x = KerasGroupNorm(layer.ln_x)

        self.key = KerasLinear(layer.key)
        self.value = KerasLinear(layer.value)
        self.receptance = KerasLinear(layer.receptance)
        self.gate = KerasLinear(layer.gate, activation='silu')

        self.out = KerasLinear(layer.output)

    def call(self, x, state, kv_state):
        H, N = self.n_head, self.head_size

        xx = state - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g

        k = tf.reshape(self.key(xk), (-1, H, N, 1))
        v = tf.reshape(self.value(xv), (-1, H, 1, N))
        r = tf.reshape(self.receptance(xr), (-1, H, 1, N))
        g = self.gate(xg) # extra gate

        w = self.time_decay
        u = self.time_faaaa

        y, kv_state = self.single_timestep(r, k, v, u, w, kv_state)

        y = tf.reshape(y, (-1, H * N))
        y = self.ln_x(y) * g
        y = self.out(y)
        return y, x, kv_state

    @staticmethod
    def single_timestep(r, k, v, u, w, kv_state):
        y = kv_state        # BHKV
        y = y + (k @ v) * u # BHKV * HK1 + BHKV = BHKV
        out = r @ y         # BH1K @ BHKV = BH1V

        kv_state = kv_state * w         # BHKV
        kv_state = kv_state + (k @ v)   # BHKV * HK1 + BHKV = BHKV

        return tf.squeeze(out, -2), kv_state # BHV, BHKV


class Keras_RWKV_Block(L.Layer):

    def __init__(self, rwkv_block=Block, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = KerasLayerNorm(rwkv_block.ln_1)
        self.tmix = Keras_RWKV_TimeMix_x051a(rwkv_block.tmix)
        self.ln_2 = KerasLayerNorm(rwkv_block.ln_2)
        self.cmix = Keras_RWKV_ChannelMix_x051a(rwkv_block.cmix)

    def call(self, x, channel_mixer_x_state, time_mixer_x_state, kv_state):
        out, time_mixer_x_state, kv_state = \
            self.tmix(self.ln_1(x), time_mixer_x_state, kv_state)
        x = x + out
        out, channel_mixer_x_state = \
            self.cmix(self.ln_2(x), channel_mixer_x_state)
        x = x + out
        return x, channel_mixer_x_state, time_mixer_x_state, kv_state


class Keras_RWKV(L.Layer):

    def __init__(self, model: GPT):
        super().__init__()
        self.wte = KerasEmbedding(model.transformer.wte)
        self.wpe = KerasEmbedding(model.transformer.wpe)
        self.blocks = [Keras_RWKV_Block(block) for block in model.transformer.h]
        self.ln_f = KerasLayerNorm(model.transformer.ln_f)
        self.lm_head = KerasLinear(model.lm_head)
        self.config: GPTConfig = model.config

    def call(self, input):
        x, pos, *states = input
        tok_emb = self.wte(x) # token embeddings of shape (b, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (n_embd)
        x = tok_emb + pos_emb
        for layer_id, block in enumerate(self.blocks):  # run each rwkv block
            x, states[layer_id * 3 + 0], states[layer_id * 3 + 1], states[layer_id * 3 + 2] = block(
                x, 
                states[layer_id * 3 + 0], 
                states[layer_id * 3 + 1], 
                states[layer_id * 3 + 2])
        logits = self.lm_head(self.ln_f(x))
        return logits, states

    def get_states(self, batch_size):
        return (np.zeros(batch_size, dtype=np.int32),
                [state for _ in range(self.config.n_layer)
                for state in KerasLayerState(self.config, batch_size)])
