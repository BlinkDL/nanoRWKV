"""
RWKV "x051a" model - does not require custom CUDA kernel to train :)

References:
https://github.com/BlinkDL/RWKV-LM

Inference:
Always fast, and VRAM will not grow, because RWKV does not need KV cache.

Training:
Because we are not using custom CUDA kernel here, training is slightly slower than gpt+flash_attn when ctxlen is short.
Training becomes faster than gpt+flash_attn when ctxlen is long.
"""

import math, warnings
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerState:
    # the recurrent neural network (RNN) state for a layer of RWKV5.2 
    def __init__(self, cfg: GPTConfig, batch_size, dtype=torch.float32, device='cpu'):
        B, C, H, K = batch_size, cfg.n_embd, cfg.n_head, cfg.n_embd // cfg.n_head
        V = K

        # a (B,C) size tensor representing latest time mixer token embedding processed
        self.time_mixer_state = torch.zeros(B,C, dtype=dtype, device=device)
        # an (B,H,K,V) size tensor representing a decaying token embedding memory for each head, where H=number_of_heads, K=key_dim_per_head, V=value_dim_per_head 
        self.kv_state = torch.zeros(B,H,K,V, dtype=dtype, device=device)
        # a (B,C) size tensor representing latest channel mixer token embedding processed
        self.channel_mixer_state = torch.zeros(B,C, dtype=dtype, device=device)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = 1e-5

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class RWKV_TimeMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5)*64)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, warn=True):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        #
        # we divide a block into chunks to speed up computation & save vram.
        # you can try to find the optimal chunk_len for your GPU.
        # avoid going below 128 if you are using bf16 (otherwise time_decay might be less accurate).
        #
        if T % 256 == 0: Q = 256
        elif T % 128 == 0: Q = 128
        else:
            Q = T
            if warn:
                warnings.warn(f'\n{"#"*80}\n\n{" "*38}Note\nThe GPT-mode forward() should only be called when we are training models.\nNow we are using it for inference for simplicity, which works, but will be very inefficient.\n\n{"#"*80}\n')
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2) # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1) # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2) # value
        g = F.silu(self.gate(xg)) # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float())) # time_decay
        u = self.time_faaaa.float() # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype) # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype) # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype) # output

        for i in range(T // Q): # the rwkv-x051a operator
            rr = r[:, :, i*Q:i*Q+Q, :]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y

    def forward_step(self, x, state, kv_state):
        B, H, N = x.size(0), self.n_head, self.head_size

        xx = state - x
        xk = x + xx * self.time_maa_k.squeeze(-2)
        xv = x + xx * self.time_maa_v.squeeze(-2)
        xr = x + xx * self.time_maa_r.squeeze(-2)
        xg = x + xx * self.time_maa_g.squeeze(-2)
        r = self.receptance(xr).view(B, H, 1, N)
        k = self.key(xk).view(B, H, N, 1)
        v = self.value(xv).view(B, H, 1, N)
        g = F.silu(self.gate(xg)) # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1) # time_decay
        u = self.time_faaaa.float().unsqueeze(-1) # time_first

        y, kv_state = self.single_time_step(r, k, v, u, w, kv_state)

        y = y.contiguous().view(B, H * N)
        y = self.ln_x(y) * g

        # output projection
        y = self.dropout(self.output(y))
        return y, x, kv_state

    @staticmethod
    def single_time_step(r, k, v, u, w, kv_state):
        y = kv_state                     # BHKV
        y = y + (k @ v) * u              # BHKV * HK1 + BHKV = BHKV
        out = r @ y                      # BH1K @ BHKV = BH1V

        kv_state = kv_state * w          # BHKV
        kv_state = kv_state + (k @ v)    # BHKV * HK1 + BHKV = BHKV

        return out.squeeze(-2), kv_state # BHV, BHKV


class RWKV_ChannelMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.value = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x

    def forward_step(self, x, state):
        xx = state - x
        xk = x + xx * self.time_maa_k.squeeze(-2)
        xr = x + xx * self.time_maa_r.squeeze(-2)

        out = self.key(xk)
        out = torch.relu(out) ** 2
        out = self.value(out)
        out = torch.sigmoid(self.receptance(xr)) * out
        out = self.dropout(out)
        return out, x


class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_TimeMix_x051a(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_ChannelMix_x051a(config, layer_id)

    def forward(self, x, warn=True):
        x = x + self.tmix(self.ln_1(x), warn=warn)
        x = x + self.cmix(self.ln_2(x))
        return x

    def forward_step(self, x, s: LayerState):
        out, s.time_mixer_state, s.kv_state = \
            self.tmix.forward_step(self.ln_1(x), s.time_mixer_state, s.kv_state)
        x = x + out
        out, s.channel_mixer_state = \
            self.cmix.forward_step(self.ln_2(x), s.channel_mixer_state)
        x = x + out
        return x, s


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('tmix.output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, warn=True):
        device = idx.device
        time = idx.size(1)
        assert time <= self.config.block_size, f"Cannot forward sequence of length {time}, block size is only {self.config.block_size}"
        pos = torch.arange(0, time, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, warn)
        x = self.transformer.ln_f(x)

        logits, loss = self.lm_head(x), None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def forward_step(self, token, pos, state):
        tok_emb = self.transformer.wte(token) # token embeddings of shape (b, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (n_embd)
        token = self.transformer.drop(tok_emb + pos_emb)
        for layer_id, block in enumerate(self.transformer.h):  # run each rwkv block
            token, state[layer_id] = block.forward_step(token, state[layer_id])
        token = self.transformer.ln_f(token)
        logits = self.lm_head(token)
        return logits, state

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'time_' not in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'time_' in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        dtype = self.lm_head.weight.dtype
        device = self.lm_head.weight.device
        states = [LayerState(self.config, 1, dtype, device) for _ in range(self.config.n_layer)]
        pos = torch.zeros(1, dtype=torch.long, device=device)

        for i in range(len(tokens)):
            logits, states = self.forward_step(tokens[i].unsqueeze(0), pos, states)
            pos += 1
        
        top_k = min(top_k, logits.size(-1))
        for _ in range(max_tokens - len(tokens)):
            # pluck the logits at the final step
            logits = logits[0]
            # optionally crop the logits to only the top k options
            if top_k is not None:
                logits, ids = torch.topk(logits, top_k)
            # scale by desired temperature and apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits / temperature, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            if top_k is not None:
                next_token = ids[next_token]
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=0)
            # forward the model to get the logits for the index in the sequence
            logits, states = self.forward_step(next_token, pos, states)
            pos += 1

        return tokens
