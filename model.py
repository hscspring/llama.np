from __future__ import annotations
from typing import TypeVar, Generic, Tuple
import math
import numpy as np


Shape = TypeVar("Shape")
DType = TypeVar("DType", np.int_, np.float_)
class Array(np.ndarray, Generic[Shape, DType]):
    ...



def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def top_k_logits(nxt_logits: Array["B,VS"], k: int):
    _bs, vs = nxt_logits.shape
    assert k < vs
    idxes = nxt_logits.argpartition(-k, axis=-1)[:,[-k]]
    k_vals = np.take_along_axis(nxt_logits, idxes, axis=1)
    scores = np.where(nxt_logits < k_vals, -np.inf, nxt_logits)
    return scores


def top_p_logits(nxt_logits: Array["B,VS"], top_p: float):
    assert 0.0 < top_p < 1.0
    bs, _vs = nxt_logits.shape
    sorted_indices = np.argsort(nxt_logits, axis=-1)
    sorted_logits = np.take_along_axis(nxt_logits, sorted_indices, axis=-1)
    cum_probs = softmax(sorted_logits).cumsum(axis=-1)
    sorted_idxes_to_remove = cum_probs <= (1 - top_p)
    # Use broadcasting to scatter the boolean values to the original shape
    indices_to_remove = np.zeros_like(sorted_logits, dtype=bool)
    indices_to_remove[np.arange(bs)[:, None], sorted_indices] = sorted_idxes_to_remove
    # Mask the logits
    scores = np.where(indices_to_remove, -np.inf, nxt_logits)
    return scores


def sampling(probs: Array["B,VS"]):
    bs, vocab_size = probs.shape
    rng = np.random.default_rng()
    res = []
    for b in range(bs):
        bp = rng.choice(vocab_size, size=1, p=probs[b])
        res.append(bp)
    samples = np.stack(res)
    return samples


def do_sampling(
    logits: Array["B,VS"], 
    temperature: float, 
    top_p: float, 
    top_k: int,
) -> Array["B,1", np.int32]:
    if temperature > 0.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p > 0.0:
        logits = top_p_logits(logits, top_p)
    
    probs = softmax(logits)
    ids = sampling(probs)
    return ids


def load_parameters(model_path):
    return np.load(model_path)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


def precompute_freqs_cos_sin(head_dim: int, max_seq_len: int, theta: int = 10000):
    # ignore type
    inv_freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len)
    freqs_np = np.outer(t, inv_freqs)
    freqs_np = freqs_np.astype(np.float32)
    freqs_cos_np = np.cos(freqs_np)
    freqs_sin_np = np.sin(freqs_np)
    return freqs_cos_np, freqs_sin_np


class RMSNorm:
    
    def __init__(self, weight: Array["H"], eps: float):
        self.weight = weight
        self.eps = eps
    
    def __call__(self, x: Array["B,L,D", np.float16]):
        xdt = x.dtype
        x = x.astype(np.float32)
        z = (x ** 2).mean(-1, keepdims=True) + self.eps
        z = x / np.sqrt(z)
        z = z.astype(xdt)
        return z * self.weight


def apply_rotary_emb(
    xq: Array["B,L,QHN,HD"],
    xk: Array["B,L,KVHN,HD"],
    freqs_cos: Array["L,HD//2"],
    freqs_sin: Array["L,HD//2"]
):
    xqt = xq.dtype
    xkt = xk.dtype
    xq = xq.astype(np.float32)
    xk = xk.astype(np.float32)
    
    xqri: Array["B,L,QHN,HD//2,2"] = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri: Array["B,L,KVHN,HD//2,2"] = xk.reshape(xk.shape[:-1] + (-1, 2))
    
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    # B,L,QHN,HD//2   B,L,KVHN,HD//2
    xq_r, xq_i = xq_r.squeeze(-1), xq_i.squeeze(-1)
    xk_r, xk_i = xk_r.squeeze(-1), xk_i.squeeze(-1)
    
    # 1,L,1,HD//2
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    # B,L,QHN,HD//2
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # B,L,KVHN,HD//2
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # B,L,QHN,HD//2,2
    xq_out = np.stack([xq_out_r, xq_out_i], axis=-1)
    # B,L,KVHN,HD//2,2
    xk_out = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: Array["B,L,QHN,HD"] = xq_out.reshape(xq_out.shape[:-2] + (-1, ))
    xk_out: Array["B,L,KVHN,HD"] = xk_out.reshape(xk_out.shape[:-2] + (-1, ))

    return xq_out.astype(xqt), xk_out.astype(xkt)


def repeat_kv(x: Array["B,L,KVHN,HD"], n_rep: int):
    if n_rep == 1:
        return x
    z: Array["B,L,QHN,HD"] = np.repeat(x, n_rep, axis=2)
    return z


class Attention:
    def __init__(
        self, 
        q_weight: Array["D,D"],
        k_weight: Array["D,D"],
        v_weight: Array["D,D"],
        o_weight: Array["D,D"],
        args: ModelArgs
    ):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        # mask = np.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        # self.mask = np.triu(mask, k=1)

        self.cache_k = np.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = np.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def __call__(
        self,
        x: Array["B,L,D"],
        start_pos: int,
        mask: Optional[Array["CL,L"]],
        freqs_cos: Array["L,HD//2"],
        freqs_sin: Array["L,HD//2"],
    ) -> Array["B,L,D"]:
        bsz, seqlen, _ = x.shape

        # QKV
        xq = x @ self.q_weight
        xk = x @ self.k_weight
        xv = x @ self.v_weight
        
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        ks = self.cache_k[:bsz, : start_pos + seqlen]
        vs = self.cache_v[:bsz, : start_pos + seqlen]

        xk = repeat_kv(ks, self.n_rep)  # (bs, cache_len+seqlen, n_local_heads, head_dim)
        xv = repeat_kv(vs, self.n_rep)  # (bs, cache_len+seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(0, 2, 1, 3)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(0, 2, 1, 3)  # (bs, n_local_heads, cache_len+seqlen, head_dim)
        xv = xv.transpose(0, 2, 1, 3)  # (bs, n_local_heads, cache_len+seqlen, head_dim)


        # manual implementation
        # (bs, nh, seqlen, hd) @ (bs, nh, hd, cache_len+seqlen) => bs, nh, seqlen, cache_len+seqlen
        scores = np.matmul(xq, xk.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # (bs, n_local_heads, seqlen, cache_len+seqlen)
        # scores = scores + self.mask[:, :, :seqlen, :seqlen]
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        
        scores = softmax(scores)
        # (bs, n_local_heads, seqlen, head_dim)
        output = np.matmul(scores, xv)

        # (bs, seqlen, dim), like `x`
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)

        output: Array["B,L,D"] = output @ self.o_weight
        return output


class FeedForward:
    def __init__(
        self, 
        up_weight: Array["FH,H"],
        gate_weight: Array["FH,H"],
        down_weight: Array["H,FH"],
    ):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: Array["B,L,D"]) -> Array["B,L,D"]:
        # (bs, seqlen, dim) @ (ffn_dim, dim).T => (bs, seqlen, ffn_dim)
        z1 = x @ self.up_weight
        # (bs, seqlen, ffn_dim)
        z2 = x @ self.gate_weight
        z2 = silu(z2)
        # (bs, seqlen, ffn_dim)
        z3 = z1 * z2
        # (bs, seqlen, ffn_dim) @ (dim, ffn_dim).T => (bs, seqlen, dim)
        z = z3 @ self.down_weight
        return z


class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"), 
            eps=args.norm_eps
        )

    def __call__(
        self, 
        x: Array["B,L,D"],
        start_pos: int,
        mask: Array["CL+L,L"],
        freqs_cos: Array["L,HD//2"],
        freqs_sin: Array["L,HD//2"],
    ) -> Array["B,L,D"]:
        norm_x = self.input_layernorm(x)
        h1: Array["B,L,D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1
        
        norm_z = self.post_attention_layernorm(z)
        h2: Array["B,L,D"] = self.feed_forward(norm_z)
        out = z + h2
        return out


class Llama:

    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args      

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS,H"] = weight.get("model.embed_tokens.weight")
        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: Array["H,VS"] = weight.get("lm_head.weight").T

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        freqs_cos, freqs_sin = precompute_freqs_cos_sin(args.dim // args.n_heads, args.max_seq_len)
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        del weight

    def __call__(
        self, 
        input_ids: Array["B,L", np.int32],
        start_pos: int,
    ):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embedding[input_ids]
        freqs_cos = self.freqs_cos[start_pos: start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = np.full((seqlen, seqlen), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((seqlen, start_pos)), mask], axis=1)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, mask, freqs_cos, freqs_sin)
        h: Array["B,L,D"] = self.norm(h)
        # inference-time mini-optimization: only forward the output on the very last position
        logits: Array["B,1,VS"] = h[:, [-1], :] @ self.lm_head_weight
        return logits
    
    def generate(
        self, 
        input_ids: Array["B,L", np.int32], 
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ):
        prev_pos = 0
        _bs, prompt_len = input_ids.shape
        max_new_tokens = min(self.args.max_seq_len - prompt_len, max_new_tokens)
        for curr_pos in range(prompt_len, prompt_len + max_new_tokens):
            logits = self(input_ids[:,prev_pos: curr_pos], prev_pos)
            nxt_logits = logits[:, -1, :]
            if do_sample:
                nxt_ids = do_sampling(nxt_logits, temperature, top_p, top_k)
            else:
                probs = softmax(nxt_logits)
                nxt_ids = probs.argmax(-1, keepdims=True)
            prev_pos = curr_pos
            input_ids = np.concatenate([input_ids, nxt_ids], axis=1)
            yield nxt_ids


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("[Model folder path]")
        exit()

    from config import ModelArgs

    print("loading parameters...")
    model_path = sys.argv[1]
    args = ModelArgs()
    model = Llama(model_path, args)
    
    print("forwarding...")
    x = np.array([[1, 2, 4], [4, 3, 2]], dtype=np.int32)
    y = model(x)
    print(y)