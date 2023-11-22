import struct
import numpy as np


from config import ModelArgs


def read_floats(file, count):
    values = struct.unpack(str(count) + 'f', file.read(count * 4))
    return values


def get_attn(conf, w, layer_id):
    step = conf.dim * conf.dim
    wnp = np.array(w[layer_id * step: (layer_id + 1) * step], dtype=np.float32)
    return wnp.reshape(conf.dim, conf.dim)


def get_ffn(conf, w, layer_id, ffn_type):
    step = conf.dim * conf.hidden_dim
    wnp = np.array(w[layer_id * step: (layer_id + 1) * step], dtype=np.float32)
    if ffn_type == "down":
        out = wnp.reshape(conf.dim, conf.hidden_dim)
    else:
        # up, gate, hidde_dim > dim
        out = wnp.reshape(conf.hidden_dim, conf.dim)
    return out


def get_rms(conf, w, layer_id):
    step = conf.dim
    wnp = np.array(w[layer_id * step: (layer_id + 1) * step], dtype=np.float32)
    return wnp


def load_and_export(checkpoint: str):
    with open(checkpoint, "rb") as file:
        _config = file.read(struct.calcsize("7i"))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", _config)
        conf = ModelArgs(dim, n_layers, n_heads, n_kv_heads, vocab_size, hidden_dim, seq_len)
        
        token_embedding = read_floats(file, conf.vocab_size * conf.dim)
        
        rms_att_weight = read_floats(file, conf.n_layers * conf.dim)
        wq = read_floats(file, conf.n_layers * conf.dim * conf.dim)
        wk = read_floats(file, conf.n_layers * conf.dim * conf.dim)
        wv = read_floats(file, conf.n_layers * conf.dim * conf.dim)
        wo = read_floats(file, conf.n_layers * conf.dim * conf.dim)
        rms_ffn_weight = read_floats(file, conf.n_layers * conf.dim)
        w1 = read_floats(file, conf.n_layers * conf.dim * conf.hidden_dim)
        w2 = read_floats(file, conf.n_layers * conf.hidden_dim * conf.dim)
        w3 = read_floats(file, conf.n_layers * conf.dim * conf.hidden_dim)
        
        rms_final_weight = read_floats(file, conf.dim)
        # donot need
        # freq_cis_real = read_floats(file, conf.max_seq_len * (conf.dim // conf.n_heads) // 2)
        # freq_cis_imag = read_floats(file, conf.max_seq_len * (conf.dim // conf.n_heads) // 2)
    
    dct = {}
    dct["model.embed_tokens.weight"] = np.array(token_embedding, dtype=np.float32).reshape(conf.vocab_size, conf.dim)
    dct["lm_head.weight"] = np.array(token_embedding, dtype=np.float32).reshape(conf.vocab_size, conf.dim)
    dct["model.norm.weight"] = np.array(rms_final_weight, dtype=np.float32)

    for layer_id in range(conf.n_layers):
        
        dct[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = get_attn(conf, wq, layer_id)
        dct[f"model.layers.{layer_id}.self_attn.k_proj.weight"] = get_attn(conf, wk, layer_id)
        dct[f"model.layers.{layer_id}.self_attn.v_proj.weight"] = get_attn(conf, wv, layer_id)
        dct[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = get_attn(conf, wo, layer_id)
        
        dct[f"model.layers.{layer_id}.mlp.up_proj.weight"] = get_ffn(conf, w3, layer_id, "up")
        dct[f"model.layers.{layer_id}.mlp.gate_proj.weight"] = get_ffn(conf, w1, layer_id, "gate")
        dct[f"model.layers.{layer_id}.mlp.down_proj.weight"] = get_ffn(conf, w2, layer_id, "down")
        
        dct[f"model.layers.{layer_id}.input_layernorm.weight"] = get_rms(conf, rms_att_weight, layer_id)
        dct[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = get_rms(conf, rms_ffn_weight, layer_id)

        # donot need        
        # dct["freq_cis_real"] = np.array(freq_cis_real, dtype=np.float32).reshape(conf.max_seq_len, (conf.dim//conf.n_heads) // 2)
        # dct["freq_cis_imag"] = np.array(freq_cis_imag, dtype=np.float32).reshape(conf.max_seq_len, (conf.dim//conf.n_heads) // 2)

    np.savez_compressed("./stories15M.model", **dct)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("[Model path]")
        exit()

    model_path = sys.argv[1]
    load_and_export(model_path)