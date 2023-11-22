import pytest

import numpy as np
import torch


np_model = "./stories15M.model.npz"
pt_model = "./stories15M.pt"

dct = np.load(np_model)
ckpt = torch.load(pt_model)
state_dict = ckpt["model"]

args = ckpt["model_args"]
n_layers = args["n_layers"]


@pytest.mark.parametrize("lid", range(n_layers))
def test_ffn(lid):
    assert np.allclose(
        state_dict[f"layers.{lid}.feed_forward.w1.weight"], 
        dct[f"model.layers.{lid}.mlp.gate_proj.weight"]
    )

    assert np.allclose(
        state_dict[f"layers.{lid}.feed_forward.w2.weight"], 
        dct[f"model.layers.{lid}.mlp.down_proj.weight"]
    )

    assert np.allclose(
        state_dict[f"layers.{lid}.feed_forward.w3.weight"], 
        dct[f"model.layers.{lid}.mlp.up_proj.weight"]
    )


@pytest.mark.parametrize("lid", range(n_layers))
def test_rms(lid):
    assert np.allclose(
        state_dict[f"layers.{lid}.attention_norm.weight"], 
        dct[f"model.layers.{lid}.input_layernorm.weight"]
    )
    assert np.allclose(
        state_dict[f"layers.{lid}.ffn_norm.weight"], 
        dct[f"model.layers.{lid}.post_attention_layernorm.weight"]
    )


@pytest.mark.parametrize("name", ["q", "k", "v", "o"])
@pytest.mark.parametrize("lid", range(n_layers))
def test_attn(name, lid):
    assert np.allclose(
        state_dict[f"layers.{lid}.attention.w{name}.weight"], 
        dct[f"model.layers.{lid}.self_attn.{name}_proj.weight"]
    )


def test_un_block():
    assert np.allclose(
        state_dict["tok_embeddings.weight"], 
        dct["model.embed_tokens.weight"]
    )
    assert np.allclose(state_dict["norm.weight"], dct["model.norm.weight"])
    assert np.allclose(state_dict["output.weight"], dct["lm_head.weight"])

