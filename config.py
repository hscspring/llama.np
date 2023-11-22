from typing import Optional
from dataclasses import dataclass



@dataclass
class ModelArgs:
    
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    max_seq_len: int = 2048
    norm_eps: float = 1e-6

    max_batch_size: int = 32
