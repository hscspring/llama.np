import sys
import time
import numpy as np

from config import ModelArgs
from model import Llama
from tokenizer import Tokenizer


args = ModelArgs(288, 6, 6, 6, 32000, None, 256)

token_model_path = "./tokenizer.model.np"
model_path = "./stories15M.model.npz"

tok = Tokenizer(token_model_path)
llama = Llama(model_path, args)

if len(sys.argv) == 1:
    prompt = "<s>"
else:
    prompt = sys.argv[1]

print(f"\n{prompt}", end="")
ids = tok.encode(prompt)
input_ids = np.array([ids], dtype=np.int32)
start = time.time()
token_num = input_ids.shape[1]
for ids in llama.generate(input_ids, args.max_seq_len, True, 1.0, 0.9, 0):
    output_ids = ids[0].tolist()
    token_num += 1
    if output_ids[-1] in [tok.eos_id, tok.bos_id]:
        break
    output_text = tok.decode(output_ids)
    print(output_text, end="")
    sys.stdout.flush()
end = time.time()
cost = end - start
print(f"\n\nToken count: {token_num}, cost: {cost:.2f}s, {round(token_num/cost)}tokens/s")