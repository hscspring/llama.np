# llama.np

Why this repo?

Nothing, just to learn the model and for fun.

It's better to run karpathy's [tinyllamas](https://huggingface.co/karpathy/tinyllamas).


## Install Deps

```bash
pip install numpy
```

## Usage

```bash
# Download stories15M.bin from https://huggingface.co/karpathy/tinyllamas/tree/main to current dir

# Convert model
python convert_bin_llama_to_np.py ./stories15M.bin

# Generate
python main.py "Once upon"
"""
Once upon a time, there wa a girl named Amy. She loved to draw and write. One day, she wa out in her house looking for something to do.
When she got to her room, she found a big, shiny box. Amy wa very excited and started to write on the box. But then, she had a bad feeling. She had to delay her writing until the sun went down.
Finally, she took the box outside and opened it. Inside, there were lot of crayon and paper. Amy started to draw a big sun in the sky. Then she began to write. She made a beautiful drawing of the sun with her crayon.
In the end, Amy had a beautiful drawing of a good sun. She wa very happy and proud of her work. She learned that sometime thing need to be stopped and paid off.

Token count: 180, cost: 23.80s, 8tokens/s
"""
```

## Reference

- [karpathy/llama2.c: Inference Llama 2 in one file of pure C](https://github.com/karpathy/llama2.c)
- [tairov/llama2.py: Inference Llama 2 in one file of pure Python](https://github.com/tairov/llama2.py)
- [facebookresearch/llama: Inference code for LLaMA models](https://github.com/facebookresearch/llama)
- [Sample the next token from a probability distribution using top-k and/or nucleus (top-p) sampling](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)
- [transformers/src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [transformers/src/transformers/generation](https://github.com/huggingface/transformers/tree/main/src/transformers/generation)
