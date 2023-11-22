import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_and_export(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.eval();
    dct = {}
    for key,val in model.named_parameters():
        dct[key] = val.detach().cpu().numpy()
    np.savez_compressed("./model", **dct)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("[Model folder path]")
        exit()

    model_path = sys.argv[1]
    load_and_export(model_path)