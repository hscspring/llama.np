import sys
import json
from sentencepiece import SentencePieceProcessor


def load_and_export(model_path):

    sp_model = SentencePieceProcessor(model_file=model_path)

    n_words: int = sp_model.vocab_size()
    bos_id: int = sp_model.bos_id()
    eos_id: int = sp_model.eos_id()
    pad_id: int = sp_model.pad_id()
    assert sp_model.vocab_size() == sp_model.get_piece_size()

    tokens, scores = [], []
    for i in range(n_words):
        t = sp_model.id_to_piece(i)
        
        # if i == bos_id:
        #     t = "\n<s>\n"
        # if i == eos_id:
        #     t = "\n</s>\n"
        if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            t = chr(int(t[3:5], 16)) # e.g. make "<0x01>" into "\x01"
        
        t = t.replace("▁", " ")
        
        tokens.append(t)
        s = sp_model.get_score(i)
        scores.append(s)

    token_model = {
        "tokens": tokens,
        "scores": scores,
    }
    with open("tokenizer.model.np", "w") as f:
        json.dump(token_model, f, ensure_ascii=False, indent=2)
    return token_model


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("[Model file path]")
        exit()

    model_path = sys.argv[1]
    load_and_export(model_path)