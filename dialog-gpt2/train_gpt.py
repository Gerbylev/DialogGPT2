import os
import time
import pandas as pd
from bpe.tokenizer import Tokenizer

if __name__ == "__main__":
    text = "\n".join(list(pd.read_csv('../data/suai_pubs.csv')['text']))

    os.makedirs("models", exist_ok=True)

    t0 = time.time()
    for TokenizerClass, name in zip([Tokenizer], ["tokenizer"]):

        tokenizer = TokenizerClass()
        tokenizer.train(text, 1256, verbose=True)
        prefix = os.path.join("models", name)
        tokenizer.save(prefix)
    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")