import os
import time
from datasets import load_dataset
from bpe.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
pattern = None # regexp for chunking
vocab_size = 1000 + 256 # our vocab size + UTF-8 codes
model_name = "bpe" # save model to file with model name
texts_path = "../data/bpe" # path for train data
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    file_paths = []
    wiki_dataset = load_dataset("danasone/wikipedia_ru")["train"]
    for filename in os.listdir(texts_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(texts_path, filename)
            file_paths.append(file_path)

    t0 = time.time()
    texts = [content for path in file_paths for content in [open(path, 'r', encoding='utf-8').read()]]
    texts.extend(wiki_dataset['text'])
    t1= time.time()
    print(sum(len(s) for s in texts))
    delta = (t1 - t0) * 1000
    print(f"load time {delta:.2f}")
    os.makedirs("bpe/models", exist_ok=True)

    t0 = time.time()
    tokenizer = Tokenizer()
    tokenizer.train(texts, vocab_size, verbose=True)
    prefix = os.path.join("bpe/models", model_name)
    tokenizer.save(prefix)
    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")