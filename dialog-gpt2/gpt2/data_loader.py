

class DataLoaderList:
    def __init__(self, B, T):
        pass
        # self.B = B
        # self.T = T
        # print(f"start {datetime.now().time()}")
        # df = pd.read_csv('../data/full_text.csv')
        # print(df.info)
        # text = '\n'.join(list(df['text']))
        # print(f"end {datetime.now().time()}")
        # # with open('../../data/input.txt', 'r') as f:
        # #      text = f.read()
        #
        # # enc = tiktoken.get_encoding('gpt2')
        # # tokens = enc.encode(text)
        # tokenizer = Tokenizer()
        # tokenizer.load('tokenizer/models/basic.model')
        # tokens = tokenizer.encode(text)
        # print(f"end encode {datetime.now().time()}")
        # self.tokens = torch.tensor(tokens)
        # print(f"load {len(self.tokens)} tokens")
        # print(f"epoch = {len(self.tokens) // (B * T)} batches")
        # self.current_position = 0

    def next_batch(self):
        pass
        # B, T = self.B, self.T
        # buf = self.tokens[self.current_position: self.current_position+B*T+1]
        # x = (buf[:-1]).view(B, T)
        # y = (buf[1:]).view(B, T)
        #
        # self.current_position += B * T
        # if self.current_position + (B * T + 1 ) > len(self.tokens):
        #     self.current_position = 0
        # return x, y