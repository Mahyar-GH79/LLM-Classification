from torch.utils.data import Dataset

class LLMTripleDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row['prompt']
        res_a = row['response_a']
        res_b = row['response_b']
        text_b = f"Response A: {res_a} Response B: {res_b}"
        encoded = self.tokenizer(
            prompt,
            text_b,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}

        if not self.is_test:
            item['labels'] = int(row['label'])
        else:
            item['id'] = row['id']
        return item
