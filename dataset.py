from torch.utils.data import Dataset


class LLMTripleDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = (
            f"Prompt: {row['prompt']} "
            f"A and B are responses to the question asked in the prompt. "
            f"A: {row['response_a']} "
            f"B: {row['response_b']}"
                )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if not self.is_test:
            item['labels'] = int(row['label'])
        else:
            item['id'] = row['id']
        return item
