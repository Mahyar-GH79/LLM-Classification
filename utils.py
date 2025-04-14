import json
import pandas as pd

def safe_parse(json_str):
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return ''.join(str(x) for x in parsed if x is not None)
        elif isinstance(parsed, str):
            return parsed
        else:
            return str(parsed)
    except Exception as e:
        return None

def preprocess_multiclass(df):
    # Apply safe parsing
    df['prompt'] = df['prompt'].apply(safe_parse)
    df['response_a'] = df['response_a'].apply(safe_parse)
    df['response_b'] = df['response_b'].apply(safe_parse)

    # Drop rows with any missing fields
    df = df.dropna(subset=['prompt', 'response_a', 'response_b'])

    def label_fn(row):
        if row['winner_model_a'] == 1:
            return 0
        elif row['winner_model_b'] == 1:
            return 1
        elif row['winner_tie'] == 1:
            return 2
        return -1

    df['label'] = df.apply(label_fn, axis=1)
    return df[df['label'] != -1].reset_index(drop=True)
