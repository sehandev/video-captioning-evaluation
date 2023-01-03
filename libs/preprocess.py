from pathlib import Path
import pandas as pd


def load_data(data_dir: str, data_filename: str, rename_dict: dict = None):
    print("\n[ Load data ]")
    data_dir = Path(data_dir).absolute()
    data_path = data_dir / data_filename
    print(f"Read csv from {data_path}")

    data_df = pd.read_csv(data_path)
    if rename_dict:
        data_df = data_df[rename_dict.keys()]
        data_df = data_df.rename(columns=rename_dict)
    data_df = data_df.sort_values(by="movie_id", ignore_index=True)
    return data_df
