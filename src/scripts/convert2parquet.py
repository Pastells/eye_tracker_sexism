import os
import glob
from pathlib import Path

import pandas as pd

def convert(tsv_file):
    tsv_path = Path(tsv_file)
    parquet_file = tsv_path.with_suffix(".parquet")
    if not os.path.exists(parquet_file):
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        df.to_parquet(parquet_file, index=False)


tsv_file = "sexism4 Data export.tsv"
for tsv_file in glob.glob("TFG/Data export - sexisme_20_*/*/*.tsv"):
    print(tsv_file)
    convert(tsv_file)
