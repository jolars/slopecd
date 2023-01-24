from pathlib import Path

import pandas as pd
from pyprojroot import here


out_dir = Path(here()).parent / "tex" / "tables"

dir = Path(here()) / "results" / "path"

flist = [p for p in dir.glob("*.csv") if p.is_file()]

col_names = ["Dataset", "Method", "Time"]

df = pd.DataFrame(columns=col_names)

for f in flist:
    df_new = pd.read_csv(f, header=None, names=col_names)
    df = pd.concat([df, df_new])


method_map = {
    "admm": "ADMM",
    "anderson": "Anderson (PGD)",
    "fista": "FISTA",
    "hybrid_cd": "Hybrid (ours)",
}

dataset_map = {
    "Rhee2006": "Rhee2006",
    "bcTCGA": "bcTCGA",
    "news20.binary": "news20",
    "rcv1.binary": "rcv1",
}

# df = df.assign(Method=df.Method.map(method_map), Dataset=df.Dataset.map(dataset_map))

df_wide = df.pivot(index="Method", columns="Dataset", values="Time")

df_wide.to_csv(out_dir / "path.csv")
