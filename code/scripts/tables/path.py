from pathlib import Path

import numpy as np
import pandas as pd
from pyprojroot import here

out_dir = Path(here()).parent / "tex" / "data"
out_dir.mkdir(exist_ok=True)

dir = Path(here()) / "results" / "path_real"

flist = [p for p in dir.glob("*.csv") if p.is_file()]

col_names = ["Dataset", "Method", "Time"]

df_real = pd.DataFrame(columns=col_names)

for f in flist:
    df_new = pd.read_csv(f, header=None, names=col_names)
    df_real = pd.concat([df_real, df_new])


method_map = {
    "admm": "ADMM",
    "anderson": "Anderson (PGD)",
    "fista": "FISTA",
    "hybrid_cd": "hybrid (ours)",
}

dataset_map = {
    "Rhee2006": "Rhee2006",
    "bcTCGA": "bcTCGA",
    "news20.binary": "news20",
    "rcv1.binary": "rcv1",
}

df_real = df_real.assign(
    Dataset=df_real.Dataset.map(dataset_map), Method=df_real.Method.map(method_map)
)

df_real_wide = df_real.pivot(index="Method", columns="Dataset", values="Time")

df_real_wide.to_csv(out_dir / "path_real.csv")

# Simulated Data

dir = Path(here()) / "results" / "path_simulated"
flist = [p for p in dir.glob("*.csv") if p.is_file()]

col_names = ["Scenario", "Method", "Time"]

df_simulated = pd.DataFrame(columns=col_names)

for f in flist:
    df_new = pd.read_csv(
        f,
        header=None,
        names=col_names,
        dtype={"Scenario": str, "Method": str, "Time": np.float64},
    )
    df_simulated = pd.concat([df_simulated, df_new])


df_simulated = df_simulated.assign(
    Method=df_simulated.Method.map(method_map),
)

df_simulated_wide = df_simulated.pivot(
    index="Method", columns="Scenario", values="Time"
)

df_simulated_wide.to_csv(out_dir / "path_simulated.csv")
