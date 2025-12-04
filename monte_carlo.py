import pandas as pd
import numpy as np

def load_raw_data(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    return df

def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["tahun", "jenis_wisatawan", "jumlah_wisatawan"]].copy()

    df["tahun"] = df["tahun"].astype(int)
    df["jumlah_wisatawan"] = df["jumlah_wisatawan"].astype(float)

    df_pivot = df.pivot_table(
        index="tahun",
        columns="jenis_wisatawan",
        values="jumlah_wisatawan",
        aggfunc="sum"
    ).reset_index()

    df_pivot["DOMESTIK"] = df_pivot.get("DOMESTIK", 0)
    df_pivot["MANCANEGARA"] = df_pivot.get("MANCANEGARA", 0)
    df_pivot["TOTAL"] = df_pivot["DOMESTIK"] + df_pivot["MANCANEGARA"]

    return df_pivot

def calculate_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("tahun")
    df["growth_rate"] = df["TOTAL"] / df["TOTAL"].shift(1)
    df_growth = df.dropna().reset_index(drop=True)
    return df_growth

def build_probability_distribution(growth_rates: list) -> pd.DataFrame:
    df = pd.DataFrame(growth_rates, columns=["growth_rate"])
    freq_df = df["growth_rate"].value_counts(normalize=True).reset_index()
    freq_df.columns = ["growth_rate", "probability"]
    return freq_df

def monte_carlo_simulation(last_total: float, dist: pd.DataFrame, n_sim: int = 2000):
    growth_values = dist["growth_rate"].values
    probabilities = dist["probability"].values

    simulations = np.random.choice(
        growth_values,
        size=n_sim,
        p=probabilities
    )

    predictions = last_total * simulations

    pred_mean = predictions.mean()
    pred_min = predictions.min()
    pred_max = predictions.max()

    return pred_mean, pred_min, pred_max
