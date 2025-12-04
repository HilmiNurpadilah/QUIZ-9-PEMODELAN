import pandas as pd
import numpy as np

# ======================================================================
# PROGRAM LENGKAP PREDIKSI WISATAWAN KOTA BANDUNG MENGGUNAKAN MONTE CARLO
# ======================================================================

def load_raw_data(path_csv: str) -> pd.DataFrame:
    """
    Membaca data mentah wisatawan dari Open Data Bandung.
    """
    print("=== LOAD DATA RAW ===")
    df = pd.read_csv(path_csv)
    print(df.head())
    return df


def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data dan membuat agregasi TOTAL wisatawan per tahun.
    1. Pilih kolom penting
    2. Ubah format data
    3. Gabungkan DOMESTIK + MANCANEGARA
    """
    print("\n=== PEMROSESAN DATA ===")

    # 1. Pilih kolom penting
    df = df[["tahun", "jenis_wisatawan", "jumlah_wisatawan"]].copy()

    # Pastikan tipe datanya benar
    df["tahun"] = df["tahun"].astype(int)
    df["jumlah_wisatawan"] = df["jumlah_wisatawan"].astype(float)

    # 2. Pivot → dari panjang menjadi lebar
    df_pivot = df.pivot_table(
        index="tahun",
        columns="jenis_wisatawan",
        values="jumlah_wisatawan",
        aggfunc="sum"
    ).reset_index()

    # 3. Jika ada yang hilang, isi 0
    df_pivot["DOMESTIK"] = df_pivot.get("DOMESTIK", 0)
    df_pivot["MANCANEGARA"] = df_pivot.get("MANCANEGARA", 0)

    # 4. Hitung total wisatawan
    df_pivot["TOTAL"] = df_pivot["DOMESTIK"] + df_pivot["MANCANEGARA"]

    print(df_pivot)
    return df_pivot


def calculate_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung growth rate (pertumbuhan) dari tahun ke tahun.
    Rumus:
        growth = TOTAL_tahun_ini / TOTAL_tahun_sebelumnya
    """
    print("\n=== HITUNG GROWTH RATE ===")

    df = df.sort_values("tahun")
    df["growth_rate"] = df["TOTAL"] / df["TOTAL"].shift(1)

    df_growth = df.dropna().reset_index(drop=True)

    print(df_growth[["tahun", "TOTAL", "growth_rate"]])
    return df_growth


def build_probability_distribution(growth_rates: list) -> pd.DataFrame:
    """
    Membangun distribusi probabilitas growth rate.
    """
    print("\n=== DISTRIBUSI PROBABILITAS GROWTH RATE ===")

    df = pd.DataFrame(growth_rates, columns=["growth_rate"])
    freq = df["growth_rate"].value_counts(normalize=True).reset_index()
    freq.columns = ["growth_rate", "probability"]

    print(freq)
    return freq


def monte_carlo_simulation(last_total: float,
                           dist: pd.DataFrame,
                           n_sim: int = 2000):
    """
    Melakukan simulasi Monte Carlo untuk memprediksi jumlah wisatawan
    tahun berikutnya.
    """

    print("\n=== SIMULASI MONTE CARLO ===")

    growth_vals = dist["growth_rate"].values
    probabilities = dist["probability"].values

    # Ambil sampel growth rate secara acak
    simulations = np.random.choice(growth_vals, size=n_sim, p=probabilities)

    # Hitung prediksi
    predictions = last_total * simulations

    # Ambil statistik penting
    pred_mean = predictions.mean()
    pred_min = predictions.min()
    pred_max = predictions.max()

    print(f"Total tahun terakhir  : {last_total}")
    print(f"Rata-rata prediksi    : {pred_mean:.2f}")
    print(f"Prediksi minimum      : {pred_min:.2f}")
    print(f"Prediksi maksimum      : {pred_max:.2f}")

    return pred_mean, pred_min, pred_max


def main():
    # Nama file mentah hasil unduhan dari Open Data Bandung
    raw_file = "jumlah_wisatawan_mancanegara_domestik_datang_ke_kota_bandung.csv"

    # 1. Load data
    df_raw = load_raw_data(raw_file)

    # 2. Bersihkan + agregasi
    df_agg = clean_and_aggregate(df_raw)

    # 3. Hitung growth rate
    df_growth = calculate_growth_rate(df_agg)

    # Convert growth rate ke list
    growth_list = df_growth["growth_rate"].tolist()

    # 4. Bangun distribusi probabilitas
    dist = build_probability_distribution(growth_list)

    # 5. Ambil total wisatawan terakhir
    last_total = df_agg["TOTAL"].iloc[-1]

    # 6. Jalankan simulasi Monte Carlo
    mean_pred, min_pred, max_pred = monte_carlo_simulation(
        last_total=last_total,
        dist=dist,
        n_sim=3000  # semakin besar semakin stabil
    )

    # 7. Tampilkan output akhir
    print("\n=== HASIL AKHIR PREDIKSI MONTE CARLO ===")
    print(f"Prediksi tahun berikutnya: {mean_pred:.0f} wisatawan")

