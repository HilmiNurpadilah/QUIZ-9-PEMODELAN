from flask import Flask, render_template, request
from monte_carlo import (
    load_raw_data,
    clean_and_aggregate,
    calculate_growth_rate,
    build_probability_distribution,
    monte_carlo_simulation
)

# =========================================================
# SIAPKAN APLIKASI
# =========================================================
app = Flask(__name__)

# =========================================================
# LOAD & PERSIAPAN DATA SEKALI SAAT APP START
# =========================================================

CSV_PATH = "jumlah_wisatawan_mancanegara_domestik_datang_ke_kota_bandung.csv"

# 1. Load data mentah
df_raw = load_raw_data(CSV_PATH)

# 2. Bersihkan & agregasi (DOMESTIK + MANCANEGARA → TOTAL)
df_agg = clean_and_aggregate(df_raw)

# 3. Hitung growth rate
df_growth = calculate_growth_rate(df_agg)
growth_list = df_growth["growth_rate"].tolist()

# 4. Bangun distribusi probabilitas
dist = build_probability_distribution(growth_list)

# 5. Ambil total wisatawan tahun terakhir
last_total = df_agg["TOTAL"].iloc[-1]

# 6. Data tahun untuk ditampilkan di halaman awal
data_tahun = df_agg[["tahun", "TOTAL"]].to_dict(orient="records")


# =========================================================
# ROUTES FLASK
# =========================================================

@app.route("/", methods=["GET"])
def index():
    """
    Halaman utama: form input jumlah simulasi + tabel data.
    """
    return render_template("index.html", data_tahun=data_tahun)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Jalankan simulasi Monte Carlo berdasarkan input user.
    """
    try:
        n_sim = int(request.form.get("n_simulasi", 3000))
        if n_sim <= 0:
            n_sim = 3000
    except:
        n_sim = 3000

    # Jalankan model
    mean_pred, min_pred, max_pred = monte_carlo_simulation(
        last_total=last_total,
        dist=dist,
        n_sim=n_sim
    )

    # Siapkan data untuk ditampilkan
    result = {
        "n_sim": n_sim,
        "last_total": int(last_total),
        "mean_pred": int(mean_pred),
        "min_pred": int(min_pred),
        "max_pred": int(max_pred),
    }

    return render_template("result.html", result=result)


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
