from flask import Flask, render_template, request
from monte_carlo import (
    load_raw_data,
    clean_and_aggregate,
    calculate_growth_rate,
    build_probability_distribution,
    monte_carlo_simulation
)

app = Flask(__name__)

# =======================
# LOAD MODEL SEKALI SAAT START
# =======================

CSV_PATH = "jumlah_wisatawan_mancanegara_domestik_datang_ke_kota_bandung.csv"

df_raw = load_raw_data(CSV_PATH)
df_agg = clean_and_aggregate(df_raw)
df_growth = calculate_growth_rate(df_agg)
growth_list = df_growth["growth_rate"].tolist()
dist = build_probability_distribution(growth_list)

last_total = df_agg["TOTAL"].iloc[-1]
data_tahun = df_agg[["tahun", "TOTAL"]].to_dict(orient="records")

years = [row["tahun"] for row in data_tahun]
totals = [row["TOTAL"] for row in data_tahun]

# =======================
# ROUTES
# =======================

@app.route("/")
def index():
    return render_template(
        "index.html",
        data_tahun=data_tahun,
        years=years,
        totals=totals
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        n_sim = int(request.form.get("n_simulasi", 3000))
        if n_sim <= 0:
            n_sim = 3000
    except:
        n_sim = 3000

    mean_pred, min_pred, max_pred = monte_carlo_simulation(
        last_total=last_total,
        dist=dist,
        n_sim=n_sim
    )

    result = {
        "n_sim": n_sim,
        "last_total": int(last_total),
        "mean_pred": int(mean_pred),
        "min_pred": int(min_pred),
        "max_pred": int(max_pred),
    }

    return render_template("result.html", result=result)

# =======================
# JANGAN JALANKAN app.run() DI RAILWAY
# =======================
# if __name__ == "__main__":
#     app.run(debug=True)
