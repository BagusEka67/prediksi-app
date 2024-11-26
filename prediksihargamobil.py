import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score

# Memuat model prediksi
model = pickle.load(open('C:/Users/Bagus/model_prediksi_harga_mobil.sav', 'rb'))

# Menampilkan dataset
st.title('Prediksi Harga Mobil')

# Menampilkan dataset
st.header("Dataset")
df1 = pd.read_csv('c:/Users/Bagus/Sistem Cerdas/CarPrice_Assignment.csv')
st.dataframe(df1)

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Prediksi", "Visualisasi"])

if page == "Dataset":
    # Menampilkan matriks korelasi hanya untuk kolom numerik
    st.write("Matriks Korelasi")
    df_numeric = df1.select_dtypes(include=[float, int])  # Pilih kolom numerik saja
    corr_matrix = df_numeric.corr()  # Hitung korelasi
    st.dataframe(corr_matrix)

elif page == "Prediksi":
    st.title('Prediksi Harga Mobil')

    # Input nilai dari variable independen
    highwaympg = st.number_input('Masukkan nilai highwaympg', min_value=0, step=1)
    curbweight = st.number_input('Masukkan nilai curbweight', min_value=0, step=1)
    horsepower = st.number_input('Masukkan nilai horsepower', min_value=0, step=1)

    # Button untuk melakukan prediksi
if st.button('Prediksi'):
    # Prediksi harga mobil menggunakan model yang sudah dilatih
    car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

    # Konversi hasil prediksi ke dalam format string atau format lainnya
    harga_mobil_float = float(car_prediction[0])  # Ambil nilai pertama dari hasil prediksi (langsung)

    # Tampilkan hasil prediksi dengan format yang lebih mudah dibaca
    harga_mobil_formatted = f'${harga_mobil_float:,.2f}'  # Format dengan simbol dolar dan 2 desimal
    st.write(f"Prediksi Harga Mobil: {harga_mobil_formatted}")

elif page == "Visualisasi":
    st.title("Visualisasi Data")

    # Menampilkan grafik untuk kolom 'highwaympg'
    st.write("Grafik Highway-mpg")
    chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
    st.line_chart(chart_highwaympg)

    # Menampilkan grafik untuk kolom 'curbweight'
    st.write("Grafik Curbweight")
    chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
    st.line_chart(chart_curbweight)

    # Menampilkan grafik untuk kolom 'horsepower'
    st.write("Grafik Horsepower")
    chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
    st.line_chart(chart_horsepower)

    # Visualisasi Korelasi dalam bentuk Heatmap
    st.write("Heatmap Korelasi Antara Fitur")
    corr_matrix = df1.select_dtypes(include=[float, int]).corr()
    st.write(corr_matrix)

    # Membuat heatmap menggunakan Altair
    chart = alt.Chart(df1).mark_rect().encode(
        x='highwaympg',
        y='curbweight',
        color='horsepower',
        tooltip=['highwaympg', 'curbweight', 'horsepower']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# Menampilkan Metrik Evaluasi Model (MSE, RMSE, R2)
st.sidebar.subheader("Metrik Evaluasi Model")
y_test = df1['price']  # Kolom target harga mobil
X_test = df1[['highwaympg', 'curbweight', 'horsepower']]  # Fitur untuk prediksi

# Prediksi harga mobil untuk evaluasi
y_pred = model.predict(X_test)

# Evaluasi menggunakan metrik
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Tampilkan metrik evaluasi
st.sidebar.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.sidebar.write(f"R2 Score: {r2:.2f}")