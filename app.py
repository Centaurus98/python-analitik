import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Fungsi untuk memuat data dan membersihkan kolom gaji
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=";", engine="python")

    # Membersihkan kolom gaji
    data[["Min Salary", "Max Salary"]] = (
        data["Salary"].str.replace("$", "").str.split(" - ", expand=True)
    )
    data["Min Salary"] = pd.to_numeric(data["Min Salary"], errors="coerce")
    data["Max Salary"] = pd.to_numeric(data["Max Salary"], errors="coerce")
    data["Avg Salary"] = data[["Min Salary", "Max Salary"]].mean(axis=1)

    return data


# Fungsi untuk menghitung analisis
def analyze_data(data):
    avg_salary_by_job = (
        data.groupby("Job Title")["Avg Salary"].mean().sort_values(ascending=False)
    )
    avg_salary_by_score = (
        data.groupby("Company Score")["Avg Salary"].mean().sort_values(ascending=False)
    )
    avg_salary_by_location = (
        data.groupby("Location")["Avg Salary"].mean().sort_values(ascending=False)
    )
    job_title_count = data["Job Title"].value_counts()
    location_count = data["Location"].value_counts()

    return (
        avg_salary_by_job,
        avg_salary_by_score,
        avg_salary_by_location,
        job_title_count,
        location_count,
    )


# Fungsi Polynomial Regression
def polynomial_regression(data, degree):
    X = data[["Company Score"]].values
    y = data["Avg Salary"].values

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return X, y, y_pred, mse, r2


# Main app
st.title("Analisis Data Gaji Perusahaan Software")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file:
    # Load and analyze data
    data = load_data(uploaded_file)
    (
        avg_salary_by_job,
        avg_salary_by_score,
        avg_salary_by_location,
        job_title_count,
        location_count,
    ) = analyze_data(data)

    st.header("Rata-rata Gaji Berdasarkan Posisi Pekerjaan")
    st.write(avg_salary_by_job)

    fig, ax = plt.subplots()
    sns.barplot(x=avg_salary_by_job.values, y=avg_salary_by_job.index, ax=ax)
    ax.set_xlabel("Average Salary")
    ax.set_title("Average Salary by Job Title")
    st.pyplot(fig)

    st.header("Rata-rata Gaji Berdasarkan Skor Perusahaan")
    st.write(avg_salary_by_score)

    fig, ax = plt.subplots()
    sns.barplot(x=avg_salary_by_score.index, y=avg_salary_by_score.values, ax=ax)
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Company Score")
    st.pyplot(fig)

    st.header("Rata-rata Gaji Berdasarkan Lokasi")
    st.write(avg_salary_by_location)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=avg_salary_by_location.values, y=avg_salary_by_location.index, ax=ax)
    ax.set_xlabel("Average Salary")
    ax.set_title("Average Salary by Location")
    st.pyplot(fig)

    st.header("Frekuensi Posisi Pekerjaan")
    st.write(job_title_count)

    fig, ax = plt.subplots()
    job_title_count.plot.pie(autopct="%1.1f%%", ax=ax, startangle=90)
    ax.set_ylabel("")
    ax.set_title("Job Title Frequency")
    st.pyplot(fig)

    st.header("Frekuensi Lokasi")
    st.write(location_count)

    fig, ax = plt.subplots()
    location_count.plot.pie(autopct="%1.1f%%", ax=ax, startangle=90)
    ax.set_ylabel("")
    ax.set_title("Location Frequency")
    st.pyplot(fig)

    # Polynomial Regression Analysis
    st.header("Polynomial Regression: Company Score vs Average Salary")
    degree = st.slider("Pilih degree untuk Polynomial Regression", 1, 5, 2)
    X, y, y_pred, mse, r2 = polynomial_regression(data, degree)

    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Data Asli")
    ax.plot(X, y_pred, color="red", label=f"Polynomial Fit (Degree={degree})")
    ax.set_xlabel("Company Score")
    ax.set_ylabel("Average Salary")
    ax.set_title(
        f"Polynomial Regression (Degree={degree})\nMSE: {mse:.2f}, R2: {r2:.2f}"
    )
    ax.legend()
    st.pyplot(fig)

    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**R-squared (R²)**: {r2:.2f}")

else:
    st.write("Silakan unggah file CSV untuk melakukan analisis data.")
