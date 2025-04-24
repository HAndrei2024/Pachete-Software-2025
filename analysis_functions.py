import pandas as pd
import io

my_df = pd.read_csv("Dataset.csv")

def custom_info(df=my_df) -> str:
    lines = []
    lines.append(f"Number of rows: {df.shape[0]}")
    lines.append(f"Number of columns: {df.shape[1]}")

    lines.append("\nColumns and types:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")

    total_missing = df.isnull().sum().sum()
    lines.append(f"\nTotal missing values: {total_missing}")
    if total_missing > 0:
        lines.append("Missing values by column:")
        for col in df.columns[df.isnull().any()]:
            lines.append(f"  - {col}: {df[col].isnull().sum()}")

    mem_usage_kb = df.memory_usage(deep=True).sum() / 1024
    lines.append(f"\nEstimated memory usage: {mem_usage_kb:.2f} KB")

    return "\n".join(lines)


def descriptive_statistics(df=my_df):
    return df.describe()


def prelucreaza_dataset(df=my_df):
    df = df.loc[df["PRICE"].between(500000, 195000000)]
    df["BATH"] = df["BATH"].astype("int64")

    return df


def show_dataset_info(df=my_df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = "📝 Informatii dataset:\n" + buffer.getvalue() + "\n"

    desc = df.describe(include='all')
    info_str += "\n📊 Statistici descriptive:\n" + str(desc) + "\n"

    nulls = df.isnull().sum()
    info_str += "\n❗ Valori lipsa pe coloane:\n" + str(nulls) + "\n"

    return info_str



def handle_missing_values(df=my_df):
    results = []

    df_dropped_rows = df.dropna()
    results.append("Eliminarea randurilor cu valori lipsa:\n" + str(df_dropped_rows))

    df_mean = df.fillna(df.mean(numeric_only=True))
    results.append("Inlocuirea valorilor lipsa cu media coloanei:\n" + str(df_mean))

    df_fixed_value = df.fillna(0)
    results.append("Inlocuirea valorilor lipsa cu valoare fixa (0):\n" + str(df_fixed_value))

    return "\n\n-------------------\n\n".join(results)


def show_unique_values(df=my_df):
    unique_counts = "Nr. valori unice in fiecare coloana:\n" + str(df.nunique(axis=0)) + "\n\n"
    unique_vals = "Valorile unice din fiecare coloana:\n"
    for col in df.columns:
        unique_vals += f"{col}: {df[col].unique()}\n"
    return unique_counts + unique_vals


def column_displays(df=my_df):
    localitate = "Coloana 'SUBLOCALITY':\n" + str(df["SUBLOCALITY"]) + "\n"
    index_col = "Coloana cu index 4 (a 5-a coloana):\n" + str(df.iloc[:, 3]) + "\n"
    return localitate + index_col


def filter_data(df=my_df):
    ny_beds = df.loc[(df["SUBLOCALITY"] == "New York") & (df["BEDS"] > 3)]
    house_price_range = df.loc[(df["TYPE"] == "House for sale") & (df["PRICE"].between(500000, 2000000))]
    
    return (
        "Proprietati in New York cu mai mult de 3 paturi:\n" + str(ny_beds) + "\n\n" +
        "Case la vanzare cu pret intre 500000 si 2000000:\n" + str(house_price_range)
    )


def display_row_selection(df=my_df):
    last_rows = "Ultimele 5 randuri (fara prima coloana):\n" + str(df.iloc[-5:, 1:]) + "\n"
    even_rows = "Randurile aflate pe pozitii pare:\n" + str(df.iloc[::2, :]) + "\n"
    return last_rows + even_rows


def groupby_analysis1(df=my_df):
    results = []

    results.append("Valori pe localitati:\n" + str(df.groupby("SUBLOCALITY").sum(numeric_only=True)))

    ny_props = df[df["SUBLOCALITY"] == "New York"].sort_values("PRICE", ascending=False)
    results.append("Proprietati din New York sortate dupa pret:\n" + str(ny_props))

    max_beds = df[df["TYPE"] == "House for sale"]["BEDS"].max()
    max_beds_df = df[(df["TYPE"] == "House for sale") & (df["BEDS"] == max_beds)]
    results.append("Case de vanzare cu nr maxim de paturi:\n" + str(max_beds_df))

    results.append("Nr. proprietati pe localitate:\n" + str(df.groupby("SUBLOCALITY")["TYPE"].count()))
    results.append("Nr. mediu de paturi pe tip de proprietate:\n" + str(df.groupby("TYPE")["BEDS"].mean()))
    results.append("Minim metri patrati pe tip de proprietate:\n" + str(df.groupby("TYPE")["MetriPatratiLocuinta"].min()))
    results.append("Pret mediu pe tip de proprietate:\n" + str(df.groupby("TYPE")["PRICE"].mean()))

    max_price_per_type = df.loc[df.groupby("TYPE")["PRICE"].idxmax()]
    results.append("Randuri cu pretul maxim pe tip de proprietate:\n" + str(max_price_per_type))

    return "\n\n-------------------\n\n".join(results)

def groupby_analysis(df=my_df):
    lines = []

    # Locality summary
    locality_summary = df.groupby("SUBLOCALITY").sum(numeric_only=True)
    lines.append(f"### Valori pe localitati:")
    lines.append(str(locality_summary))

    # New York properties sorted by price
    ny_props = df[df["SUBLOCALITY"] == "New York"].sort_values("PRICE", ascending=False)
    lines.append(f"\n### Proprietati din New York sortate dupa pret:")
    lines.append(str(ny_props))

    # Maximum number of beds for "House for sale"
    max_beds = df[df["TYPE"] == "House for sale"]["BEDS"].max()
    max_beds_df = df[(df["TYPE"] == "House for sale") & (df["BEDS"] == max_beds)]
    lines.append(f"\n### Case de vanzare cu nr maxim de paturi:")
    lines.append(str(max_beds_df))

    # Property counts per locality
    locality_counts = df.groupby("SUBLOCALITY")["TYPE"].count()
    lines.append(f"\n### Nr. proprietati pe localitate:")
    lines.append(str(locality_counts))

    # Average number of beds per property type
    avg_beds_by_type = df.groupby("TYPE")["BEDS"].mean()
    lines.append(f"\n### Nr. mediu de paturi pe tip de proprietate:")
    lines.append(str(avg_beds_by_type))

    # Minimum square meters per property type
    min_size_by_type = df.groupby("TYPE")["MetriPatratiLocuinta"].min()
    lines.append(f"\n### Minim metri patrati pe tip de proprietate:")
    lines.append(str(min_size_by_type))

    # Average price per property type
    avg_price_by_type = df.groupby("TYPE")["PRICE"].mean()
    lines.append(f"\n### Pret mediu pe tip de proprietate:")
    lines.append(str(avg_price_by_type))

    # Maximum price per property type
    max_price_per_type = df.loc[df.groupby("TYPE")["PRICE"].idxmax()]
    lines.append(f"\n### Randuri cu pretul maxim pe tip de proprietate:")
    lines.append(str(max_price_per_type))

    return "\n".join(lines)



# ----

import streamlit as st
import pandas as pd

def get_locality_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("SUBLOCALITY").sum(numeric_only=True)

def get_ny_properties_sorted_by_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["SUBLOCALITY"] == "New York"].sort_values("PRICE", ascending=False)

def get_max_beds_house_for_sale(df: pd.DataFrame) -> pd.DataFrame:
    max_beds = df[df["TYPE"] == "House for sale"]["BEDS"].max()
    return df[(df["TYPE"] == "House for sale") & (df["BEDS"] == max_beds)]

def get_property_counts_per_locality(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("SUBLOCALITY")["TYPE"].count()

def get_avg_beds_per_property_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("TYPE")["BEDS"].mean()

def get_min_size_per_property_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("TYPE")["MetriPatratiLocuinta"].min()

def get_avg_price_per_property_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("TYPE")["PRICE"].mean()

def get_max_price_per_property_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("TYPE")["PRICE"].idxmax()]
