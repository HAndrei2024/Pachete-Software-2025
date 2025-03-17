import numpy as np
import pandas as pd

df = pd.read_csv("Dataset.csv")

print("Informatii dataset: \n")
print(df.info())

print("Descriptive statistics: \n")
print(df.describe(include='all'))

print("Valori lipsa: \n")
print(df.isnull().sum(axis = 0))

# Metode pentru tratarea valorilor lipsa dintr-un dataset

# 1. Eliminarea randurilor cu valori lipsa

df_dropped_rows = df.dropna()
print("\nEliminarea randurilor cu valori lipsa\n")
print(df_dropped_rows)
print("\n-------------------\n")

# 2. Inlocuirea valorilor lipsa cu valoarea medie a coloanei

df_mean = df.fillna(df.mean(numeric_only = True))
print("\nInlocuirea valorilor lipsa cu valoarea medie a coloanei\n")
print(df_mean)
print("\n-------------------\n")

# 3. Inlocuirea valorilor lipsa cu o valoare fixa

df_fixed_value = df.fillna(0)
print("\nInlocuirea valorilor lipsa cu o valoare fixa\n")
print(df_mean)
print("\n-------------------\n")


print("Nr. valori unice in fiecare coloana:\n")
print(df.nunique(axis = 0))

print("Valorile unice din fiecare coloana: \n")
for col in df.columns:
    print(f"Valori unice: {col} : {df[col].unique()}")
    print()

print("Afisare coloana localitate: \n")
print(df.loc[:,"SUBLOCALITY"])

print("Afisare coloana cu index 4: \n")
print(df.iloc[:,3])

print("Afisare proprietatilor din New York ce au numarul de paturi mai mare decat 3: \n")
print(df.loc[(df["SUBLOCALITY"] == "New York") & (df["BEDS"] > 3)])

print("Afisare caselor la vanzare ce au un pret intre 500000 - 2000000")
print(df.loc[(df["TYPE"] == "House for sale") & (df["PRICE"].between(500000, 2000000))])

print("Afisare ultimele 5 randuri si toate coloanele fara prima: \n")
print(df.iloc[-5:, 1:])

print("Afisarea proprietatilor aflate pe pozitii pare: \n")
print(df.iloc[::2, :])

print("Afisare valori pe localitati: \n")
df_values_localities = df.groupby(by="SUBLOCALITY").sum(numeric_only = True)
print(df_values_localities)

print("Afisare valori pt o localitate specifica: \n")
df_specific_locality = df[df["SUBLOCALITY"] == "New York"].sort_values("PRICE",ascending=False)
print(df_specific_locality)

print("Afisare valoari pt case de vanzare cu nr maxim de paturi: \n")
max_beds_house = df[df["TYPE"] == "House for sale"]["BEDS"].max()
df_max_house = df[(df["TYPE"] == "House for sale") & (df["BEDS"] == max_beds_house)]
print(df_max_house)

print("Nr. proprietati pe localitate: \n")
df_count_per_locality = df.groupby("SUBLOCALITY")["TYPE"].count()
print(df_count_per_locality)

print("Nr. mediu de paturi pe tip de proprietate: \n")
df_avg_beds_per_type = df.groupby("TYPE")["BEDS"].mean()
print(df_avg_beds_per_type)

print("Maxim metri patrati pe tip de proprietate: \n")
df_max_size_per_type = df.groupby("TYPE")["MetriPatratiLocuinta"].min()
print(df_max_size_per_type)

print("Pret mediu pe tip de proprietate: \n")
df_avg_price_per_type = df.groupby("TYPE")["PRICE"].mean()
print(df_avg_price_per_type)

print("Afisare randurilor cu cel mai mare pret pe tip de proprietate: \n")
df_max_price_per_type = df.loc[df.groupby("TYPE")["PRICE"].idxmax()]
print(df_max_price_per_type)