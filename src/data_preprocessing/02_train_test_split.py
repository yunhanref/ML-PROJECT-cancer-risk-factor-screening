import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler 


df = pd.read_csv('SONbrfss2024_temiz.csv')

BAGIMSIZ = [
    'CHCSCNC1', 'CHECKUP1', '_AGEG5YR', 'SMOKE100',
    'DIABETE4', 'EXERANY2', 'ASTHMA3', 'CHCKDNY2',
    'HAVARTH4', 'CVDINFR4', 'PERSDOC3', '_RFHLTH',
    'INCOME3', 'EDUCA', '_BMI5CAT',
]
X = df[BAGIMSIZ]
y = df['CHCOCNC1'] 

X_train, X_test, y_train, y_test = train_test_split( 
    X, y,
    test_size=0.20, 
    random_state=42,
    stratify=y
)

print(f"Eğitim seti: {len(X_train):,} satır")
print(f"Test seti  : {len(X_test):,} satır")
print(f"Eğitimde kanser oranı: %{y_train.mean()*100:.1f}")
print(f"Testte kanser oranı  : %{y_test.mean()*100:.1f}")

imputer = IterativeImputer(random_state=42, max_iter=10)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=BAGIMSIZ
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=BAGIMSIZ
)

print("Eksik veri doldurma (IterativeImputer) tamamlandı.")
print(f"Train'de kalan NaN: {X_train.isna().sum().sum()}")
print(f"Test'te kalan NaN : {X_test.isna().sum().sum()}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=BAGIMSIZ
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=BAGIMSIZ
)
print("Ölçeklendirme tamamlandı.")

X_train_scaled.to_csv("X_train_hazir.csv", index=False)
X_test_scaled.to_csv("X_test_hazir.csv", index=False)
y_train.to_csv("y_train_hazir.csv", index=False)
y_test.to_csv("y_test_hazir.csv", index=False)
print("Tüm veriler ayrı dosyalar olarak başarıyla kaydedildi!")
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=BAGIMSIZ
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=BAGIMSIZ
)

print("Veri Ölçekleme (Scaling) tamamlandı.")
