import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_sas('LLCP2024.XPT', format='xport')

df.columns = [
    col.decode('utf-8') if isinstance(col, bytes) else col
    for col in df.columns
]

HEDEF = 'CHCOCNC1'

BAGIMSIZ = [
    'CHCSCNC1',
    'CHECKUP1',
    '_AGEG5YR',
    'SMOKE100',
    'DIABETE4',
    'EXERANY2',
    'ASTHMA3',
    'CHCKDNY2',
    'HAVARTH4',
    'CVDINFR4',
    'PERSDOC3',
    '_RFHLTH',
    'INCOME3',
    'EDUCA',
    '_BMI5CAT',
]

df = df[[HEDEF] + BAGIMSIZ].copy() 
print("Boyut:", df.shape)

df['CHCOCNC1'] = df['CHCOCNC1'].replace([7, 9], np.nan) 
df['CHCSCNC1'] = df['CHCSCNC1'].replace([7, 9], np.nan) 
df['CHECKUP1'] = df['CHECKUP1'].replace([7, 8, 9], np.nan) 
df['_AGEG5YR'] = df['_AGEG5YR'].replace([14], np.nan)
df['SMOKE100']  = df['SMOKE100'].replace([7, 9], np.nan) 
df['DIABETE4']  = df['DIABETE4'].replace([7, 9], np.nan) 
df['EXERANY2']  = df['EXERANY2'].replace([7, 9], np.nan) 
df['ASTHMA3']   = df['ASTHMA3'].replace([7, 9], np.nan) 
df['CHCKDNY2']  = df['CHCKDNY2'].replace([7, 9], np.nan) 
df['HAVARTH4']  = df['HAVARTH4'].replace([7, 9], np.nan) 
df['CVDINFR4']  = df['CVDINFR4'].replace([7, 9], np.nan) 
df['PERSDOC3']  = df['PERSDOC3'].replace([7, 9], np.nan) 
df['_RFHLTH']   = df['_RFHLTH'].replace([9], np.nan) 
df['INCOME3']   = df['INCOME3'].replace([77, 99], np.nan)
df['EDUCA']     = df['EDUCA'].replace([9], np.nan)




ikili_kolonlar = [
    'CHCSCNC1',  # cilt kanseri var mı (1 0)
    'EXERANY2',  # egzersiz yaptı mı (1 0)
    'ASTHMA3',   # astım var mı (1 0) 
    'CHCKDNY2',  # böbrek hastalığı var mı (1 0)
    'HAVARTH4',  # artrit var mı (1 0)
    'CVDINFR4',  # kalp krizi geçirdi mi (1 0)
    'PERSDOC3',  # kişisel doktoru var mı (1 1 0)
    'SMOKE100',  # 100+ sigara içti mi (1 0)
]

for kolon in ikili_kolonlar:
    df[kolon] = df[kolon].map({1.0: 1, 2.0: 0})
df['DIABETE4'] = df['DIABETE4'].map({1.0: 1, 2.0: 0, 3.0: 0, 4.0: 0}) # (1 0 0 0) yaptık çünkü hamileliği ve sınır diyabeti katmıyoruz.
df['PERSDOC3'] = df['PERSDOC3'].map({1.0: 1, 2.0: 1, 3.0: 0}) # (1 1 0) şeklinde yaptık çünkü evet evet hayır yazıyor kısaca.
df['CHCOCNC1'] = df['CHCOCNC1'].map({1.0: 1, 2.0: 0}) # hedef değişkeni de 1 ve 0 formatına getirdik.


df = df.dropna(subset=['CHCOCNC1'])
print(f"Kalan satır: {len(df):,}")

df.to_csv('brfss2024_temiz.csv', index=False)
