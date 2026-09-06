import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

X_train = pd.read_csv('X_train_hazir.csv')
X_test  = pd.read_csv('X_test_hazir.csv')
y_train = pd.read_csv('y_train_hazir.csv').squeeze()
y_test  = pd.read_csv('y_test_hazir.csv').squeeze()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_lr.pkl')
print("Veri ölçeklendirildi ve 'scaler_lr.pkl' kaydedildi.")

model_lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(
    model_lr,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Grid Search basliyor...")
grid.fit(X_train_scaled, y_train)

print(f"En iyi C degeri : {grid.best_params_['C']}")
print(f"En iyi F1 skoru : {grid.best_score_:.3f}")


best_lr = grid.best_estimator_
y_pred  = best_lr.predict(X_test_scaled)
y_prob  = best_lr.predict_proba(X_test_scaled)[:, 1]


joblib.dump(best_lr, 'kanser_taramasi_lr_model.pkl')
print("Model 'kanser_taramasi_lr_model.pkl' olarak kaydedildi.")

np.save('y_pred_lr.npy', y_pred)
np.save('y_prob_lr.npy', y_prob)

print("Tahminler kaydedildi: y_pred_lr.npy, y_prob_lr.npy")
print("Lojistik Regresyon basaraili.")

katsayilar = pd.DataFrame({
    'Öznitelik': X_train.columns,
    'Katsayı (Etki)': best_lr.coef_[0]
}).sort_values(by='Katsayı (Etki)', ascending=False)

katsayilar.to_csv('oznitelik_katsayilari_lr.csv', index=False)
print("Öznitelik katsayıları 'oznitelik_katsayilari_lr.csv' olarak kaydedildi.")
