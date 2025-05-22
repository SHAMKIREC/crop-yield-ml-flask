import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Проверяем наличие датасета, если его нет - генерируем
if not os.path.exists("crop_yield_dataset.csv"):
    print("Датасет не найден. Запускаем скрипт генерации данных...")
    exec(open("generate_dataset.py").read())

# Загрузка данных
print("Загрузка данных из crop_yield_dataset.csv...")
df = pd.read_csv("crop_yield_dataset.csv")

print("\nРазмер датасета:", df.shape)
print("\nПервые 5 строк датасета:")
print(df.head())

# One-hot encoding категориальных признаков
print("\nПрименяем one-hot encoding для категориальных признаков...")
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop"])

# Разделение признаков и целевой переменной
X = df_encoded.drop("yield", axis=1)
y = df_encoded["yield"]

print("\nПризнаки после one-hot encoding:")
print(X.columns.tolist())

# Деление на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nРазмер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Обучение модели
print("\nОбучение модели RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка на тренировочной выборке
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)  # Вычисляем RMSE вручную
train_r2 = r2_score(y_train, train_predictions)
print(f"Тренировочная выборка - RMSE: {train_rmse:.2f} т/га, R²: {train_r2:.2f}")

# Оценка на тестовой выборке
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)  # Вычисляем RMSE вручную
test_r2 = r2_score(y_test, test_predictions)
print(f"Тестовая выборка - RMSE: {test_rmse:.2f} т/га, R²: {test_r2:.2f}")

# Важность признаков
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков:")
print(feature_importance)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Признак'], feature_importance['Важность'])
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.title('Важность признаков в модели RandomForest')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("График важности признаков сохранен в feature_importance.png")

# Визуализация предсказаний и реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Реальные значения урожайности (т/га)')
plt.ylabel('Предсказанные значения урожайности (т/га)')
plt.title('Сравнение предсказанных и реальных значений урожайности')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png')
print("График сравнения предсказаний и реальных значений сохранен в prediction_vs_actual.png")

# Сохранение модели
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nМодель сохранена в model.pkl")
