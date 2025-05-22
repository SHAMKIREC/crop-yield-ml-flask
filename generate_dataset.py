import pandas as pd
import numpy as np
import random

# Установка seed для воспроизводимости
np.random.seed(42)
random.seed(42)

# Параметры для генерации данных
n_samples = 500  # Количество записей
soil_types = ["Чернозем", "Суглинок", "Песчаная"]
crops = ["Пшеница", "Кукуруза", "Подсолнечник"]

# Функция для генерации синтетических данных
def generate_crop_yield_data():
    data = {
        "temperature": np.random.uniform(15, 35, n_samples),  # Температура в °C
        "rainfall": np.random.uniform(50, 200, n_samples),    # Осадки в мм
        "fertilizer_amount": np.random.uniform(100, 300, n_samples),  # Удобрения в кг/га
        "soil_type": [random.choice(soil_types) for _ in range(n_samples)],
        "crop": [random.choice(crops) for _ in range(n_samples)]
    }
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Генерируем целевую переменную (урожайность) на основе входных данных
    # с добавлением некоторого шума для реалистичности
    
    # Базовые коэффициенты влияния для разных культур
    crop_base = {
        "Пшеница": 3.5,
        "Кукуруза": 5.0,
        "Подсолнечник": 2.8
    }
    
    # Коэффициенты влияния для разных типов почв
    soil_factor = {
        "Чернозем": 1.2,
        "Суглинок": 1.0,
        "Песчаная": 0.8
    }
    
    # Генерация урожайности
    yields = []
    for i in range(n_samples):
        # Базовая урожайность для культуры
        base_yield = crop_base[df.loc[i, "crop"]]
        
        # Влияние почвы
        soil_effect = soil_factor[df.loc[i, "soil_type"]]
        
        # Влияние температуры (оптимум около 22-28°C)
        temp = df.loc[i, "temperature"]
        temp_effect = 1.0 - 0.03 * abs(temp - 25)
        
        # Влияние осадков (больше - лучше, но с насыщением)
        rain = df.loc[i, "rainfall"]
        rain_effect = min(1.2, 0.5 + rain / 200)
        
        # Влияние удобрений (больше - лучше, но с насыщением)
        fert = df.loc[i, "fertilizer_amount"]
        fert_effect = min(1.3, 0.7 + fert / 300)
        
        # Итоговая урожайность с небольшим случайным шумом
        final_yield = base_yield * soil_effect * temp_effect * rain_effect * fert_effect
        final_yield *= np.random.uniform(0.9, 1.1)  # Добавляем случайность ±10%
        
        yields.append(round(final_yield, 2))
    
    df["yield"] = yields
    
    return df

# Генерируем данные
crop_data = generate_crop_yield_data()

# Сохраняем в CSV
crop_data.to_csv("crop_yield_dataset.csv", index=False)

print(f"Сгенерирован датасет с {n_samples} записями и сохранен в crop_yield_dataset.csv")
print("\nПример данных:")
print(crop_data.head())

print("\nСтатистика по урожайности для разных культур:")
print(crop_data.groupby("crop")["yield"].describe())

print("\nСтатистика по урожайности для разных типов почв:")
print(crop_data.groupby("soil_type")["yield"].describe())
