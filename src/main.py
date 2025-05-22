import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Загружаем модель (будет создана отдельно)
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")

# Проверяем, существует ли модель, если нет - создаем заглушку
if not os.path.exists(model_path):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Создаем простые данные для обучения модели-заглушки
    X = pd.DataFrame({
        'temperature': [20, 25, 22, 18, 30],
        'rainfall': [100, 80, 120, 90, 70],
        'fertilizer_amount': [200, 150, 180, 220, 190],
        'soil_type_Чернозем': [1, 0, 1, 0, 1],
        'soil_type_Суглинок': [0, 1, 0, 0, 0],
        'soil_type_Песчаная': [0, 0, 0, 1, 0],
        'crop_Пшеница': [1, 0, 0, 1, 0],
        'crop_Кукуруза': [0, 1, 0, 0, 1],
        'crop_Подсолнечник': [0, 0, 1, 0, 0]
    })
    y = pd.Series([4.5, 5.2, 3.8, 4.0, 6.1])  # Примерные значения урожайности
    model.fit(X, y)
    # Сохраняем модель-заглушку
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Создана модель-заглушка, так как настоящая модель не найдена")

# Загружаем модель
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Простейший HTML-интерфейс с улучшенным дизайном
HTML_FORM = """
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Прогноз урожайности</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        input[type="text"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
        }
        button {
            background-color: #27ae60;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #219653;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
            border-left: 5px solid #27ae60;
        }
        .error {
            color: #c0392b;
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Прогноз урожайности сельскохозяйственных культур</h1>
        <p>Введите параметры для прогнозирования урожайности с помощью модели машинного обучения.</p>
        
        <form method="post">
            <div class="form-group">
                <label for="temperature">Средняя температура (°C):</label>
                <input type="text" id="temperature" name="temperature" placeholder="Например: 22.5" required>
            </div>

            <div class="form-group">
                <label for="rainfall">Количество осадков (мм):</label>
                <input type="text" id="rainfall" name="rainfall" placeholder="Например: 120" required>
            </div>

            <div class="form-group">
                <label for="fertilizer_amount">Количество удобрений (кг/га):</label>
                <input type="text" id="fertilizer_amount" name="fertilizer_amount" placeholder="Например: 200" required>
            </div>

            <div class="form-group">
                <label for="soil_type">Тип почвы:</label>
                <select id="soil_type" name="soil_type" required>
                    <option value="Чернозем">Чернозем</option>
                    <option value="Суглинок">Суглинок</option>
                    <option value="Песчаная">Песчаная</option>
                </select>
            </div>

            <div class="form-group">
                <label for="crop">Сельскохозяйственная культура:</label>
                <select id="crop" name="crop" required>
                    <option value="Пшеница">Пшеница</option>
                    <option value="Кукуруза">Кукуруза</option>
                    <option value="Подсолнечник">Подсолнечник</option>
                </select>
            </div>

            <button type="submit">Рассчитать прогноз</button>
        </form>
        
        {% if prediction is not none %}
        <div class="result">
            <h3>Прогноз урожайности: {{ prediction }} т/га</h3>
            <p>Данный прогноз основан на введенных параметрах и исторических данных.</p>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="error">
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Обработка формы
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            temp = float(request.form["temperature"])
            rain = float(request.form["rainfall"])
            fert = float(request.form["fertilizer_amount"])
            soil = request.form["soil_type"]
            crop = request.form["crop"]

            # One-hot encoding
            input_df = pd.DataFrame([{
                "temperature": temp, 
                "rainfall": rain, 
                "fertilizer_amount": fert,
                "soil_type": soil, 
                "crop": crop
            }])
            
            input_encoded = pd.get_dummies(input_df, columns=["soil_type", "crop"])

            # Совмещаем с тренировочными колонками
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Убедимся, что порядок колонок совпадает
            input_encoded = input_encoded[model_features]

            # Предсказание
            pred = model.predict(input_encoded)[0]
            prediction = round(pred, 2)
            
        except Exception as e:
            error = f"Ошибка при обработке данных: {str(e)}"
            print(f"Error: {str(e)}")

    return render_template_string(HTML_FORM, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
