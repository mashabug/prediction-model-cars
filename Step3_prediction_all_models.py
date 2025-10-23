import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

# Загрузка обработанных данных
try:
    df = pd.read_csv('/Users/mariabug/ML/preprocessed_car_data_with_binary.csv')
    print("Данные успешно загружены")
    print(f"Размерность данных: {df.shape}")
except:
    print("Файл не найден. Используем данные из предыдущего шага...")
    # Если файла нет, используем df_encoded из предыдущего кода
    final_df = X.copy()
    final_df['price_log'] = y
    df = final_df

# Проверяем данные
print("\nИнформация о данных:")
print(df.info())
print(f"\nПропущенные значения:")
print(df.isnull().sum().sum())

# Удаляем строки с пропущенной целевой переменной
df = df.dropna(subset=['price_log'])

# Разделение на признаки и целевую переменную
X = df.drop('price_log', axis=1)
y = df['price_log']

print(f"\nЦелевая переменная (price_log) статистика:")
print(f"Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")

# Проверяем и обрабатываем пропущенные значения в признаках
print(f"\nПропущенные значения в признаках: {X.isnull().sum().sum()}")

# Заполняем пропуски медианными значениями для числовых колонок
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())

# Если остались пропуски в нечисловых колонках, заполняем модой
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')

print(f"Пропущенные значения после обработки: {X.isnull().sum().sum()}")

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nРазмеры выборок:")
print(f"Обучающая: {X_train.shape}")
print(f"Тестовая: {X_test.shape}")


# Создаем пайплайн с импутацией и масштабированием для линейных моделей
def create_preprocessing_pipeline():
    """Создает пайплайн предобработки для линейных моделей"""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return preprocessor


# Модифицированная функция оценки моделей с обработкой пропусков
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Оценка модели и возврат метрик"""
    try:
        # Для линейных моделей используем пайплайн с импутацией
        if model_name in ['Linear Regression', 'Ridge', 'Lasso']:
            preprocessor = create_preprocessing_pipeline()
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Обучение модели
            pipeline.fit(X_train, y_train)

            # Предсказания
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            model_used = pipeline
        else:
            # Для tree-based моделей заполняем пропуски медианой
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()

            numeric_columns = X_train_processed.select_dtypes(include=[np.number]).columns
            X_train_processed[numeric_columns] = X_train_processed[numeric_columns].fillna(
                X_train_processed[numeric_columns].median()
            )
            X_test_processed[numeric_columns] = X_test_processed[numeric_columns].fillna(
                X_train_processed[numeric_columns].median()  # Используем медианы из тренировочной выборки
            )

            # Заполняем категориальные колонки
            categorical_columns = X_train_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in X_train_processed.columns:
                    mode_val = X_train_processed[col].mode()[0] if not X_train_processed[
                        col].mode().empty else 'unknown'
                    X_train_processed[col] = X_train_processed[col].fillna(mode_val)
                    X_test_processed[col] = X_test_processed[col].fillna(mode_val)

            # Обучение модели
            model.fit(X_train_processed, y_train)

            # Предсказания
            y_pred_train = model.predict(X_train_processed)
            y_pred_test = model.predict(X_test_processed)

            model_used = model

        # Преобразование обратно из логарифма
        y_test_rub = np.expm1(y_test)
        y_pred_test_rub = np.expm1(y_pred_test)

        # Метрики в логарифмированной шкале
        mse_log = mean_squared_error(y_test, y_pred_test)
        mae_log = mean_absolute_error(y_test, y_pred_test)

        # Метрики в рублях
        mse_rub = mean_squared_error(y_test_rub, y_pred_test_rub)
        mae_rub = mean_absolute_error(y_test_rub, y_pred_test_rub)
        rmse_rub = np.sqrt(mse_rub)
        r2_rub = r2_score(y_test_rub, y_pred_test_rub)

        # Средняя абсолютная процентная ошибка (MAPE)
        mape = np.mean(np.abs((y_test_rub - y_pred_test_rub) / np.maximum(y_test_rub, 1))) * 100

        return {
            'model': model_used,
            'mse_log': mse_log,
            'mae_log': mae_log,
            'mse_rub': mse_rub,
            'mae_rub': mae_rub,
            'rmse_rub': rmse_rub,
            'r2_rub': r2_rub,
            'mape': mape,
            'predictions': y_pred_test_rub
        }
    except Exception as e:
        print(f"Ошибка при обучении {model_name}: {e}")
        return None


# Список моделей для сравнения
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

# Сравнение моделей
print("\n" + "=" * 80)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 80)

results = {}
best_model = None
best_score = float('inf')

for name, model in models.items():
    print(f"\nОбучение {name}...")

    # Оценка модели
    result = evaluate_model(model, X_train, X_test, y_train, y_test, name)

    if result is not None:
        results[name] = result

        print(f"RMSE (рубли): {result['rmse_rub']:,.0f}")
        print(f"MAE (рубли): {result['mae_rub']:,.0f}")
        print(f"R²: {result['r2_rub']:.4f}")
        print(f"MAPE: {result['mape']:.2f}%")

        # Обновление лучшей модели
        if result['rmse_rub'] < best_score:
            best_score = result['rmse_rub']
            best_model = name
    else:
        print(f"Модель {name} не удалось обучить")

if not results:
    raise ValueError("Ни одна модель не была успешно обучена")

print(f"\nЛучшая модель: {best_model} с RMSE {best_score:,.0f} рублей")


# Дополнительная обработка данных для tree-based моделей (если они лучше)
def prepare_tree_data(X_train, X_test):
    """Подготовка данных для tree-based моделей"""
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Заполняем пропуски
    numeric_columns = X_train_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        median_val = X_train_processed[col].median()
        X_train_processed[col] = X_train_processed[col].fillna(median_val)
        X_test_processed[col] = X_test_processed[col].fillna(median_val)

    # Обрабатываем категориальные переменные (если остались)
    categorical_columns = X_train_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in X_train_processed.columns:
            # Frequency encoding для категориальных переменных
            freq_encoding = X_train_processed[col].value_counts().to_dict()
            X_train_processed[f'{col}_freq'] = X_train_processed[col].map(freq_encoding)
            X_test_processed[f'{col}_freq'] = X_test_processed[col].map(freq_encoding)

    # Удаляем исходные категориальные колонки
    X_train_processed = X_train_processed.drop(categorical_columns, axis=1)
    X_test_processed = X_test_processed.drop(categorical_columns, axis=1)

    return X_train_processed, X_test_processed


# Настройка гиперпараметров для лучшей модели
print(f"\nНастройка гиперпараметров для {best_model}...")

if best_model == 'Random Forest':
    # Подготавливаем данные для Random Forest
    X_train_rf, X_test_rf = prepare_tree_data(X_train, X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_rf, y_train)

    best_model_tuned = grid_search.best_estimator_
    print(f"Лучшие параметры: {grid_search.best_params_}")

    # Оценка tuned модели
    y_pred_test_rf = best_model_tuned.predict(X_test_rf)
    y_test_rub = np.expm1(y_test)
    y_pred_test_rub = np.expm1(y_pred_test_rf)

    tuned_rmse = np.sqrt(mean_squared_error(y_test_rub, y_pred_test_rub))
    print(f"Улучшение RMSE: {best_score:,.0f} -> {tuned_rmse:,.0f} рублей")

    # Сохраняем tuned модель
    results[f'{best_model} (tuned)'] = {
        'model': best_model_tuned,
        'rmse_rub': tuned_rmse,
        'predictions': y_pred_test_rub
    }

elif best_model == 'Gradient Boosting':
    # Подготавливаем данные для Gradient Boosting
    X_train_gb, X_test_gb = prepare_tree_data(X_train, X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1
    )
    grid_search.fit(X_train_gb, y_train)

    best_model_tuned = grid_search.best_estimator_
    print(f"Лучшие параметры: {grid_search.best_params_}")

    # Оценка tuned модели
    y_pred_test_gb = best_model_tuned.predict(X_test_gb)
    y_test_rub = np.expm1(y_test)
    y_pred_test_rub = np.expm1(y_pred_test_gb)

    tuned_rmse = np.sqrt(mean_squared_error(y_test_rub, y_pred_test_rub))
    print(f"Улучшение RMSE: {best_score:,.0f} -> {tuned_rmse:,.0f} рублей")

    results[f'{best_model} (tuned)'] = {
        'model': best_model_tuned,
        'rmse_rub': tuned_rmse,
        'predictions': y_pred_test_rub
    }


# Визуализация результатов
def plot_results(results, y_test):
    """Визуализация результатов моделей"""
    y_test_rub = np.expm1(y_test)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Сравнение метрик
    metrics_data = {}
    for model_name, result in results.items():
        if 'rmse_rub' in result:
            metrics_data[model_name] = {
                'RMSE': result['rmse_rub'],
                'MAE': result.get('mae_rub', 0),
                'R²': result.get('r2_rub', 0)
            }

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data).T

        if 'RMSE' in metrics_df.columns and 'MAE' in metrics_df.columns:
            metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Сравнение RMSE и MAE (рубли)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        if 'R²' in metrics_df.columns:
            metrics_df[['R²']].plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Сравнение R²')
            axes[0, 1].tick_params(axis='x', rotation=45)

    # 2. Предсказания vs Фактические значения (лучшая модель)
    best_model_name = min(results.items(), key=lambda x: x[1].get('rmse_rub', float('inf')))[0]
    best_predictions = results[best_model_name].get('predictions', [])

    if len(best_predictions) > 0:
        axes[1, 0].scatter(y_test_rub, best_predictions, alpha=0.5)
        axes[1, 0].plot([y_test_rub.min(), y_test_rub.max()],
                        [y_test_rub.min(), y_test_rub.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Фактическая цена')
        axes[1, 0].set_ylabel('Предсказанная цена')
        axes[1, 0].set_title(f'Предсказания vs Фактические значения ({best_model_name})')

    # 3. Ошибки предсказания
    if len(best_predictions) > 0:
        errors = best_predictions - y_test_rub
        axes[1, 1].hist(errors, bins=50, alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Ошибка предсказания (рубли)')
        axes[1, 1].set_ylabel('Частота')
        axes[1, 1].set_title('Распределение ошибок предсказания')

    plt.tight_layout()
    plt.show()


# Строим графики
plot_results(results, y_test)

# Сохранение лучшей модели
final_model = results[best_model]['model']

# Сохраняем модель
joblib.dump(final_model, 'car_price_predictor.pkl')
print(f"\nМодель сохранена как 'car_price_predictor.pkl'")


# Функция для предсказания на новых данных
def predict_car_price(car_features, model_path='car_price_predictor.pkl'):
    """
    Функция для предсказания цены автомобиля
    """
    try:
        # Загрузка модели
        model = joblib.load(model_path)

        # Обработка пропусков в новых данных
        car_features_processed = car_features.copy()

        # Заполняем пропуски медианами (нужно сохранить медианы из тренировочных данных)
        numeric_columns = car_features_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in car_features_processed.columns and car_features_processed[col].isnull().any():
                # В реальном применении нужно использовать медианы из тренировочных данных
                car_features_processed[col] = car_features_processed[col].fillna(
                    car_features_processed[col].median()
                )

        # Предсказание
        if hasattr(model, 'predict'):
            price_log_pred = model.predict(car_features_processed)[0]
        else:
            # Если это пайплайн
            price_log_pred = model.predict(car_features_processed)[0]

        # Преобразование обратно в рубли
        price_pred_rub = np.expm1(price_log_pred)

        # Доверительный интервал
        confidence_interval = price_pred_rub * 0.15  # ±15%

        return {
            'price_rub': round(price_pred_rub),
            'price_log': price_log_pred,
            'confidence_interval': round(confidence_interval),
            'price_range': {
                'min': round(price_pred_rub - confidence_interval),
                'max': round(price_pred_rub + confidence_interval)
            }
        }
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None


# Пример использования
print("\n" + "=" * 80)
print("ПРИМЕР ПРЕДСКАЗАНИЯ ЦЕНЫ")
print("=" * 80)

# Берем первый автомобиль из тестовой выборки для демонстрации
if len(X_test) > 0:
    sample_car = X_test.iloc[[0]]
    actual_price = np.expm1(y_test.iloc[0])

    prediction = predict_car_price(sample_car)

    if prediction:
        print(f"Предсказанная цена: {prediction['price_rub']:,} руб.")
        print(f"Фактическая цена: {actual_price:,.0f} руб.")
        print(f"Диапазон: {prediction['price_range']['min']:,} - {prediction['price_range']['max']:,} руб.")
        print(f"Ошибка: {abs(prediction['price_rub'] - actual_price):,.0f} руб.")

# Сводная таблица результатов
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)

summary_data = []
for model_name, result in results.items():
    if 'rmse_rub' in result:
        summary_data.append({
            'Model': model_name,
            'RMSE (рубли)': f"{result['rmse_rub']:,.0f}",
            'MAE (рубли)': f"{result.get('mae_rub', 0):,.0f}",
            'R²': f"{result.get('r2_rub', 0):.4f}",
            'MAPE (%)': f"{result.get('mape', 0):.2f}%" if 'mape' in result else "N/A"
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

print(f"\nЛучшая модель: {best_model}")
print(f"RMSE лучшей модели: {best_score:,.0f} рублей")
print(f"Средняя ошибка: {best_score / np.expm1(y_test).mean() * 100:.1f}% от средней цены")

print("\nОбработка данных и обучение модели завершены успешно!")