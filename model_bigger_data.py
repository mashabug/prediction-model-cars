import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
import sys
import subprocess
import json

warnings.filterwarnings('ignore')


# ПРОВЕРКА И УСТАНОВКА LightGBM
def install_package(package_name):
    """Установка пакета если не установлен"""
    try:
        __import__(package_name)
        print(f"✓ {package_name} уже установлен")
        return True
    except ImportError:
        print(f"⏳ Установка {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} успешно установлен")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Не удалось установить {package_name}")
            return False


# Проверяем и устанавливаем необходимые пакеты
CATBOOST_AVAILABLE = install_package('catboost')
LIGHTGBM_AVAILABLE = install_package('lightgbm')

# Импортируем после установки
if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor

if LIGHTGBM_AVAILABLE:
    import lightgbm as lgb

# Загрузка обработанных данных
try:
    df = pd.read_csv('/Users/mariabug/ML/data/combined_ohe.csv')
    print("Данные успешно загружены")
    print(f"Размерность данных: {df.shape}")

except FileNotFoundError:
    print("Файл не найден. Завершение работы, так как нет резервных данных.")
    raise FileNotFoundError("Файл с данными не найден и нет резервных данных")

except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    print("Завершение работы, так как нет резервных данных.")
    raise

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

# УСКОРЕННАЯ ОБРАБОТКА ПРОПУСКОВ
print(f"\nПропущенные значения в признаках: {X.isnull().sum().sum()}")

# Быстрое заполнение пропусков
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())

# Для категориальных - просто заполняем 'unknown'
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    X[col] = X[col].fillna('unknown')

print(f"Пропущенные значения после обработки: {X.isnull().sum().sum()}")

# ПРОВЕРКА ДУБЛИКАТОВ В ВЫБОРКАХ
print("\n" + "=" * 80)
print("ПРОВЕРКА ДУБЛИКАТОВ В ВЫБОРКАХ")
print("=" * 80)


def check_and_remove_duplicates(X_train, X_test, y_train, y_test):
    """Проверка и удаление дубликатов между train и test выборками"""
    print("Проверка дубликатов...")

    # 1. Проверка дубликатов внутри train
    train_duplicates = X_train.duplicated().sum()
    print(f"Дубликатов в train выборке: {train_duplicates}")

    if train_duplicates > 0:
        X_train = X_train.drop_duplicates()
        y_train = y_train[X_train.index]
        print(f"Удалено {train_duplicates} дубликатов из train")

    # 2. Проверка дубликатов внутри test
    test_duplicates = X_test.duplicated().sum()
    print(f"Дубликатов в test выборке: {test_duplicates}")

    if test_duplicates > 0:
        X_test = X_test.drop_duplicates()
        y_test = y_test[X_test.index]
        print(f"Удалено {test_duplicates} дубликатов из test")

    # 3. Проверка дубликатов между train и test
    combined = pd.concat([X_train, X_test])
    between_duplicates = combined.duplicated().sum()
    print(f"Дубликатов между train и test: {between_duplicates}")

    if between_duplicates > 0:
        # Удаляем дубликаты из test, которые есть в train
        train_hashes = pd.util.hash_pandas_object(X_train).values
        test_hashes = pd.util.hash_pandas_object(X_test).values

        train_hash_set = set(train_hashes)
        duplicate_mask = np.array([h in train_hash_set for h in test_hashes])

        X_test = X_test[~duplicate_mask]
        y_test = y_test[~duplicate_mask]
        print(f"Удалено {duplicate_mask.sum()} дубликатов из test (присутствующих в train)")

    print(f"Итоговые размеры: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ОПТИМИЗИРОВАННОЕ РАЗДЕЛЕНИЕ ДАННЫХ
print(f"\nРазмеры данных перед разделением: {X.shape}")

# Используем стратифицированное разделение по квартилям цены
y_quantiles = pd.qcut(y, q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_quantiles
)

print(f"Размеры выборок до обработки дубликатов:")
print(f"Обучающая: {X_train.shape}")
print(f"Тестовая: {X_test.shape}")

# Проверяем и удаляем дубликаты
X_train, X_test, y_train, y_test = check_and_remove_duplicates(X_train, X_test, y_train, y_test)


# Функция для подготовки категориальных признаков
def get_categorical_features(X):
    """Определяет категориальные признаки"""
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    return categorical_features


# ФУНКЦИЯ ДЛЯ РАСЧЕТА R² ПО ЦЕНОВЫМ ДИАПАЗОНАМ
def calculate_price_range_r2(y_true, y_pred, price_ranges=None):
    """
    Вычисляет R² для разных ценовых диапазонов
    """
    if price_ranges is None:
        # Автоматическое определение диапазонов по квантилям
        price_ranges = [0, 500000, 1000000, 2000000, 5000000, np.inf]

    results = {}
    total_samples = len(y_true)
    weighted_r2 = 0

    for i in range(len(price_ranges) - 1):
        low = price_ranges[i]
        high = price_ranges[i + 1]

        # Маска для текущего диапазона
        mask = (y_true >= low) & (y_true < high)

        if np.sum(mask) > 5:  # Минимум 5 samples для расчета R²
            y_true_range = y_true[mask]
            y_pred_range = y_pred[mask]

            if len(np.unique(y_true_range)) > 1:  # Нужна вариативность для R²
                r2 = r2_score(y_true_range, y_pred_range)
            else:
                r2 = np.nan

            n_samples = len(y_true_range)
            weight = n_samples / total_samples

            results[f'{low:,} - {high:,}'] = {
                'r2': r2,
                'samples': n_samples,
                'weight': weight
            }

            if not np.isnan(r2):
                weighted_r2 += r2 * weight
        else:
            results[f'{low:,} - {high:,}'] = {
                'r2': np.nan,
                'samples': np.sum(mask),
                'weight': np.sum(mask) / total_samples
            }

    return {
        'range_r2': results,
        'weighted_r2': weighted_r2
    }


# ФУНКЦИЯ ДЛЯ АНАЛИЗА ПЕРЕОБУЧЕНИЯ
def analyze_overfitting(results):
    """Детальный анализ переобучения для всех моделей"""
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРЕОБУЧЕНИЯ")
    print("=" * 80)

    overfitting_analysis = {}

    for model_name, result in results.items():
        train_r2 = result['train_r2']
        test_r2 = result['test_r2']
        r2_gap = result['r2_gap']

        train_rmse = result['train_rmse']
        test_rmse = result['test_rmse']
        rmse_gap = result['rmse_gap']

        train_mape = result['train_mape']
        test_mape = result['test_mape']
        mape_gap = result['mape_gap']

        # Анализ степени переобучения
        if r2_gap > 0.15:
            overfitting_level = "СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ"
            recommendation = "Увеличить регуляризацию, уменьшить сложность модели"
        elif r2_gap > 0.08:
            overfitting_level = "УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ"
            recommendation = "Добавить регуляризацию, использовать кросс-валидацию"
        elif r2_gap > 0.03:
            overfitting_level = "СЛАБОЕ ПЕРЕОБУЧЕНИЕ"
            recommendation = "Приемлемый уровень, можно попробовать тонкую настройку"
        else:
            overfitting_level = "МИНИМАЛЬНОЕ ПЕРЕОБУЧЕНИЕ"
            recommendation = "Отличное обобщение, модель хорошо настроена"

        overfitting_analysis[model_name] = {
            'overfitting_level': overfitting_level,
            'recommendation': recommendation,
            'r2_gap': r2_gap,
            'rmse_gap': rmse_gap,
            'mape_gap': mape_gap,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        print(f"\n📊 {model_name}:")
        print(f"   Уровень переобучения: {overfitting_level}")
        print(f"   R²: Train={train_r2:.4f}, Test={test_r2:.4f}, Gap={r2_gap:.4f}")
        print(f"   RMSE Gap: {rmse_gap:,.0f} руб")
        print(f"   MAPE Gap: {mape_gap:.2f}%")
        print(f"   Рекомендация: {recommendation}")

    return overfitting_analysis


# ПОЛНАЯ ФУНКЦИЯ ОЦЕНКИ МОДЕЛЕЙ С МЕТРИКАМИ ПО ДИАПАЗОНАМ
def evaluate_model_detailed(model, X_train, X_test, y_train, y_test, model_name, categorical_features=None):
    """Детальная оценка модели с метриками по диапазонам"""
    try:
        # Быстрая обработка данных
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        # Заполняем пропуски
        numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        X_train_processed[numeric_cols] = X_train_processed[numeric_cols].fillna(
            X_train_processed[numeric_cols].median())
        X_test_processed[numeric_cols] = X_test_processed[numeric_cols].fillna(X_train_processed[numeric_cols].median())

        # Обработка категориальных признаков в зависимости от модели
        if model_name in ['LightGBM', 'CatBoost'] and categorical_features:
            if model_name == 'LightGBM':
                # Для LightGBM преобразуем в category
                for col in categorical_features:
                    if col in X_train_processed.columns:
                        X_train_processed[col] = X_train_processed[col].astype('category')
                        X_test_processed[col] = X_test_processed[col].astype('category')
                model.fit(X_train_processed, y_train, categorical_feature=categorical_features)
            else:  # CatBoost
                cat_indices = [i for i, col in enumerate(X_train_processed.columns)
                               if col in categorical_features]
                model.fit(X_train_processed, y_train, cat_features=cat_indices, verbose=False)
        else:
            # Для других моделей используем Label Encoding
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_features:
                if col in X_train_processed.columns:
                    le = LabelEncoder()
                    # Объединяем train и test для кодирования
                    combined = pd.concat([X_train_processed[col], X_test_processed[col]])
                    le.fit(combined)
                    X_train_processed[col] = le.transform(X_train_processed[col])
                    X_test_processed[col] = le.transform(X_test_processed[col])

            X_train_processed = X_train_processed.fillna(0)
            X_test_processed = X_test_processed.fillna(0)
            model.fit(X_train_processed, y_train)

        # Предсказания
        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)

        # Преобразование в рубли
        y_train_rub = np.expm1(y_train)
        y_test_rub = np.expm1(y_test)
        y_pred_train_rub = np.expm1(y_pred_train)
        y_pred_test_rub = np.expm1(y_pred_test)

        # ВСЕ МЕТРИКИ НА ТРЕНИРОВОЧНОЙ ВЫБОРКЕ
        train_mse_rub = mean_squared_error(y_train_rub, y_pred_train_rub)
        train_mae_rub = mean_absolute_error(y_train_rub, y_pred_train_rub)
        train_rmse_rub = np.sqrt(train_mse_rub)
        train_r2_rub = r2_score(y_train_rub, y_pred_train_rub)
        train_mape = np.mean(np.abs((y_train_rub - y_pred_train_rub) / np.maximum(y_train_rub, 1))) * 100

        # ВСЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ
        test_mse_rub = mean_squared_error(y_test_rub, y_pred_test_rub)
        test_mae_rub = mean_absolute_error(y_test_rub, y_pred_test_rub)
        test_rmse_rub = np.sqrt(test_mse_rub)
        test_r2_rub = r2_score(y_test_rub, y_pred_test_rub)
        test_mape = np.mean(np.abs((y_test_rub - y_pred_test_rub) / np.maximum(y_test_rub, 1))) * 100

        # МЕТРИКИ В ЛОГАРИФМИЧЕСКОЙ ШКАЛЕ
        test_mse_log = mean_squared_error(y_test, y_pred_test)
        test_mae_log = mean_absolute_error(y_test, y_pred_test)

        # РАЗНИЦА МЕЖДУ TRAIN И TEST (показатель переобучения)
        r2_gap = train_r2_rub - test_r2_rub
        mae_gap = train_mae_rub - test_mae_rub
        rmse_gap = train_rmse_rub - test_rmse_rub
        mape_gap = train_mape - test_mape

        # R² ПО ЦЕНОВЫМ ДИАПАЗОНАМ
        price_range_metrics = calculate_price_range_r2(y_test_rub, y_pred_test_rub)

        return {
            'model': model,
            # ТРЕНИРОВОЧНЫЕ МЕТРИКИ
            'train_r2': train_r2_rub,
            'train_mae': train_mae_rub,
            'train_rmse': train_rmse_rub,
            'train_mape': train_mape,
            'train_mse': train_mse_rub,
            # ТЕСТОВЫЕ МЕТРИКИ
            'test_r2': test_r2_rub,
            'test_mae': test_mae_rub,
            'test_rmse': test_rmse_rub,
            'test_mape': test_mape,
            'test_mse': test_mse_rub,
            # МЕТРИКИ В ЛОГАРИФМИЧЕСКОЙ ШКАЛЕ
            'mse_log': test_mse_log,
            'mae_log': test_mae_log,
            # РАЗНИЦА (GAP)
            'r2_gap': r2_gap,
            'mae_gap': mae_gap,
            'rmse_gap': rmse_gap,
            'mape_gap': mape_gap,
            # МЕТРИКИ ПО ДИАПАЗОНАМ
            'price_range_r2': price_range_metrics['range_r2'],
            'weighted_range_r2': price_range_metrics['weighted_r2'],
            # ДАННЫЕ ДЛЯ АНАЛИЗА
            'predictions': y_pred_test_rub,
            'y_train_rub': y_train_rub,
            'y_test_rub': y_test_rub,
            'y_pred_train_rub': y_pred_train_rub,
            'y_pred_test_rub': y_pred_test_rub
        }

    except Exception as e:
        print(f"Ошибка при обучении {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# БЫСТРЫЕ МОДЕЛИ ДЛЯ БОЛЬШИХ ДАННЫХ
print("\n" + "=" * 80)
print("БЫСТРЫЕ МОДЕЛИ ДЛЯ БОЛЬШИХ ДАННЫХ")
print("=" * 80)

models = {}

# 1. HistGradientBoosting - самая быстрая
models['HistGradientBoosting'] = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    verbose=0
)

# 2. LightGBM - очень быстрая и эффективная
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

# 3. CatBoost - хорош для категориальных признаков
if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=8,
        random_state=42,
        verbose=False,
        thread_count=-1
    )

# 4. Random Forest с оптимизированными параметрами
models['RandomForest'] = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Получаем категориальные признаки
cat_features = get_categorical_features(X)

# ДЕТАЛЬНОЕ СРАВНЕНИЕ МОДЕЛЕЙ
print("\nДетальное сравнение моделей...")
results = {}
best_model = None
best_score = float('inf')

for name, model in models.items():
    print(f"\nОбучение {name}...")
    start_time = pd.Timestamp.now()

    result = evaluate_model_detailed(model, X_train, X_test, y_train, y_test, name, cat_features)

    if result is not None:
        results[name] = result
        training_time = (pd.Timestamp.now() - start_time).total_seconds()

        print(f"✓ {name} обучена за {training_time:.1f} сек")
        print(f"  R²: Train = {result['train_r2']:.4f}, Test = {result['test_r2']:.4f}, Gap = {result['r2_gap']:.4f}")
        print(f"  RMSE: {result['test_rmse']:,.0f} руб, MAE: {result['test_mae']:,.0f} руб")
        print(f"  MAPE: {result['test_mape']:.2f}%")
        print(f"  Средневзвешенный R² по диапазонам: {result['weighted_range_r2']:.4f}")

        if result['test_rmse'] < best_score:
            best_score = result['test_rmse']
            best_model = name
    else:
        print(f"✗ {name} не удалось обучить")

if not results:
    raise ValueError("Ни одна модель не была успешно обучена")

print(f"\n🎯 Лучшая модель: {best_model} с RMSE {best_score:,.0f} рублей")


# БЫСТРАЯ НАСТРОЙКА ГИПЕРПАРАМЕТРОВ ДЛЯ ЛУЧШЕЙ МОДЕЛИ
def fast_hyperparameter_tuning(best_model_name, X_train, y_train, X_test, y_test, categorical_features):
    """Быстрая настройка гиперпараметров"""
    print(f"\n⚡ Быстрая настройка гиперпараметров для {best_model_name}...")

    if best_model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 63, 127],
            'max_depth': [8, 10, 12]
        }

        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=10, cv=2,
            scoring='neg_mean_squared_error', n_jobs=-1,
            random_state=42, verbose=1
        )

        X_processed = X_train.copy()
        if categorical_features:
            for col in categorical_features:
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].astype('category')

        random_search.fit(X_processed, y_train)
        return random_search.best_estimator_, random_search.best_params_

    elif best_model_name == 'CatBoost' and CATBOOST_AVAILABLE:
        param_dist = {
            'iterations': [200, 300, 400],
            'learning_rate': [0.05, 0.1, 0.15],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [1, 3, 5]
        }

        model = CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)

        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=8, cv=2,
            scoring='neg_mean_squared_error', n_jobs=-1,
            random_state=42, verbose=1
        )

        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_

    elif best_model_name == 'HistGradientBoosting':
        param_dist = {
            'max_iter': [200, 300, 400],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [8, 10, 12],
            'min_samples_leaf': [10, 20, 30]
        }

        model = HistGradientBoostingRegressor(random_state=42, verbose=0)

        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=8, cv=2,
            scoring='neg_mean_squared_error', n_jobs=-1,
            random_state=42, verbose=1
        )

        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_

    else:  # RandomForest
        param_dist = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [15, 20, 25],
            'max_features': [0.4, 0.5, 0.6]
        }

        model = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)

        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=8, cv=2,
            scoring='neg_mean_squared_error', n_jobs=-1,
            random_state=42, verbose=1
        )

        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_


# Настраиваем лучшую модель
best_params = None
try:
    tuned_model, best_params = fast_hyperparameter_tuning(
        best_model, X_train, y_train, X_test, y_test, cat_features
    )

    print(f"🎯 Лучшие параметры для {best_model}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Оцениваем настроенную модель
    print(f"\nОценка настроенной модели {best_model}...")
    tuned_result = evaluate_model_detailed(
        tuned_model, X_train, X_test, y_train, y_test, best_model, cat_features
    )

    if tuned_result:
        results[f'{best_model} (tuned)'] = tuned_result
        tuned_result['best_params'] = best_params  # Сохраняем параметры в результат
        print(f"✅ Настроенная модель улучшила RMSE: {best_score:,.0f} -> {tuned_result['test_rmse']:,.0f} рублей")

except Exception as e:
    print(f"⚠️ Настройка гиперпараметров не удалась: {e}")
    print("Используем базовую модель")


# СОХРАНЕНИЕ ГИПЕРПАРАМЕТРОВ В ФАЙЛ
def save_hyperparameters(best_params, model_name, file_path):
    """Сохранение гиперпараметров в JSON файл"""
    if best_params:
        hyperparameters_data = {
            'model_name': model_name,
            'hyperparameters': best_params,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(hyperparameters_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Гиперпараметры сохранены в: {file_path}")
    else:
        print("⚠️ Нет гиперпараметров для сохранения")


# АНАЛИЗ ПЕРЕОБУЧЕНИЯ
overfitting_analysis = analyze_overfitting(results)


# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
def plot_detailed_results(results):
    """Детальная визуализация результатов"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Сравнение основных метрик
    metrics_data = {}
    for model_name, result in results.items():
        metrics_data[model_name] = {
            'RMSE': result['test_rmse'],
            'MAE': result['test_mae'],
            'R²': result['test_r2']
        }

    metrics_df = pd.DataFrame(metrics_data).T

    # RMSE и MAE
    metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Сравнение RMSE и MAE (рубли)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # R²
    metrics_df[['R²']].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Сравнение R²')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 2. R² по ценовым диапазонам для лучшей модели
    best_model_name = min(results.items(), key=lambda x: x[1]['test_rmse'])[0]
    best_result = results[best_model_name]

    range_data = []
    for range_name, range_metrics in best_result['price_range_r2'].items():
        if not np.isnan(range_metrics['r2']):
            range_data.append({
                'Диапазон': range_name,
                'R²': range_metrics['r2'],
                'Количество образцов': range_metrics['samples']
            })

    if range_data:
        range_df = pd.DataFrame(range_data)
        axes[0, 2].bar(range_df['Диапазон'], range_df['R²'])
        axes[0, 2].set_title(f'R² по ценовым диапазонам ({best_model_name})')
        axes[0, 2].tick_params(axis='x', rotation=45)

        for i, v in enumerate(range_df['R²']):
            axes[0, 2].text(i, v + 0.01, f"n={range_df['Количество образцов'].iloc[i]}",
                            ha='center', va='bottom', fontsize=8)

    # 3. Предсказания vs Фактические значения
    best_predictions = best_result['predictions']
    y_test_rub = best_result['y_test_rub']

    axes[1, 0].scatter(y_test_rub, best_predictions, alpha=0.5)
    axes[1, 0].plot([y_test_rub.min(), y_test_rub.max()],
                    [y_test_rub.min(), y_test_rub.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Фактическая цена')
    axes[1, 0].set_ylabel('Предсказанная цена')
    axes[1, 0].set_title(f'Предсказания vs Фактические значения ({best_model_name})')

    # 4. Ошибки предсказания
    errors = best_predictions - y_test_rub
    axes[1, 1].hist(errors, bins=50, alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Ошибка предсказания (рубли)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title('Распределение ошибок предсказания')

    # 5. Сравнение Train/Test R²
    model_names = list(results.keys())
    train_r2 = [results[name]['train_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[1, 2].bar(x - width / 2, train_r2, width, label='Train R²', alpha=0.7)
    axes[1, 2].bar(x + width / 2, test_r2, width, label='Test R²', alpha=0.7)
    axes[1, 2].set_xlabel('Модели')
    axes[1, 2].set_ylabel('R²')
    axes[1, 2].set_title('Сравнение R² на Train и Test выборках')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(model_names, rotation=45)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


# Строим детальные графики
plot_detailed_results(results)

# СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ И ГИПЕРПАРАМЕТРОВ
final_model_name = best_model
final_model = results[final_model_name]['model']
final_best_params = None

if f'{best_model} (tuned)' in results:
    final_model_name = f'{best_model} (tuned)'
    final_model = results[final_model_name]['model']
    final_best_params = results[final_model_name].get('best_params')

# Сохраняем модель
model_path = '/Users/mariabug/ML/data/detailed_car_price_predictor.pkl'
joblib.dump(final_model, model_path)
print(f"\n💾 Модель сохранена как '{model_path}'")

# Сохраняем гиперпараметры
if final_best_params:
    hyperparams_path = '/Users/mariabug/ML/data/best_hyperparameters.json'
    save_hyperparameters(final_best_params, final_model_name, hyperparams_path)

# ДЕТАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ
print("\n" + "=" * 80)
print("ДЕТАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 80)

# Сводная таблица по моделям
summary_data = []
for model_name, result in results.items():
    summary_data.append({
        'Model': model_name,
        'Train R²': f"{result['train_r2']:.4f}",
        'Test R²': f"{result['test_r2']:.4f}",
        'R² Gap': f"{result['r2_gap']:.4f}",
        'Test RMSE': f"{result['test_rmse']:,.0f}",
        'Test MAE': f"{result['test_mae']:,.0f}",
        'Test MAPE': f"{result['test_mape']:.2f}%",
        'Weighted R²': f"{result['weighted_range_r2']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n📊 СРАВНЕНИЕ МОДЕЛЕЙ:")
print(summary_df.to_string(index=False))

# Детальный анализ лучшей модели
print(f"\n🎯 ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ: {final_model_name}")
best_result = results[final_model_name]

print(f"\n📈 ОСНОВНЫЕ МЕТРИКИ:")
print(f"   R²: Train = {best_result['train_r2']:.4f}, Test = {best_result['test_r2']:.4f}")
print(f"   RMSE: {best_result['test_rmse']:,.0f} руб")
print(f"   MAE: {best_result['test_mae']:,.0f} руб")
print(f"   MAPE: {best_result['test_mape']:.2f}%")
print(f"   Средневзвешенный R² по диапазонам: {best_result['weighted_range_r2']:.4f}")

# Анализ переобучения для лучшей модели
if final_model_name in overfitting_analysis:
    analysis = overfitting_analysis[final_model_name]
    print(f"\n⚠️  АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
    print(f"   Уровень: {analysis['overfitting_level']}")
    print(f"   R² Gap: {analysis['r2_gap']:.4f}")
    print(f"   Рекомендация: {analysis['recommendation']}")

# R² по ценовым диапазонам
print(f"\n🎯 R² ПО ЦЕНОВЫМ ДИАПАЗОНАМ:")
print("-" * 70)
for range_name, range_metrics in best_result['price_range_r2'].items():
    r2_value = range_metrics['r2']
    samples = range_metrics['samples']
    weight = range_metrics['weight'] * 100

    if not np.isnan(r2_value):
        print(f"   {range_name:25} | R² = {r2_value:7.4f} | Образцов = {samples:4d} | Доля = {weight:5.1f}%")
    else:
        print(f"   {range_name:25} | R² = {'N/A':7} | Образцов = {samples:4d} | Доля = {weight:5.1f}%")

# Вывод информации о дубликатах
print(f"\n📊 ИНФОРМАЦИЯ О ДАННЫХ:")
print(f"   Исходный размер данных: {df.shape}")
print(f"   Обучающая выборка: {X_train.shape}")
print(f"   Тестовая выборка: {X_test.shape}")

print(f"\n⚡ Обучение завершено успешно!")
print(f"📁 Модель сохранена: {model_path}")
if final_best_params:
    print(f"📁 Гиперпараметры сохранены: {hyperparams_path}")
