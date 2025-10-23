import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import json

# Загрузка данных
df = pd.read_csv("/Users/mariabug/ML/data/combined_cleaned.csv")  # укажите путь к файлу

# 1. Предварительный анализ данных
print("Информация о данных:")
print(df.info())
print("\nПервые 5 строк:")
print(df.head())
print("\nПропущенные значения:")
print(df.isnull().sum())

# УДАЛЕНИЕ КОЛОНОК, ВЫЗЫВАЮЩИХ УТЕЧКУ ДАННЫХ
leakage_columns = ['price_low', 'price_high', 'price_diff_last_update', 'soft_key']
df = df.drop(columns=[col for col in leakage_columns if col in df.columns], axis=1)

# 2. Обработка seller_id - Frequency Encoding
if 'seller_id' in df.columns:
    seller_freq = df['seller_id'].value_counts().to_dict()
    df['seller_freq'] = df['seller_id'].map(seller_freq)
    # Удаляем исходный seller_id
    df = df.drop('seller_id', axis=1)

# 3. Обработка offer_created
if 'offer_created' in df.columns:
    # Преобразование в datetime
    df['offer_created'] = pd.to_datetime(df['offer_created'], errors='coerce')

    # Извлечение признаков из даты
    df['offer_year'] = df['offer_created'].dt.year
    df['offer_month'] = df['offer_created'].dt.month
    df['offer_day'] = df['offer_created'].dt.day
    df['offer_dayofweek'] = df['offer_created'].dt.dayofweek
    df['offer_weekofyear'] = df['offer_created'].dt.isocalendar().week
    df['offer_quarter'] = df['offer_created'].dt.quarter

    # Удаляем исходную колонку
    df = df.drop('offer_created', axis=1)

# 4. Удаление last_update и date_closed
columns_to_drop_immediately = ['last_update', 'date_closed']
df = df.drop(columns=[col for col in columns_to_drop_immediately if col in df.columns], axis=1)


# 5. Обработка географических и категориальных колонок
def process_categorical_columns(df):
    """Обработка категориальных колонок"""
    df_copy = df.copy()

    # Обработка города - One-Hot Encoding для городов с малым числом уникальных значений
    if 'city' in df_copy.columns:
        df_copy['city'] = df_copy['city'].astype(str).str.lower().str.strip().fillna('unknown')
        # Если уникальных значений немного, используем One-Hot, иначе Frequency Encoding
        if df_copy['city'].nunique() <= 20:
            # One-Hot Encoding для города
            ohe_city = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            city_encoded = ohe_city.fit_transform(df_copy[['city']])
            city_encoded_df = pd.DataFrame(city_encoded,
                                           columns=ohe_city.get_feature_names_out(['city']))
            df_copy = pd.concat([df_copy.drop('city', axis=1), city_encoded_df], axis=1)
        else:
            # Frequency Encoding для города
            city_freq = df_copy['city'].value_counts().to_dict()
            df_copy['city_freq'] = df_copy['city'].map(city_freq)
            df_copy = df_copy.drop('city', axis=1)

    # Обработка региона - Frequency Encoding (обычно много уникальных значений)
    if 'region' in df_copy.columns:
        df_copy['region'] = df_copy['region'].astype(str).str.lower().str.strip().fillna('unknown')
        region_freq = df_copy['region'].value_counts().to_dict()
        df_copy['region_freq'] = df_copy['region'].map(region_freq)
        df_copy = df_copy.drop('region', axis=1)

    # Обработка типа продавца - One-Hot Encoding (обычно мало уникальных значений)
    if 'seller_type' in df_copy.columns:
        df_copy['seller_type'] = df_copy['seller_type'].astype(str).str.lower().str.strip().fillna('unknown')
        ohe_seller = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        seller_encoded = ohe_seller.fit_transform(df_copy[['seller_type']])
        seller_encoded_df = pd.DataFrame(seller_encoded,
                                         columns=ohe_seller.get_feature_names_out(['seller_type']))
        df_copy = pd.concat([df_copy.drop('seller_type', axis=1), seller_encoded_df], axis=1)

    # Обработка продавца - Frequency Encoding
    if 'seller' in df_copy.columns:
        df_copy['seller'] = df_copy['seller'].astype(str).str.lower().str.strip().fillna('unknown')
        seller_freq = df_copy['seller'].value_counts().to_dict()
        df_copy['seller_freq'] = df_copy['seller'].map(seller_freq)
        df_copy = df_copy.drop('seller', axis=1)

    return df_copy


# Применяем обработку категориальных колонок
df = process_categorical_columns(df)


# 6. Создание бинарных признаков
def create_binary_features(df):
    """Создание бинарных признаков"""
    df_copy = df.copy()

    # Бинарный признак для таможенной очистки на основе custom
    if 'custom' in df_copy.columns:
        df_copy['is_cleared'] = df_copy['custom'].apply(
            lambda x: 1 if str(x).lower() in ['растоможен', 'растаможен', 'cleared', 'да', 'yes', 'true'] else 0
        )
    else:
        # Если колонки custom нет, используем альтернативный метод
        df_copy['is_cleared'] = 0  # или другую логику по умолчанию

    # Бинарный признак для автомобилей с низким пробегом
    if 'km_age' in df_copy.columns:
        df_copy['is_low_mileage'] = (df_copy['km_age'] < 50000).astype(int)

    # Бинарный признак для новых автомобилей (моложе 3 лет)
    current_year = pd.Timestamp.now().year
    if 'year' in df_copy.columns:
        df_copy['is_recent'] = (df_copy['year'] >= (current_year - 3)).astype(int)

    # Бинарный признак для премиальных марок
    premium_brands = ['BMW', 'Mercedes', 'Audi', 'Porsche', 'Lexus', 'Volvo']
    if 'mark' in df_copy.columns:
        df_copy['is_premium'] = df_copy['mark'].isin(premium_brands).astype(int)

    # Бинарный признак для дизельных двигателей
    if 'engine_type' in df_copy.columns:
        df_copy['is_diesel'] = df_copy['engine_type'].apply(
            lambda x: 1 if 'дизель' in str(x).lower() else 0
        )

    # Бинарный признак для автомата
    if 'transmission' in df_copy.columns:
        df_copy['is_automatic'] = df_copy['transmission'].apply(
            lambda x: 1 if 'автомат' in str(x).lower() else 0
        )

    # Бинарный признак для полного привода
    if 'drive_type' in df_copy.columns:
        df_copy['is_4wd'] = df_copy['drive_type'].apply(
            lambda x: 1 if 'полный' in str(x).lower() else 0
        )

    # Бинарный признак для количества фото (есть ли фото)
    if 'image_urls_count' in df_copy.columns:
        df_copy['has_images'] = (df_copy['image_urls_count'] > 0).astype(int)

    # Бинарный признак для автомобилей в отличном состоянии
    if 'condition' in df_copy.columns:
        df_copy['is_excellent_condition'] = df_copy['condition'].apply(
            lambda x: 1 if str(x).lower() in ['отличное', 'excellent'] else 0
        )

    # Бинарный признак для наличия VIN
    if 'vin' in df_copy.columns:
        df_copy['has_vin'] = (df_copy['vin'].notna() & (df_copy['vin'] != '')).astype(int)

    # Бинарный признак для наличия PTS информации
    if 'has_pts_info' in df_copy.columns:
        df_copy['has_pts_info'] = df_copy['has_pts_info'].fillna(0).astype(int)

    return df_copy


df = create_binary_features(df)


# 7. Обработка целевой переменной price_rub (БЕЗ создания признаков на основе цены)
def prepare_target_variable(df):
    """Подготовка целевой переменной БЕЗ создания признаков на основе цены"""
    df_copy = df.copy()

    # Только логарифмирование цены для нормализации распределения
    df_copy['price_log'] = np.log1p(df_copy['price_rub'])

    # Только вычисление возраста автомобиля (не на основе цены)
    current_year = pd.Timestamp.now().year
    if 'year' in df_copy.columns:
        df_copy['car_age'] = current_year - df_copy['year']

    return df_copy


df = prepare_target_variable(df)

# 8. Обработка числовых столбцов (БЕЗ признаков на основе цены)
numeric_columns = [
    'displacement', 'horse_power', 'km_age', 'year',
    'image_urls_count', 'car_age', 'seller_freq',
    'offer_year', 'offer_month', 'offer_day', 'offer_dayofweek',
    'offer_weekofyear', 'offer_quarter', 'update_count'
]

# Обработка колонки owners_count если она существует
if 'owners_count' in df.columns:
    # Преобразуем owners_count в строку и обрабатываем значения типа "4+"
    df['owners_count'] = df['owners_count'].astype(str)

    # Заменяем "4+" на "4" и другие подобные значения
    df['owners_count'] = df['owners_count'].str.replace(r'(\d+)\+', r'\1', regex=True)

    # Удаляем все нечисловые символы, оставляем только цифры
    df['owners_count'] = df['owners_count'].str.extract('(\d+)', expand=False)

    # Преобразуем в числовой тип
    df['owners_count'] = pd.to_numeric(df['owners_count'], errors='coerce')

    # Добавляем owners_count в список числовых колонок для дальнейшей обработки
    numeric_columns.append('owners_count')

for col in numeric_columns:
    if col in df.columns:
        # Замена пропусков медианным значением
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Обработка выбросов (cap при 99 перцентиле)
        upper_limit = df[col].quantile(0.99)
        lower_limit = df[col].quantile(0.01)
        df[col] = np.clip(df[col], lower_limit, upper_limit)

        # Заполнение пропусков
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# РАСШИРЕННЫЙ FEATURE ENGINEERING ДЛЯ ВСЕХ ЦЕНОВЫХ КАТЕГОРИЙ
print("\n" + "=" * 80)
print("РАСШИРЕННЫЙ FEATURE ENGINEERING ДЛЯ ВСЕХ ЦЕНОВЫХ КАТЕГОРИЙ")
print("=" * 80)


def advanced_feature_engineering(df):
    """Создание расширенных признаков для всех ценовых категорий"""
    df_copy = df.copy()

    # 1. СЕГМЕНТАЦИЯ АВТОМОБИЛЕЙ ПО ХАРАКТЕРИСТИКАМ (БЕЗ ИСПОЛЬЗОВАНИЯ ЦЕНЫ)
    # Сегментация по марке
    premium_brands = ['bmw', 'mercedes', 'audi', 'porsche', 'lexus', 'volvo', 'jaguar', 'land rover']
    mid_brands = ['volkswagen', 'toyota', 'honda', 'ford', 'hyundai', 'kia', 'nissan', 'skoda', 'mazda']

    if 'mark' in df_copy.columns:
        df_copy['brand_segment'] = df_copy['mark'].apply(
            lambda x: 'premium' if str(x).lower() in premium_brands else
            'mid' if str(x).lower() in mid_brands else 'economy'
        )

    # Сегментация по возрасту и пробегу
    if 'car_age' in df_copy.columns and 'km_age' in df_copy.columns:
        conditions = [
            (df_copy['car_age'] <= 3) & (df_copy['km_age'] <= 50000),
            (df_copy['car_age'] <= 5) & (df_copy['km_age'] <= 100000),
            (df_copy['car_age'] > 10) | (df_copy['km_age'] > 200000)
        ]
        choices = ['new_low_mileage', 'mid_age_mileage', 'old_high_mileage']
        df_copy['age_mileage_segment'] = np.select(conditions, choices, default='average')

    # 2. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ ДЛЯ РАЗНЫХ СЕГМЕНТОВ
    # Для бюджетных автомобилей
    if 'car_age' in df_copy.columns and 'km_age' in df_copy.columns:
        df_copy['budget_car_ratio'] = (df_copy['car_age'] * df_copy['km_age']) / 1000000

    if 'horse_power' in df_copy.columns and 'car_age' in df_copy.columns:
        df_copy['economy_power_ratio'] = df_copy['horse_power'] / (df_copy['car_age'] + 1)

    # Для премиальных автомобилей
    if 'horse_power' in df_copy.columns and 'displacement' in df_copy.columns:
        df_copy['premium_power_density'] = df_copy['horse_power'] / (df_copy['displacement'] + 1)

    if all(col in df_copy.columns for col in ['is_automatic', 'is_4wd', 'horse_power', 'displacement']):
        df_copy['luxury_tech_score'] = (
                df_copy['is_automatic'] +
                df_copy['is_4wd'] +
                (df_copy['horse_power'] > 200).astype(int) +
                (df_copy['displacement'] > 2.0).astype(int)
        )

    # 3. ПРИЗНАКИ ДЛЯ СРЕДНЕГО ЦЕНОВОГО СЕГМЕНТА
    if all(col in df_copy.columns for col in ['displacement', 'horse_power', 'car_age']):
        df_copy['family_car_score'] = (
                (df_copy['displacement'] <= 2.5).astype(int) +
                (df_copy['horse_power'] >= 100).astype(int) +
                (df_copy['horse_power'] <= 200).astype(int) +
                (df_copy['car_age'] <= 8).astype(int)
        )

    # 4. ЛОГАРИФМИРОВАНИЕ И СТАНДАРТИЗАЦИЯ КЛЮЧЕВЫХ ПРИЗНАКОВ
    skewed_columns = ['horse_power', 'km_age', 'displacement', 'image_urls_count']
    for col in skewed_columns:
        if col in df_copy.columns:
            df_copy[f'{col}_log'] = np.log1p(df_copy[col])

    # 5. БИННИНГ ЧИСЛОВЫХ ПРИЗНАКОВ ДЛЯ НЕЛИНЕЙНЫХ ЗАВИСИМОСТЕЙ
    numeric_for_binning = ['horse_power', 'displacement', 'car_age', 'km_age']

    for col in numeric_for_binning:
        if col in df_copy.columns:
            # Равномерное бининг
            try:
                df_copy[f'{col}_bins_5'] = pd.cut(df_copy[col], bins=5, labels=False, duplicates='drop')
                df_copy[f'{col}_bins_10'] = pd.cut(df_copy[col], bins=10, labels=False, duplicates='drop')
            except:
                pass

            # Квантильное бининг
            try:
                df_copy[f'{col}_q_bins_5'] = pd.qcut(df_copy[col], q=5, labels=False, duplicates='drop')
            except:
                pass

    # 6. КЛАСТЕРИЗАЦИЯ АВТОМОБИЛЕЙ ПО ТЕХНИЧЕСКИМ ХАРАКТЕРИСТИКАМ
    cluster_features = ['horse_power', 'displacement', 'car_age', 'km_age']
    if all(feat in df_copy.columns for feat in cluster_features):
        try:
            # Нормализуем признаки для кластеризации
            cluster_data = df_copy[cluster_features].fillna(0)
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)

            # Кластеризация K-means
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df_copy['tech_cluster'] = kmeans.fit_predict(cluster_data_scaled)

            # PCA для уменьшения размерности
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(cluster_data_scaled)
            df_copy['tech_pca_1'] = pca_result[:, 0]
            df_copy['tech_pca_2'] = pca_result[:, 1]
        except Exception as e:
            print(f"Кластеризация не удалась: {e}")

    # 7. ВРЕМЕННЫЕ ПРИЗНАКИ И СЕЗОННОСТЬ
    if 'offer_year' in df_copy.columns and 'offer_month' in df_copy.columns:
        df_copy['is_year_end'] = ((df_copy['offer_month'] == 12) | (df_copy['offer_month'] <= 2)).astype(int)
        df_copy['is_spring'] = ((df_copy['offer_month'] >= 3) & (df_copy['offer_month'] <= 5)).astype(int)
        df_copy['is_summer'] = ((df_copy['offer_month'] >= 6) & (df_copy['offer_month'] <= 8)).astype(int)

        # Признак для новых модельных годов
        if 'year' in df_copy.columns:
            df_copy['is_new_model_year'] = (
                    (df_copy['offer_month'] >= 9) &
                    (df_copy['year'] == df_copy['offer_year'])
            ).astype(int)

    # 8. ГЕОГРАФИЧЕСКИЕ ПРИЗНАКИ (если есть данные)
    if 'city_freq' in df_copy.columns:
        df_copy['is_large_city'] = (df_copy['city_freq'] > df_copy['city_freq'].median()).astype(int)

    # 9. ПРИЗНАКИ ДЛЯ РЕДКИХ И ЭКСКЛЮЗИВНЫХ АВТОМОБИЛЕЙ
    if 'mark' in df_copy.columns:
        mark_counts = df_copy['mark'].value_counts()
        rare_marks = mark_counts[mark_counts < len(df_copy) * 0.01].index
        df_copy['is_rare_mark'] = df_copy['mark'].isin(rare_marks).astype(int)

    # 10. КОМПОЗИТНЫЕ ИНДЕКСЫ
    if all(col in df_copy.columns for col in ['horse_power', 'car_age', 'km_age', 'is_automatic', 'is_4wd']):
        df_copy['value_index'] = (
                df_copy['horse_power'] * 0.3 +
                (1 / (df_copy['car_age'] + 1)) * 0.3 +
                (1 / (df_copy['km_age'] / 10000 + 1)) * 0.2 +
                df_copy['is_automatic'] * 0.1 +
                df_copy['is_4wd'] * 0.1
        )

    if all(col in df_copy.columns for col in ['car_age', 'km_age', 'is_automatic', 'is_diesel']):
        df_copy['maintenance_index'] = (
                df_copy['car_age'] * 0.4 +
                df_copy['km_age'] / 10000 * 0.4 +
                (1 - df_copy['is_automatic']) * 0.1 +
                df_copy['is_diesel'] * 0.1
        )

    new_features_count = len([col for col in df_copy.columns if col not in df.columns])
    print(f"Создано {new_features_count} дополнительных признаков")
    print(f"Общее количество признаков: {len(df_copy.columns)}")

    return df_copy


# Применяем расширенный feature engineering
df = advanced_feature_engineering(df)


# 9. Обработка опций (options_* колонки)
def process_options_columns(df):
    """Обработка колонок с опциями"""
    df_copy = df.copy()

    # Находим все колонки, начинающиеся с 'options_'
    options_columns = [col for col in df_copy.columns if col.startswith('options_')]

    for col in options_columns:
        if col in df_copy.columns:
            # Заполняем пропуски
            df_copy[col] = df_copy[col].fillna(0)

            # Преобразуем в числовой тип, если это строка
            if df_copy[col].dtype == 'object':
                # Пробуем преобразовать в число
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                # Заполняем пропуски после преобразования
                df_copy[col] = df_copy[col].fillna(0)

            # Создаем бинарный признак наличия опции
            df_copy[f'{col}_exists'] = (df_copy[col] > 0).astype(int)

            # Для числовых значений создаем также нормализованную версию
            if df_copy[col].dtype in [np.int64, np.float64]:
                max_val = df_copy[col].max()
                if max_val > 0:
                    df_copy[f'{col}_norm'] = df_copy[col] / max_val

    return df_copy


df = process_options_columns(df)


# 10. Обработка history колонок
def process_history_columns(df):
    """Обработка history колонок"""
    df_copy = df.copy()

    # Находим все колонки, начинающиеся с 'history_'
    history_columns = [col for col in df_copy.columns if col.startswith('history_')]

    for col in history_columns:
        if col in df_copy.columns:
            # Заполняем пропуски
            df_copy[col] = df_copy[col].fillna(0)

            # Преобразуем в числовой тип, если это строка
            if df_copy[col].dtype == 'object':
                # Пробуем преобразовать в число
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                # Заполняем пропуски после преобразования
                df_copy[col] = df_copy[col].fillna(0)

            # Создаем бинарный признак наличия истории
            df_copy[f'{col}_exists'] = (df_copy[col] > 0).astype(int)

            # Для числовых значений создаем также нормализованную версию
            if df_copy[col].dtype in [np.int64, np.float64]:
                max_val = df_copy[col].max()
                if max_val > 0:
                    df_copy[f'{col}_norm'] = df_copy[col] / max_val

    return df_copy


df = process_history_columns(df)

# 11. Обработка остальных категориальных признаков
categorical_columns = [
    'mark', 'model', 'body_type', 'color', 'drive_type',
    'engine_type', 'transmission', 'wheel', 'condition',
    'brand_segment', 'age_mileage_segment', 'pts'  # Добавляем новые категориальные признаки
]

# One-Hot Encoding для категориальных признаков с малым числом уникальных значений
low_cardinality_cols = [col for col in categorical_columns
                        if col in df.columns and df[col].nunique() <= 20]

# Frequency Encoding для категориальных признаков с большим числом уникальных значений
high_cardinality_cols = [col for col in categorical_columns
                         if col in df.columns and df[col].nunique() > 20]

df_encoded = df.copy()

# One-Hot Encoding
for col in low_cardinality_cols:
    if col in df_encoded.columns:
        # Заполняем пропуски
        df_encoded[col] = df_encoded[col].fillna('unknown')

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = ohe.fit_transform(df_encoded[[col]])
        encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out([col]))
        df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)

# Frequency Encoding для высококардинальных признаков
for col in high_cardinality_cols:
    if col in df_encoded.columns:
        # Заполняем пропуски
        df_encoded[col] = df_encoded[col].fillna('unknown')
        freq_encoding = df_encoded[col].value_counts().to_dict()
        df_encoded[f'{col}_freq'] = df_encoded[col].map(freq_encoding)
        df_encoded = df_encoded.drop(col, axis=1)

# 12. Обработка текстовых колонок (только извлечение признаков)
text_columns = ['generation', 'complectation', 'configuration']

for col in text_columns:
    if col in df_encoded.columns:
        # Очистка текста
        df_encoded[col] = df_encoded[col].astype(str).str.lower().str.strip()

        # Создание признаков длины текста
        df_encoded[f'{col}_length'] = df_encoded[col].str.len()

        # Извлечение числовых значений из текста
        if 'configuration' in col:
            extracted_vol = df_encoded[col].str.extract('(\d+\.\d+)')[0].astype(float)
            df_encoded['extracted_engine_volume'] = extracted_vol

        # Удаление исходного текстового столбца
        df_encoded = df_encoded.drop(col, axis=1)


# 13. Обработка колонки image_urls
if 'image_urls' in df_encoded.columns:
    # Создаем признак количества изображений (если image_urls_count нет)
    if 'image_urls_count' not in df_encoded.columns:
        df_encoded['image_urls_count'] = df_encoded['image_urls'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
    df_encoded = df_encoded.drop('image_urls', axis=1)

# 14. Удаление ненужных колонок (включая те, что указаны)
columns_to_drop = [
    # Идентификаторы и URL
    'inner_id', 'id', 'url', 'image_urls_first',

    # Удаляем указанные колонки
    'description', 'address', 'seller_url', 'vin',

    # Исходные колонки для целевой переменной
    'price_rub',

    # Исходные колонки, использованные для создания бинарных признаков
    'mark', 'km_age', 'year', 'engine_type', 'transmission',
    'drive_type', 'condition', 'image_urls_count', 'custom',

    # Исходные категориальные колонки
    'body_type', 'color', 'wheel', 'pts'
]

# Удаляем только те колонки, которые существуют в DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
df_encoded = df_encoded.drop(existing_columns_to_drop, axis=1)

# 15. Проверяем, что целевая переменная существует
if 'price_log' not in df_encoded.columns:
    df_encoded['price_log'] = np.log1p(df['price_rub'])

# 16. ФИНАЛЬНАЯ ОБРАБОТКА ДЛЯ УЛУЧШЕНИЯ ПРЕДСКАЗАНИЙ
print("\n" + "=" * 80)
print("ФИНАЛЬНАЯ ОБРАБОТКА ДЛЯ УЛУЧШЕНИЯ ПРЕДСКАЗАНИЙ")
print("=" * 80)

# Проверяем наличие числовых колонок перед нормализацией
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
if 'price_log' in numeric_cols:
    numeric_cols.remove('price_log')

# Более аккуратная обработка выбросов для разных сегментов
for col in numeric_cols:
    if col in df_encoded.columns and col != 'price_log':
        # Используем робастные методы для обработки выбросов
        Q1 = df_encoded[col].quantile(0.05)  # Более мягкие границы
        Q3 = df_encoded[col].quantile(0.95)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Применяем winsorization вместо полного обрезания
        df_encoded[col] = np.where(df_encoded[col] < lower_bound, lower_bound, df_encoded[col])
        df_encoded[col] = np.where(df_encoded[col] > upper_bound, upper_bound, df_encoded[col])

        # Заполняем пропуски медианой
        median_val = df_encoded[col].median()
        df_encoded[col] = df_encoded[col].fillna(median_val)

# Разделение на признаки и целевую переменную
if 'price_log' in df_encoded.columns:
    X = df_encoded.drop('price_log', axis=1)
    y = df_encoded['price_log']
else:
    raise ValueError("Целевая переменная 'price_log' не найдена в данных")

# Нормализация числовых признаков
if numeric_cols:
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"\nПризнаки для модели: {X.shape[1]}")
print(f"Целевая переменная: {y.name}")

# Сохранение обработанных данных
final_df = X.copy()
final_df['price_log'] = y
final_df.to_csv('/Users/mariabug/ML/data/combined_ohe.csv', index=False)


# 17. Создание функции для обработки новых данных
def preprocess_new_data(new_df, scaler=None, kmeans_model=None, pca_model=None):
    """
    Функция для обработки новых данных с учетом расширенных признаков
    """
    df_copy = new_df.copy()

    # УДАЛЕНИЕ КОЛОНОК, ВЫЗЫВАЮЩИХ УТЕЧКУ ДАННЫХ В НОВЫХ ДАННЫХ
    leakage_columns = ['price_low', 'price_high', 'price_diff_last_update', 'price_diff_since_posted']
    df_copy = df_copy.drop(columns=[col for col in leakage_columns if col in df_copy.columns], axis=1)

    # Обработка seller_id - Frequency Encoding
    if 'seller_id' in df_copy.columns:
        seller_freq = df_copy['seller_id'].value_counts().to_dict()
        df_copy['seller_freq'] = df_copy['seller_id'].map(seller_freq)
        df_copy = df_copy.drop('seller_id', axis=1)

    # Обработка offer_created
    if 'offer_created' in df_copy.columns:
        df_copy['offer_created'] = pd.to_datetime(df_copy['offer_created'], errors='coerce')
        df_copy['offer_year'] = df_copy['offer_created'].dt.year
        df_copy['offer_month'] = df_copy['offer_created'].dt.month
        df_copy['offer_day'] = df_copy['offer_created'].dt.day
        df_copy['offer_dayofweek'] = df_copy['offer_created'].dt.dayofweek
        df_copy['offer_weekofyear'] = df_copy['offer_created'].dt.isocalendar().week
        df_copy['offer_quarter'] = df_copy['offer_created'].dt.quarter
        df_copy = df_copy.drop('offer_created', axis=1)

    # Удаление last_update и date_closed
    df_copy = df_copy.drop(columns=[col for col in ['last_update', 'date_closed'] if col in df_copy.columns], axis=1)

    # Применяем остальные преобразования
    df_copy = process_categorical_columns(df_copy)
    df_copy = create_binary_features(df_copy)
    df_copy = prepare_target_variable(df_copy)

    # Обработка owners_count в новых данных
    if 'owners_count' in df_copy.columns:
        df_copy['owners_count'] = df_copy['owners_count'].astype(str)
        df_copy['owners_count'] = df_copy['owners_count'].str.replace(r'(\d+)\+', r'\1', regex=True)
        df_copy['owners_count'] = df_copy['owners_count'].str.extract('(\d+)', expand=False)
        df_copy['owners_count'] = pd.to_numeric(df_copy['owners_count'], errors='coerce')
        median_owners = df_copy['owners_count'].median()
        df_copy['owners_count'] = df_copy['owners_count'].fillna(median_owners)

    # Применяем расширенный feature engineering
    df_copy = advanced_feature_engineering(df_copy)

    # Обработка опций и истории
    df_copy = process_options_columns(df_copy)
    df_copy = process_history_columns(df_copy)

    # Обработка категориальных колонок для новых данных
    categorical_columns_new = [col for col in categorical_columns if col in df_copy.columns]
    low_cardinality_cols_new = [col for col in categorical_columns_new if df_copy[col].nunique() <= 20]
    high_cardinality_cols_new = [col for col in categorical_columns_new if df_copy[col].nunique() > 20]

    # One-Hot Encoding для новых данных
    for col in low_cardinality_cols_new:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('unknown')
            # В реальном сценарии здесь нужно использовать обученный encoder
            # Для простоты используем простую обработку
            df_copy = pd.get_dummies(df_copy, columns=[col], prefix=[col])

    # Frequency Encoding для новых данных
    for col in high_cardinality_cols_new:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('unknown')
            # В реальном сценарии здесь нужно использовать обученный frequency mapping
            freq_map = df_copy[col].value_counts().to_dict()
            df_copy[f'{col}_freq'] = df_copy[col].map(freq_map)
            df_copy = df_copy.drop(col, axis=1)

    # Обработка текстовых колонок
    for col in text_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
            df_copy[f'{col}_length'] = df_copy[col].str.len()
            if 'configuration' in col:
                extracted_vol = df_copy[col].str.extract('(\d+\.\d+)')[0].astype(float)
                df_copy['extracted_engine_volume'] = extracted_vol
            df_copy = df_copy.drop(col, axis=1)

    # Обработка image_urls
    if 'image_urls' in df_copy.columns:
        if 'image_urls_count' not in df_copy.columns:
            df_copy['image_urls_count'] = df_copy['image_urls'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        df_copy = df_copy.drop('image_urls', axis=1)


    # Удаляем колонки
    existing_columns_to_drop_new = [col for col in columns_to_drop if col in df_copy.columns]
    df_copy = df_copy.drop(existing_columns_to_drop_new, axis=1)

    # Применяем сохраненные модели для кластеризации и PCA если они переданы
    if kmeans_model is not None and pca_model is not None and scaler is not None:
        cluster_features = ['horse_power', 'displacement', 'car_age', 'km_age']
        if all(feat in df_copy.columns for feat in cluster_features):
            cluster_data = df_copy[cluster_features].fillna(0)
            cluster_data_scaled = scaler.transform(cluster_data)

            df_copy['tech_cluster'] = kmeans_model.predict(cluster_data_scaled)
            pca_result = pca_model.transform(cluster_data_scaled)
            df_copy['tech_pca_1'] = pca_result[:, 0]
            df_copy['tech_pca_2'] = pca_result[:, 1]

    # Если передан scaler, применяем нормализацию
    if scaler is not None and 'price_log' in df_copy.columns:
        X_new = df_copy.drop('price_log', axis=1)
        numeric_cols_new = X_new.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_new) > 0:
            X_new[numeric_cols_new] = scaler.transform(X_new[numeric_cols_new])
        df_copy = pd.concat([X_new, df_copy['price_log']], axis=1)

    return df_copy


print("\n" + "=" * 80)
print("РЕЗЮМЕ РАСШИРЕННОГО FEATURE ENGINEERING")
print("=" * 80)
print("✓ Сегментация автомобилей по характеристикам")
print("✓ Взаимодействия признаков для разных ценовых категорий")
print("✓ Биннинг и нелинейные преобразования")
print("✓ Кластеризация и PCA для технических характеристик")
print("✓ Временные и географические признаки")
print("✓ Композитные индексы стоимости и обслуживания")
print("✓ Признаки для редких и эксклюзивных автомобилей")
print(f"✓ Итого создано: {X.shape[1]} признаков")
print(f"✓ Общее количество наблюдений: {X.shape[0]}")
print("\nОбработка данных завершена!")








# # Загрузка данных
# df = pd.read_csv("/Users/mariabug/ML/new_car_sales.csv")  # укажите путь к файлу


# final_df.to_csv('/Users/mariabug/ML/preprocessed_car_data_with_binary.csv', index=False)


# ПЕРЕИМЕНОВЫВАЕМ КОЛОНКИ, УБИРАЕМ ПРОПУСКИ, ВЕРХНИЙ РЕГИСТР, КИРИЛЛИЦУ

# df.rename(columns={ 'options_Салон': 'options_salon',
#                     'options_Обогрев': 'options_heating',
#                     'options_Электростеклоподъемники': 'options_electric_windows',
#                     'options_Электропривод': 'options_electric_drive',
#                     'options_Помощь при вождении': 'options_drive_help',
#                     'options_Противоугонная система': 'options_antitheft',
#                     'options_Подушки безопасности': 'options_airbags',
#                     'options_Активная безопасность': 'options_activesafety',
#                     'options_Мультимедиа и навигация': 'options_multimedia_navigation',
#                     'options_Управление климатом': 'options_climatecontrol'
#                     }, inplace=True)


# # Сохранение результата с правильными параметрами
# output_path = "/Users/mariabug/ML/new_car_sales.csv"
# df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
#
#
# print("\n=== ПРОВЕРКА СОХРАНЕННЫХ ДАННЫХ ===")
# check_df = pd.read_csv(output_path, encoding='utf-8-sig')



