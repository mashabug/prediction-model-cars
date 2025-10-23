"""

CLEANING, CREATING COLUMNS


"""




import pandas as pd
import ast
import csv
import numpy as np

"""Изначальный файл"""
file_path = "/Users/mariabug/ML/data/combined.csv"

df = pd.read_csv(file_path, encoding="utf-8",
                 low_memory=False)

# Убираем дубликаты по inner_id (если они есть)
df = df.drop_duplicates(subset=["inner_id"])
# Сбрасываем индекс чтобы избежать проблем с присвоением
df = df.reset_index(drop=True)


# ФУНКЦИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ДАННЫХ ИЗ СЛОВАРЕЙ
def extract_dict_data(df, column_name):
    """
    Извлекает данные из колонок, содержащих словари в виде строк
    """
    if column_name not in df.columns:
        return df

    print(f"Обработка колонки {column_name}...")

    # Создаем временный DataFrame для извлеченных данных
    extracted_data = []
    error_count = 0
    success_count = 0

    for idx, value in df[column_name].items():
        if pd.isna(value) or value == '' or value == 'nan' or value == '{}':
            extracted_data.append({})
            continue

        try:
            # Пытаемся распарсить строку как словарь
            if isinstance(value, str):
                # Удаляем лишние пробелы
                value_clean = value.strip()

                # Пропускаем пустые значения после очистки
                if not value_clean or value_clean == '{}' or value_clean == '[]':
                    extracted_data.append({})
                    continue

                # МНОГОУРОВНЕВАЯ ОБРАБОТКА ОШИБОК ПАРСИНГА
                dict_data = None

                # Попытка 1: стандартный literal_eval
                if value_clean.startswith('{') and value_clean.endswith('}'):
                    try:
                        dict_data = ast.literal_eval(value_clean)
                        if isinstance(dict_data, dict):
                            success_count += 1
                        else:
                            dict_data = None
                    except (ValueError, SyntaxError, MemoryError):
                        dict_data = None

                # Попытка 2: обработка случая, когда есть проблемы с кавычками
                if dict_data is None and value_clean.startswith('{') and value_clean.endswith('}'):
                    try:
                        # Пробуем заменить одинарные кавычки на двойные для JSON-совместимости
                        value_fixed = value_clean.replace("'", '"')
                        # Экранируем неэкранированные кавычки внутри строк
                        import re
                        value_fixed = re.sub(r'(?<!\\)"', '\\"', value_fixed)
                        value_fixed = value_fixed.replace('\\"', '"')
                        dict_data = ast.literal_eval(value_fixed)
                        if isinstance(dict_data, dict):
                            success_count += 1
                        else:
                            dict_data = None
                    except (ValueError, SyntaxError, MemoryError):
                        dict_data = None

                # Попытка 3: обработка через json (если установлен)
                if dict_data is None:
                    try:
                        import json
                        # Пробуем разные варианты форматирования
                        dict_data = json.loads(value_clean)
                        if isinstance(dict_data, dict):
                            success_count += 1
                        else:
                            dict_data = None
                    except (json.JSONDecodeError, ValueError):
                        dict_data = None

                # Попытка 4: ручной парсинг для простых случаев
                if dict_data is None:
                    try:
                        # Упрощенный парсинг для словарей с простой структурой
                        if value_clean.startswith('{') and value_clean.endswith('}'):
                            # Удаляем внешние скобки
                            content = value_clean[1:-1].strip()
                            if content:
                                pairs = []
                                current_key = None
                                current_value = None
                                in_string = False
                                escape_next = False
                                string_char = None

                                i = 0
                                while i < len(content):
                                    char = content[i]

                                    if not in_string and char in [' ', '\t', '\n', ',']:
                                        i += 1
                                        continue

                                    if not in_string and char in ['"', "'"]:
                                        in_string = True
                                        string_char = char
                                        i += 1
                                        continue

                                    if in_string and not escape_next and char == string_char:
                                        in_string = False
                                        string_char = None
                                        i += 1
                                        continue

                                    if in_string and char == '\\' and not escape_next:
                                        escape_next = True
                                        i += 1
                                        continue

                                    if escape_next:
                                        escape_next = False
                                        i += 1
                                        continue

                                    if not in_string and char == ':':
                                        if current_key is not None:
                                            # Начинаем значение
                                            current_value = ''
                                            i += 1
                                            # Пропускаем пробелы после двоеточия
                                            while i < len(content) and content[i] in [' ', '\t', '\n']:
                                                i += 1
                                            continue
                                    i += 1

                                # Если удалось извлечь какие-то данные
                                if ':' in content:
                                    dict_data = {}
                                    try:
                                        # Простая логика разделения по запятым
                                        pairs = content.split(',')
                                        for pair in pairs:
                                            if ':' in pair:
                                                key_part, value_part = pair.split(':', 1)
                                                # Очистка ключа и значения
                                                key_clean = key_part.strip().strip('"\'').strip()
                                                value_clean = value_part.strip().strip('"\'').strip()
                                                if key_clean and value_clean:
                                                    dict_data[key_clean] = value_clean
                                        if dict_data:
                                            success_count += 1
                                    except:
                                        dict_data = None
                    except Exception:
                        dict_data = None

                # Если все попытки не удались
                if dict_data is None or not isinstance(dict_data, dict):
                    error_count += 1
                    if error_count <= 10:  # Выводим только первые 10 ошибок
                        print(f"Ошибка парсинга в строке {idx}: {value[:100]}...")
                        print(f"  Полное значение: {value}")
                    extracted_data.append({})
                else:
                    extracted_data.append(dict_data)
            else:
                # Если значение не строка
                extracted_data.append({})

        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Выводим только первые 10 ошибок
                print(f"Критическая ошибка в строке {idx}: {e}")
                print(f"Значение: {value[:100]}...")
            extracted_data.append({})

    # Выводим общую статистику
    print(f"Успешно обработано: {success_count}, ошибок парсинга: {error_count}")

    # Преобразуем список словарей в DataFrame
    if extracted_data:
        temp_df = pd.DataFrame(extracted_data)
        temp_df.index = df.index  # Сохраняем оригинальные индексы

        # Переименовываем колонки
        temp_df.columns = [f"{column_name}_{col}" for col in temp_df.columns]

        # Объединяем с основным DataFrame
        df = pd.concat([df, temp_df], axis=1)

        print(f"Из колонки {column_name} извлечено {len(temp_df.columns)} признаков")
    else:
        print(f"Не удалось извлечь данные из колонки {column_name}")

    return df


# ФУНКЦИЯ ДЛЯ ОБРАБОТКИ СПИСКОВ (например, image_urls)
def extract_list_data(df, column_name):
    """
    Обрабатывает колонки, содержащие списки (например, URLs)
    """
    if column_name not in df.columns:
        return df

    print(f"Обработка списка в колонке {column_name}...")

    # Создаем временные списки
    count_values = []
    first_values = []

    processed_count = 0
    error_count = 0

    for idx, value in df[column_name].items():
        if pd.isna(value) or value == '' or value == 'nan':
            count_values.append(0)
            first_values.append('')
            continue

        # Обработка случая, когда значение уже является списком (не строкой)
        if isinstance(value, list):
            count_values.append(len(value))
            first_values.append(value[0] if value else '')
            processed_count += 1
            continue

        # Обработка строковых представлений списков
        if isinstance(value, str):
            value_clean = value.strip()

            if value_clean == '[]' or value_clean == 'nan' or value_clean == '':
                count_values.append(0)
                first_values.append('')
                continue

            try:
                # МНОГОУРОВНЕВАЯ ОБРАБОТКА ОШИБОК ПАРСИНГА
                list_data = None

                # Попытка 1: стандартный парсинг
                if value_clean.startswith('[') and value_clean.endswith(']'):
                    try:
                        # Обработка специфического случая с null в JSON
                        if '{"url":null}' in value_clean or '[{"url":null}' in value_clean:
                            # Это некорректный JSON - обрабатываем как пустой список
                            count_values.append(0)
                            first_values.append('')
                            error_count += 1
                            continue

                        # Заменяем null на None для корректного парсинга
                        value_clean_fixed = value_clean.replace('null', 'None')
                        list_data = ast.literal_eval(value_clean_fixed)
                    except (ValueError, SyntaxError):
                        list_data = None

                # Попытка 2: обработка через json
                if list_data is None:
                    try:
                        import json
                        list_data = json.loads(value_clean)
                    except (json.JSONDecodeError, ValueError):
                        list_data = None

                # Попытка 3: ручной парсинг для простых случаев
                if list_data is None:
                    # Если это не список в формате [...], пробуем другие форматы
                    if ',' in value_clean:
                        # Пробуем разделить по запятым
                        items = [item.strip().strip('"\'') for item in value_clean.split(',')]
                        list_data = items
                    else:
                        # Рассматриваем как одиночный элемент
                        list_data = [value_clean]

                if isinstance(list_data, list):
                    # Извлекаем URL из словарей, если необходимо
                    if list_data and isinstance(list_data[0], dict) and 'url' in list_data[0]:
                        urls = [item['url'] for item in list_data if item.get('url')]
                        count_values.append(len(urls))
                        first_values.append(urls[0] if urls else '')
                    else:
                        count_values.append(len(list_data))
                        first_values.append(list_data[0] if list_data else '')
                    processed_count += 1
                else:
                    count_values.append(0)
                    first_values.append('')
                    error_count += 1

            except (ValueError, SyntaxError, IndexError) as e:
                error_count += 1
                if error_count <= 10:  # Выводим только первые 10 ошибок
                    print(f"Ошибка парсинга списка в строке {idx}: {value[:100]}... Ошибка: {e}")

                # Попытка извлечь URL с помощью регулярного выражения
                import re
                urls = re.findall(r'https?://[^\s,\]]+', value_clean)
                if urls:
                    count_values.append(len(urls))
                    first_values.append(urls[0])
                    processed_count += 1
                else:
                    count_values.append(0)
                    first_values.append('')
        else:
            # Для других типов данных
            count_values.append(0)
            first_values.append('')
            error_count += 1

    print(f"Успешно обработано: {processed_count}, ошибок: {error_count}")

    # Создаем Series с правильными индексами
    count_series = pd.Series(count_values, index=df.index, dtype=int)
    first_series = pd.Series(first_values, index=df.index, dtype=object)

    # Заменяем пустые строки на None для корректного сохранения
    first_series = first_series.replace('', None)

    # Присваиваем колонки
    df[f'{column_name}_count'] = count_series
    df[f'{column_name}_first'] = first_series

    # Проверяем результат
    non_null_count = df[f'{column_name}_first'].notna().sum()
    print(f"Создано {non_null_count} непустых значений в {column_name}_first")

    return df

# ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА КОЛОНОК СО СЛОВАРЯМИ И СПИСКАМИ
print("=== ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА СЛОВАРЕЙ И СПИСКОВ ===")

# Проверяем исходные данные
print(f"Всего строк в DataFrame: {len(df)}")
print(f"Непустых значений в image_urls: {df['image_urls'].notna().sum()}")


# Анализ исходных данных image_urls
def analyze_initial_image_urls(df):
    """Анализирует исходные данные в колонке image_urls"""
    print("\nАнализ исходных данных image_urls:")

    # Примеры различных форматов
    sample_data = df['image_urls'].dropna().head(10)
    for idx, value in sample_data.items():
        print(f"Строка {idx}: тип={type(value)}, значение='{value}'")

    # Статистика по длине значений
    lengths = df['image_urls'].dropna().apply(lambda x: len(str(x)))
    print(f"\nСтатистика длины значений image_urls:")
    print(f"Минимум: {lengths.min()}, Максимум: {lengths.max()}, Среднее: {lengths.mean():.2f}")

    # Подсчет пустых списков
    empty_lists = df['image_urls'].apply(lambda x: x == '[]' if pd.notna(x) else False).sum()
    print(f"Пустых списков '[]': {empty_lists}")


# Вызываем анализ перед обработкой
analyze_initial_image_urls(df)

# Обрабатываем колонки со словарями
dict_columns = ['options', 'history']
for col in dict_columns:
    df = extract_dict_data(df, col)

# Обрабатываем колонки со списками
list_columns = ['image_urls']
for col in list_columns:
    df = extract_list_data(df, col)


# Анализ проблемных строк в image_urls
def analyze_problematic_rows(df):
    """Анализирует строки, где image_urls не пустое, но image_urls_first пустое"""
    mask = (df['image_urls'].notna()) & (df['image_urls'] != '') & (df['image_urls'] != '[]') & (
                df['image_urls_first'].isna() | (df['image_urls_first'] == ''))
    problematic_rows = df[mask]

    if len(problematic_rows) > 0:
        print(f"\nНайдено {len(problematic_rows)} проблемных строк:")
        for idx, row in problematic_rows.head(10).iterrows():
            print(f"Строка {idx}: image_urls = '{row['image_urls']}'")
            print(f"         image_urls_first = '{row['image_urls_first']}'")
    else:
        print("\nПроблемных строк не найдено")

    return problematic_rows


# Вызываем эту функцию после обработки
problematic = analyze_problematic_rows(df)

# Проверяем результат обработки
print(f"\nРезультат обработки image_urls:")
print(f"Всего строк: {len(df)}")
print(f"Непустых значений в image_urls_first: {df['image_urls_first'].notna().sum()}")
print(f"Пустых значений в image_urls_first: {df['image_urls_first'].isna().sum()}")

# Удаляем оригинальные колонки после извлечения данных
columns_to_drop = ['image_urls', 'options', 'history', 'update_history', 'last_update_price', 'last_update_other',
                   'source_file']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

""" 1.1 ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ (MISSING VALUES) """

# 1. Анализ пропущенных значений
print("\n=== ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
print("Пропущенные значения в данных:")
print(df.isnull().sum())

# 1. Обработка pts - создаем бинарный признак вместо удаления
if 'pts' in df.columns:
    # Сохраняем исходные значения pts перед заполнением
    df['has_pts_info'] = df['pts'].notna().astype(int)

    # Определяем моду для заполнения (не предполагаем, что это всегда 1)
    pts_mode = df['pts'].mode()
    if not pts_mode.empty:
        fill_value = pts_mode.iloc[0]
    else:
        fill_value = 1  # Значение по умолчанию, если моды нет

    df['pts'] = df['pts'].fillna(fill_value)  # Заполняем реальной модой

# 2. Удаление столбцов с большим количеством NaN (>70%), кроме pts
threshold = 0.7
nan_ratio = df.isnull().sum() / len(df)
cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()

# Исключаем pts из списка удаления, если он там есть
if 'pts' in cols_to_drop:
    cols_to_drop.remove('pts')

if cols_to_drop:
    print(f"\nУдаляем столбцы с >70% NaN: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# 3. Обработка идентификаторов и URL
id_columns = ['inner_id', 'id', 'seller_id']
for col in id_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        # Удаляем строки только если в столбце есть пропуски
        df = df.dropna(subset=[col])

# Для URL заполняем пустой строкой
url_columns = ['url', 'seller_url']
for col in url_columns:
    if col in df.columns:
        # Приводим всё к строке
        df[col] = df[col].astype(str)
        # Заменяем технические пустышки на ""
        df[col] = df[col].replace(
            to_replace=["nan", "NaN", "None", "NULL", "null", " ", "  "],
            value=""
        )
        # Заполняем настоящие NaN
        df[col] = df[col].fillna("")

# 4. Обработка основных характеристик автомобиля
# Марка и модель - заполняем самой частой комбинацией
if 'mark' in df.columns:
    if df['mark'].isnull().sum() > 0:
        # Безопасное получение моды
        mark_mode = df['mark'].mode()
        if not mark_mode.empty:
            most_common_mark = mark_mode.iloc[0]
            # Если мода это список, берем первый элемент
            if isinstance(most_common_mark, (list, pd.Series)):
                most_common_mark = most_common_mark[0] if len(most_common_mark) > 0 else 'Unknown'
        else:
            most_common_mark = 'Unknown'
        df['mark'] = df['mark'].fillna(most_common_mark)


if 'mark' in df.columns and 'model' in df.columns:
    if df['model'].isnull().sum() > 0:
        # Безопасное заполнение модели на основе марки
        df['model'] = df.groupby('mark')['model'].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        )
        # Дополнительная проверка для случая, когда mode возвращает список
        df['model'] = df['model'].apply(lambda x: x[0] if isinstance(x, (list, pd.Series)) and len(x) > 0 else x)

# Год выпуска - заполняем медианным значением по марке и модели
if 'year' in df.columns and 'mark' in df.columns and 'model' in df.columns:
    # Безопасное вычисление медианы
    df['year'] = df.groupby(['mark', 'model'])['year'].transform(
        lambda x: x.fillna(x.median()) if not x.isnull().all() else x
    )
    # Если остались пропуски, заполняем общим медианным годом
    df['year'] = df['year'].fillna(df['year'].median())

# Пробег - заполняем средним по марке, модели и году
if 'km_age' in df.columns and 'mark' in df.columns and 'model' in df.columns and 'year' in df.columns:
    # Безопасное вычисление среднего
    df['km_age'] = df.groupby(['mark', 'model', 'year'])['km_age'].transform(
        lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
    )
    # Если остались пропуски, заполняем медианой
    df['km_age'] = df['km_age'].fillna(df['km_age'].median())

# 5. Обработка технических характеристик
tech_columns = [
    'generation', 'configuration', 'complectation', 'body_type', 'color',
    'displacement', 'drive_type', 'engine_type', 'horse_power', 'transmission',
    'wheel', 'owners_count', 'condition', 'custom', 'vin'
]

for col in tech_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        # Для числовых характеристик
        if pd.api.types.is_numeric_dtype(df[col]):
            # Заполняем медианой по марке и модели
            if 'mark' in df.columns and 'model' in df.columns:
                df[col] = df.groupby(['mark', 'model'])[col].transform(
                    lambda x: x.fillna(x.median()) if not x.isnull().all() else x
                )
            # Если остались пропуски, заполняем общей медианой
            df[col] = df[col].fillna(df[col].median())
        # Для категориальных характеристик
        else:
            # Заполняем модой по марке и модели
            if 'mark' in df.columns and 'model' in df.columns:
                df[col] = df.groupby(['mark', 'model'])[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else 'Unknown'
                )
            # Если остались пропуски, заполняем общей модой
            mode_vals = df[col].mode()
            # ИСПРАВЛЕНИЕ: Безопасное получение моды
            if not mode_vals.empty:
                mode_val = mode_vals.iloc[0]
                # Если mode_val это список, берем первый элемент
                if isinstance(mode_val, (list, pd.Series)):
                    mode_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            else:
                mode_val = 'Unknown'
            df[col] = df[col].fillna(mode_val)

# 6. Обработка информации о продавце
seller_columns = ['seller', 'seller_type', 'region', 'city', 'address', 'soft_key']
for col in seller_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        # Безопасное получение моды
        mode_vals = df[col].mode()
        # ИСПРАВЛЕНИЕ: Безопасное получение моды
        if not mode_vals.empty:
            mode_val = mode_vals.iloc[0]
            # Если mode_val это список, берем первый элемент
            if isinstance(mode_val, (list, pd.Series)):
                mode_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
        else:
            mode_val = 'Unknown'
        df[col] = df[col].fillna(mode_val)


# 7. Обработка временных меток
date_columns = ['offer_created', 'last_update', 'date_closed']
for col in date_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        # Для дат можно заполнить самой частой датой или удалить строки
        if col == 'offer_created':
            # Это важное поле, лучше удалить строки без даты создания
            df = df.dropna(subset=[col])
        else:
            # Для других дат можно заполнить датой создания предложения
            if 'offer_created' in df.columns:
                df[col] = df[col].fillna(df['offer_created'])
            else:
                # Если нет даты создания, удаляем строки
                df = df.dropna(subset=[col])

# 8. Обработка текстовых полей
text_columns = ['description']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('')

# 9. Обработка ценовых характеристик
price_columns = ['price_low', 'price_high', 'price_diff_last_update',
                 'price_diff_since_posted', 'update_count']
for col in price_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        # Заполняем на основе основной цены
        if 'price_rub' in df.columns:
            if col == 'price_low':
                df[col] = df[col].fillna(df['price_rub'] * 0.9)  # 10% ниже
            elif col == 'price_high':
                df[col] = df[col].fillna(df['price_rub'] * 1.1)  # 10% выше
            else:
                # Для остальных ценовых полей заполняем 0
                df[col] = df[col].fillna(0)
        else:
            # Если нет основной цены, заполняем медианой
            df[col] = df[col].fillna(df[col].median())

# 10. Обработка новых колонок, созданных из словарей
# Для числовых колонок заполняем медианой, для строковых - модой
for col in df.columns:
    if col.startswith(('options_', 'history_', 'image_urls_')):
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_vals = df[col].mode()
                # ИСПРАВЛЕНИЕ: Безопасное получение моды
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    # Если mode_val это список, берем первый элемент
                    if isinstance(mode_val, (list, pd.Series)):
                        mode_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                else:
                    mode_val = 'Unknown'
                df[col] = df[col].fillna(mode_val)

# 11. Проверка результата
print("\nПосле обработки пропусков:")
print(df.isnull().sum())

print("\nИнформация о данных:")
print(df.info())

# Сохранение результата с правильными параметрами
output_path = "/Users/mariabug/ML/data/combined_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
print(f"\nДанные сохранены в: {output_path}")
print(f"Размер сохраненных данных: {df.shape}")

# Проверка сохраненных данных с правильным подсчетом
print("\n=== ПРОВЕРКА СОХРАНЕННЫХ ДАННЫХ ===")
check_df = pd.read_csv(output_path, encoding='utf-8', low_memory=False)
print(f"Размер загруженных данных: {check_df.shape}")
print("\nКолонки, связанные с image_urls, options, history:")
related_cols = [col for col in check_df.columns if any(x in col for x in ['image_urls', 'options', 'history'])]
for col in related_cols:
    # Считаем непустые значения правильно (исключаем пустые строки)
    non_null_count = check_df[col].apply(lambda x: x != '' and pd.notna(x)).sum()
    print(f"{col}: {non_null_count} непустых значений")

