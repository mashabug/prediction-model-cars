import pandas as pd
import os


# Функция для создания колонки soft_key
def create_soft_key(df):
    # Функция очистки текста
    def clean_text(text):
        text = str(text)
        # Замена пробелов, точек и запятых на нижнее подчеркивание
        text = text.replace(' ', '_').replace('.', '_').replace(',', '_')
        # Удаление множественных подчеркиваний
        while '__' in text:
            text = text.replace('__', '_')
        return text.strip('_')

    # Создаем колонку soft_key
    if all(col in df.columns for col in ['seller', 'region', 'city', 'address']):
        df['soft_key'] = (
                df['seller'].apply(clean_text) + '_' +
                df['region'].apply(clean_text) + '_' +
                df['city'].apply(clean_text) + '_' +
                df['address'].apply(clean_text)
        )
        return df
    else:
        missing_cols = [col for col in ['seller', 'region', 'city', 'address'] if col not in df.columns]
        print(f"В файле отсутствуют колонки: {missing_cols}")
        return None


# Загрузка списка уникальных soft_key для поиска
soft_keys_df = pd.read_csv('/Users/mariabug/ML/DataJanuary-April/final_soft_keys.csv')  # Замените на ваш файл
unique_soft_keys = set(soft_keys_df.iloc[:, 0])  # Предполагаем, что софт-кеи в первой колонке

print(f"Загружено {len(unique_soft_keys)} уникальных soft_key для поиска")

# Папка с исходными CSV файлами
source_folder = '/Users/mariabug/ML/Fresh_car_data/'  # Замените на путь к вашим исходным файлам
output_folder = '/Users/mariabug/ML/Fresh_car_data/Sorted_fresh_data/'  # Папка для обработанных файлов

# Создаем папку для обработанных файлов, если ее нет
os.makedirs(output_folder, exist_ok=True)

# Список для хранения всех найденных данных (включая дубликаты)
all_found_data = []

# Обработка всех CSV файлов в папке
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_folder, filename)
        print(f"Обрабатывается файл: {filename}")

        try:
            # Чтение CSV файла
            df = pd.read_csv(file_path, sep="|", encoding="utf-8",
                             low_memory=False)

            # Создаем колонку soft_key
            df_with_soft_key = create_soft_key(df)

            if df_with_soft_key is not None:
                # Сохраняем файл с добавленной колонкой soft_key
                output_path = os.path.join(output_folder, f"with_soft_key_{filename}")
                df_with_soft_key.to_csv(output_path, index=False)
                print(f"Файл сохранен с колонкой soft_key: {output_path}")

                # Фильтруем строки, где soft_key есть в нашем списке
                # Сохраняем ВСЕ строки, включая дубликаты
                filtered_df = df_with_soft_key[df_with_soft_key['soft_key'].isin(unique_soft_keys)]

                if not filtered_df.empty:
                    # ИСПРАВЛЕНИЕ: Создаем копию перед модификацией
                    filtered_df = filtered_df.copy()

                    # Добавляем информацию о файле-источнике
                    filtered_df['source_file'] = filename

                    # Добавляем найденные данные в общий список (включая дубликаты)
                    all_found_data.append(filtered_df)

                    print(f"Найдено {len(filtered_df)} строк (включая дубликаты) в файле {filename}")

                    # Показываем информацию о дубликатах
                    duplicate_count = len(filtered_df) - len(filtered_df['soft_key'].unique())
                    if duplicate_count > 0:
                        print(f"  В том числе {duplicate_count} дубликатов soft_key")
                else:
                    print(f"В файле {filename} не найдено совпадений")
            else:
                print(f"Не удалось создать soft_key для файла {filename}")

        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

# Объединяем все найденные данные (сохраняем все дубликаты)
if all_found_data:
    result_df = pd.concat(all_found_data, ignore_index=True)

    # Сохраняем результат со ВСЕМИ строками, включая дубликаты
    result_df.to_csv('/Users/mariabug/ML/Fresh_car_data/Sorted_fresh_data/filtered_data_by_soft_key.csv',
                     index=False)
    print(
        f"Всего найдено {len(result_df)} строк (включая дубликаты). Результат сохранен в "
        f"'/Users/mariabug/ML/Fresh_car_data/Sorted_fresh_data/filtered_data_by_soft_key.csv'")

    # Дополнительно: сохраняем полную статистику
    total_unique_soft_keys_found = result_df['soft_key'].nunique()
    total_duplicates = len(result_df) - total_unique_soft_keys_found

    print(f"Уникальных soft_key найдено: {total_unique_soft_keys_found}")
    print(f"Всего дубликатов: {total_duplicates}")

    # Сохраняем список всех найденных soft_key (включая дубликаты)
    all_soft_keys = result_df[['soft_key']]
    all_soft_keys.to_csv('/Users/mariabug/ML/Fresh_car_data/Sorted_fresh_data/all_found_soft_keys.csv', index=False)

    # Статистика по файлам-источникам
    file_stats = result_df['source_file'].value_counts()
    print("\nСтатистика по файлам-источникам:")
    for file, count in file_stats.items():
        print(f"  {file}: {count} строк")

    # Статистика по дубликатам soft_key
    soft_key_counts = result_df['soft_key'].value_counts()
    duplicate_soft_keys = soft_key_counts[soft_key_counts > 1]
    if not duplicate_soft_keys.empty:
        print(f"\nSoft_key с дубликатами (всего {len(duplicate_soft_keys)}):")
        for soft_key, count in duplicate_soft_keys.head(10).items():  # Показываем первые 10
            print(f"  {soft_key}: {count} повторений")
        if len(duplicate_soft_keys) > 10:
            print(f"  ... и еще {len(duplicate_soft_keys) - 10} soft_key с дубликатами")
else:
    print("Не найдено ни одного совпадения")