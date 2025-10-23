import pandas as pd

# Загрузка данных из CSV файла
df = pd.read_csv("/Users/mariabug/ML/DataJanuary-April/for_step1_white_diller_cars.csv",
                 sep='|', encoding= 'utf-8', low_memory=False)  # Замените на путь к вашему файлу

# Функция для очистки текста
def clean_text(text):
    text = str(text)
    # Замена пробелов, точек и запятых на нижнее подчеркивание
    text = text.replace(' ', '_').replace('.', '_').replace(',', '_')
    # Удаление множественных подчеркиваний
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_')

# Создание новой колонки soft_key
df['soft_key'] = (
    df['seller'].apply(clean_text) + '_' +
    df['region'].apply(clean_text) + '_' +
    df['city'].apply(clean_text) + '_' +
    df['address'].apply(clean_text)
)

# Просмотр результата
print(df[['seller', 'region', 'city', 'address', 'soft_key']].head())

# Сохранение всей таблицы с новой колонкой
df.to_csv('/Users/mariabug/ML/output_file_with_soft_key.csv', index=False)

# Копирование только колонки soft_key в отдельный файл
soft_key_df = df[['soft_key']]

# Сохранение уникальных значений soft_key в отдельный файл
unique_soft_keys = df['soft_key'].drop_duplicates().to_frame()
unique_soft_keys.to_csv('/Users/mariabug/ML/unique_soft_keys.csv', index=False)

print("Уникальные значения soft_key сохранены в файл 'unique_soft_keys.csv'")