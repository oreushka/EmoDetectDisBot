import re

import numpy as np
from scipy.signal import butter, lfilter


# Предобработка текста
def clean_text(text):
    """
    Очищает текст от HTML-тегов, символов пунктуации и специальных символов.
    """
    text = re.sub(r"<.*?>", "", text)  # Удаление HTML-тегов
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Удаление символов пунктуации
    text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
    text = text.strip().lower()  # Приведение к нижнему регистру
    return text


def tokenize_text(text):
    """
    Токенизация текста на слова.
    """
    return text.split()


def remove_stopwords(tokens, stopwords):
    """
    Удаление стоп-слов из токенов.
    """
    return [token for token in tokens if token not in stopwords]


def preprocess_text(text, stopwords):
    """
    Полный цикл предобработки текста.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens, stopwords)
    return " ".join(tokens)


# Предобработка PPG данных
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Функция для создания коэффициентов фильтрации для полосового фильтра.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Применение полосового фильтра к данным PPG.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize_ppg(data):
    """
    Нормализация данных PPG к диапазону [0, 1].
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def preprocess_ppg(ppg_data, fs=100, lowcut=0.5, highcut=5.0, order=5):
    """
    Полный цикл предобработки данных PPG.
    """
    ppg_data = bandpass_filter(ppg_data, lowcut, highcut, fs, order)
    ppg_data = normalize_ppg(ppg_data)
    return ppg_data


# Пример использования
if __name__ == "__main__":
    # Пример текста для предобработки
    sample_text = "Hello world! This is an example text for preprocessing. <br> Let's clean it up."
    stopwords = ["is", "an", "for", "it"]
    cleaned_text = preprocess_text(sample_text, stopwords)
    print(f"Cleaned Text: {cleaned_text}")

    # Пример данных PPG для предобработки
    sample_ppg = np.random.rand(1000)  # Генерация случайного сигнала для примера
    fs = 100  # Частота дискретизации
    preprocessed_ppg = preprocess_ppg(sample_ppg, fs)
    print(f"Preprocessed PPG: {preprocessed_ppg[:10]}")  # Вывод первых 10 значений
