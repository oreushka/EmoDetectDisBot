import numpy as np
import torch

# preprocessing.py
from scipy.signal import butter, filtfilt, find_peaks, medfilt, welch
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionModel:
    def __init__(self):
        # Загрузка модели и токенизатора для текста
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.text_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return self._get_emotion_label(predicted_class)

    def _get_emotion_label(self, class_id):
        emotion_labels = ["happy", "sad", "angry", "neutral"]
        return emotion_labels[class_id]


def butterworth_filter(ppg_signal, lowcut, highcut, fs, order=5):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, ppg_signal)
    return filtered_signal


def preprocess_ppg_signal(ppg_signal):
    filtered_signal = butterworth_filter(
        ppg_signal, lowcut=0.5, highcut=10, fs=50, order=5
    )
    baseline_removed_signal = filtered_signal - np.mean(filtered_signal)
    normalized_signal = baseline_removed_signal / np.max(
        np.abs(baseline_removed_signal)
    )
    return normalized_signal


def extract_features_from_points(interest_points, ppg_signal):
    features = []
    for i in range(len(interest_points) - 1):
        peak_amplitude = ppg_signal[interest_points[i]]
        next_peak_amplitude = ppg_signal[interest_points[i + 1]]
        peak_distance = interest_points[i + 1] - interest_points[i]
        mean_amplitude_between_peaks = np.mean(
            ppg_signal[interest_points[i] : interest_points[i + 1]]
        )

        features.extend([peak_amplitude, peak_distance, mean_amplitude_between_peaks])

    return features


def find_interest_points(ppg_signal):
    peaks, _ = find_peaks(ppg_signal, height=0)  # Находим пики в сигнале
    valleys, _ = find_peaks(-ppg_signal, height=0)  # Находим долины в сигнале
    interest_points = np.concatenate([peaks, valleys])  # Объединяем пики и долины
    return np.sort(interest_points)


def extract_hrv_features(ppg_signal):
    # Здесь может быть другой код для извлечения признаков из вариабельности сердечного ритма
    hrv_features = [np.std(ppg_signal)]
    return hrv_features


def extract_spectrum_features(ppg_signal):
    # Вычисляем спектр сигнала
    freqs, psd = welch(ppg_signal)

    # Находим доминирующие частоты
    dominant_freqs = freqs[np.argmax(psd)]

    # Вычисляем мощность в различных диапазонах частот
    low_freq_power = np.sum(
        psd[(freqs >= 0.04) & (freqs < 0.15)]
    )  # Диапазон низких частот (0.04-0.15 Гц)
    high_freq_power = np.sum(
        psd[(freqs >= 0.15) & (freqs < 0.4)]
    )  # Диапазон высоких частот (0.15-0.4 Гц)

    spectrum_features = [dominant_freqs, low_freq_power, high_freq_power]
    return spectrum_features


def analyze_emotions(ppg_signal):
    # Извлечение признаков из сигнала PPG
    preprocessed_signal = preprocess_ppg_signal(ppg_signal)
    interest_points = find_interest_points(preprocessed_signal)
    features = extract_features_from_points(interest_points, preprocessed_signal)

    # Извлечение HRV и спектральных признаков
    hrv_features = extract_hrv_features(ppg_signal)
    spectrum_features = extract_spectrum_features(ppg_signal)

    # Объединение признаков
    all_features = np.concatenate([features, hrv_features, spectrum_features])

    # Нормализация признаков
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features.reshape(1, -1))

    # Загрузка SVM модели для анализа эмоций
    svm_model = joblib.load("svm_model.pkl")

    # Предсказание эмоций с использованием SVM модели
    predicted_emotion = svm_model.predict(scaled_features)
    return predicted_emotion
