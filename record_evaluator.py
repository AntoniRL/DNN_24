import os
import numpy as np
import wfdb
import tensorflow as tf
from signal_reader import SignalReader
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import load_model
from keras.losses import BinaryCrossentropy
import wfdb.processing

def bce_dice_weighted_loss_wrapper(bce_w, dice_w, smooth=10e-6):
	bce_loss = keras.losses.BinaryCrossentropy()
	dice_loss = dice_coef_loss_wrapper(smooth)
	def bce_dice_weighted_loss(y_true, y_pred):
			return bce_w * bce_loss(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)
	return bce_dice_weighted_loss

def dice_coef_wrapper(smooth=10e-6):
	def dice_coef(y_true, y_pred):
			y_true_f = y_true
			y_pred_f = y_pred
			intersection = tf.reduce_sum(y_true_f * y_pred_f)
			dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
			return dice
	return dice_coef

def dice_coef_loss_wrapper(smooth=10e-6):
	dice_coef = dice_coef_wrapper(smooth)
	def dice_coef_loss(y_true, y_pred):
			return 1 - dice_coef(y_true, y_pred)
	return dice_coef_loss

def calculate_hrv(rr_intervals):
	rr_intervals = np.array(rr_intervals)
	diff_rr = np.diff(rr_intervals)

	hrv_rmssd = np.sqrt(np.mean(diff_rr**2))
	hrv_mean_nn = np.mean(rr_intervals)
	hrv_sdnn = np.std(rr_intervals)
	return hrv_rmssd, hrv_mean_nn, hrv_sdnn

def process_ecg_interval(signal_reader, record_name, start_sample, end_sample, interval_index):
  signal = signal_reader.read_signal()
  total_samples = signal.shape[0]
  ecg_signal = signal[:, 0]
  sampling_rate = signal_reader.read_fs()

  end_sample = min(end_sample, total_samples)

  if end_sample - start_sample < 300:
    print(f"Error processing interval {interval_index}: The data length is too small to be segmented.")
    return None

  ecg_segment = ecg_signal[start_sample:end_sample]
  # Use XQRS for QRS detection and RR interval calculation
  xqrs = wfdb.processing.XQRS(sig=ecg_segment, fs=sampling_rate)
  xqrs.detect()
  qrs_inds = xqrs.qrs_inds
  rr_intervals = wfdb.processing.calc_rr(qrs_inds, fs=sampling_rate, min_rr=None, max_rr=None, qrs_units='samples', rr_units='seconds')

  # Calculate heart rate and HRV features
  heart_rate = 60 / rr_intervals
  hrv_rmssd, hrv_mean_nn, hrv_sdnn = calculate_hrv(rr_intervals)

  # Signal quality (placeholder)
  avg_quality = np.mean(np.abs(ecg_segment))

  features = {"record_name": record_name,
			  "start_time": start_sample / sampling_rate,
			  "sampling_rate": sampling_rate,
			  "heart_rate_mean": np.mean(heart_rate),
			  "heart_rate_std": np.std(heart_rate),
			  "signal_quality": avg_quality,
			  "hrv_rmssd": hrv_rmssd,
			  "hrv_mean": hrv_mean_nn,
			  "hrv_sdnn": hrv_sdnn,
			  "cv": np.std(heart_rate) / np.mean(heart_rate)}
  return features

# 5 minutes intervals
def process_ecg_record(signal_reader, record_name, ecg_signal, sampling_rate):
  total_samples = ecg_signal.shape[0]
  five_min_intervals = sampling_rate * 300
  num_intervals = total_samples // five_min_intervals

  if total_samples % five_min_intervals != 0:
    num_intervals += 1

  all_features = []
  for i in range(num_intervals):
    start_sample = i * five_min_intervals
    end_sample = start_sample + five_min_intervals
    features = process_ecg_interval(signal_reader, record_name, start_sample, end_sample, i)
    if features is not None:
      all_features.append(features)
  return all_features

def preprocess_data(features_list):
    filtered_features = []
    for feature in features_list:
        if feature['hrv_sdnn'] <= 500 and feature['hrv_rmssd'] <= 500 and feature['cv'] <= 0.5 and feature['signal_quality'] >= 0.3:
            filtered_feature = [
                feature['hrv_sdnn'],
                feature['hrv_rmssd'],
                feature['hrv_mean'],
                feature['cv'],
                feature['heart_rate_std'],
                feature['heart_rate_mean']
            ]
            filtered_features.append(filtered_feature)
    filtered_features = np.array(filtered_features)
    scaler = StandardScaler()
    filtered_features = scaler.fit_transform(filtered_features)
    return filtered_features

def preprocess_data_without_filtering(features_list):
    all_features = []
    for feature in features_list:
        raw_features = [
            feature['hrv_sdnn'],
            feature['hrv_rmssd'],
            feature['hrv_mean'],
            feature['cv'],
            feature['heart_rate_std'],
            feature['heart_rate_mean']
        ]
        all_features.append(raw_features)
    all_features = np.array(all_features)
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    return all_features

def predict(model, features):
  features = np.array(features).reshape((features.shape[0], 1, features.shape[1]))
  print("features shape: ", features.shape)
  predictions = model.predict(features)
  return np.argmax(predictions, axis=1)

class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = keras.models.load_model('./LSTM_model_trained.keras',
                                              custom_objects={
                                                  'bce_dice_weighted_loss': bce_dice_weighted_loss_wrapper(0.5, 0.5),
                                                  'dice_coef': dice_coef_wrapper()
                                              })

    def evaluate(self, signal_reader: SignalReader):
        # Preprocessing
        print("Preprocessing")
        ecg_signal = signal_reader.read_signal()
        signal = ecg_signal[:, 0]
        sampling_rate = signal_reader.read_fs()
        record_path = signal_reader.record_path
        record_name = os.path.basename(record_path)
        record_name = os.path.splitext(record_name)[0]
        print("Record path: ", record_path)
        print("Record name: ", record_name)

        features_list = process_ecg_record(signal_reader, record_name, signal, sampling_rate)
        preprocessed_features = preprocess_data_without_filtering(features_list)

        # Predictions
        print("Predictions")
        predictions = predict(self._model, preprocessed_features)

        # Postprocess
        print("Postprocess")
        expanded_predictions = []
        for pred in predictions:
            expanded_predictions.extend([pred] * 37500)
        predictions = np.array(expanded_predictions)

        print("Saving results")
        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), predictions)


if __name__ == '__main__':
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader('./val_db/6.csv')
    record_eval.evaluate(signal_reader)