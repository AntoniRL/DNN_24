import os
import numpy as np
import wfdb
import neurokit2 as nk
import pandas as pd

import wfdb.processing
import tensorflow as tf
from signal_reader import SignalReader
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import load_model
from keras.losses import BinaryCrossentropy

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

def add_has_afib_column(df):
    df['has_AFIB'] = (df['num_AFIB_annotations'] > 0).astype(int)
    return df

def calculate_pr_interval(info, sampling_rate):
  p_peaks = info["ECG_P_Peaks"]
  r_peaks = info["ECG_R_Peaks"]
  pr_intervals = []
  for p_peak, r_peak in zip(p_peaks, r_peaks):
      if not np.isnan(p_peak) and not np.isnan(r_peak):
          pr_interval = ((r_peak - p_peak) / sampling_rate) * 1000
          pr_intervals.append(pr_interval)
  pr_interval_mean = np.mean(pr_intervals) if pr_intervals else 0
  pr_interval_std = np.std(pr_intervals) if pr_intervals else 0
  return pr_interval_mean, pr_interval_std

def calculate_qrs_duration(info, sampling_rate):
  q_peaks = info["ECG_Q_Peaks"]
  r_onsets = info["ECG_R_Onsets"]
  s_peaks = info["ECG_S_Peaks"]
  qrs_durations = []
  for i in range(min(len(q_peaks), len(s_peaks))):
      if not np.isnan(q_peaks[i]) and not np.isnan(r_onsets[i]) and not np.isnan(s_peaks[i]):
          nearest_q_onset = min(q_peaks[i], r_onsets[i])
          qrs_duration = ((s_peaks[i] - nearest_q_onset) / sampling_rate) * 1000
          qrs_durations.append(qrs_duration)
  qrs_duration_mean = np.mean(qrs_durations) if qrs_durations else 0
  qrs_duration_std = np.std(qrs_durations) if qrs_durations else 0
  return qrs_duration_mean, qrs_duration_std


def calculate_qt_interval(info, sampling_rate):
  q_peaks = info["ECG_Q_Peaks"]
  t_offsets = info["ECG_T_Offsets"]
  qt_intervals = []
  for i in range(min(len(q_peaks), len(t_offsets))):
      if not np.isnan(q_peaks[i]) and not np.isnan(t_offsets[i]):
          nearest_q_peak = min(q_peaks[i], t_offsets[i])
          qt_interval = ((t_offsets[i] - nearest_q_peak) / sampling_rate) * 1000
          qt_intervals.append(qt_interval)
  qt_interval_mean = np.mean(qt_intervals) if qt_intervals else 0
  qt_interval_std = np.std(qt_intervals) if qt_intervals else 0
  return qt_interval_mean, qt_interval_std


def calculate_poincare(rr_intervals):
  rr_n = rr_intervals[:-1]
  rr_n1 = rr_intervals[1:]
  sd1 = np.std(np.subtract(rr_n1, rr_n) / np.sqrt(2))
  sd2 = np.std(np.add(rr_n1, rr_n) / np.sqrt(2))
  return sd1, sd2

def process_ecg_interval(record_path, record_name, start_sample, end_sample, interval_index):
  global total_N_annotations
  global total_AFIB_annotations
  global last_annotation
  global last_annotation_type

  record_header = wfdb.rdheader(record_path)
  total_samples = record_header.sig_len
  end_sample = min(end_sample, total_samples)

  if end_sample - start_sample < 10: 
      print(f"Error processing interval {interval_index}: The data length is too small to be segmented.")
      return None

  record_segment = wfdb.rdrecord(record_path, sampfrom=start_sample, sampto=end_sample)
  ecg_signal = record_segment.p_signal[:, 0] 
  sampling_rate = record_segment.fs

  try:
      annotations = wfdb.rdann(record_path, 'atr', sampfrom=start_sample, sampto=end_sample)
      print("atr annotations")
      print(annotations)
      print("sample" + str(annotations.sample))
      print("symbol" + str(annotations.symbol) + str(annotations.subtype))
      print("aux_note" + str(annotations.aux_note))
      print("")
  except FileNotFoundError:
      annotations = None

  try:
      qrs_annotations = wfdb.rdann(record_path, 'qrs', sampfrom=start_sample, sampto=end_sample)
  except FileNotFoundError:
      qrs_annotations = None

  try:
      ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
  except Exception as e:
      print(f"Error processing interval {interval_index}: {e}")
      return None

  heart_rate = ecg_signals["ECG_Rate"]
  signal_quality = ecg_signals["ECG_Quality"]
  avg_quality = np.mean(signal_quality)
  r_peaks = info["ECG_R_Peaks"]
  hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

  pr_interval_mean, pr_interval_std = calculate_pr_interval(info, sampling_rate)
  qrs_duration_mean, qrs_duration_std = calculate_qrs_duration(info, sampling_rate)
  qt_interval_mean, qt_interval_std = calculate_qt_interval(info, sampling_rate)

  rr_intervals = np.diff(r_peaks) / sampling_rate * 1000 
  cv = hrv_time["HRV_SDNN"].iloc[0] / hrv_time["HRV_MeanNN"].iloc[0]

  sd1, sd2 = calculate_poincare(rr_intervals)

  features = {
      "record_name": record_name,
      "start_time": interval_index / 6,
      "sampling_rate": sampling_rate,
      "heart_rate_mean": heart_rate.mean(),
      "heart_rate_std": heart_rate.std(),
      "signal_quality": avg_quality,
      "pr_interval_mean": pr_interval_mean, 
      "pr_interval_std": pr_interval_std,
      "qrs_duration_mean": qrs_duration_mean,
      "qrs_duration_std": qrs_duration_std,
      "qt_interval_mean": qt_interval_mean,
      "qt_interval_std": qt_interval_std,
      "hrv_rmssd": hrv_time["HRV_RMSSD"].iloc[0],
      "hrv_mean": hrv_time["HRV_MeanNN"].iloc[0],
      "hrv_sdnn": hrv_time["HRV_SDNN"].iloc[0],
      "cv": cv,
      "sd1": sd1, 
      "sd2": sd2,
  }

  if annotations is not None:
      features["num_annotations"] = len(annotations.sample)
      num_N_annotations = 0
      num_AFIB_annotations = 0
      aux_notes = annotations.aux_note
      if aux_notes:
          for note in aux_notes:
              if note == '(N':
                  num_N_annotations = 1
                  num_AFIB_annotations = 0
                  total_N_annotations += 1
                  last_annotation = 'N'
              elif note == '(AFIB':
                  num_N_annotations = 0
                  num_AFIB_annotations = 1
                  total_AFIB_annotations += 1
                  last_annotation = 'AFIB'

      features["num_N_annotations"] = num_N_annotations
      features["num_AFIB_annotations"] = num_AFIB_annotations
      features["total_N_annotations"] = total_N_annotations
      features["total_AFIB_annotations"] = total_AFIB_annotations
  else:
      features["num_annotations"] = 1
      features["num_AFIB_annotations"] = 1
      features["total_AFIB_annotations"] = 1
      features["num_N_annotations"] = 0
      features["total_N_annotations"] = 0

  if last_annotation == 'N':
      features["num_N_annotations"] = 1
      features["num_AFIB_annotations"] = 0
  elif last_annotation == 'AFIB':
      features["num_N_annotations"] = 0
      features["num_AFIB_annotations"] = 1

  if qrs_annotations is not None:
      features["num_qrs_annotations"] = len(qrs_annotations.sample)
  else:
      features["num_qrs_annotations"] = 0

  return features

# 10 seconds intervals
def process_ecg_record(record_path, record_name):
  record = wfdb.rdheader(record_path)
  sampling_rate = record.fs
  total_samples = record.sig_len
  ten_sec_intervals = sampling_rate * 10

  num_intervals = total_samples // ten_sec_intervals
  if total_samples % ten_sec_intervals != 0:
      num_intervals += 1 
  all_features = []

  for i in range(num_intervals):
      start_sample = i * ten_sec_intervals
      end_sample = start_sample + ten_sec_intervals
      try:
          features = process_ecg_interval(record_path, record_name, start_sample, end_sample, i)
          if features is not None:
              all_features.append(features)
      except Exception as e:
          print(f"Error processing interval {i}: {e}")

  return all_features

def load_and_combine_data(data_dir, output_file):
    data_frames = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"File {file_name} contains {len(df)} records")
            print(df.head()) 
            data_frames.append(df)

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df = add_has_afib_column(combined_df)

        print(f"Combined DataFrame contains {len(combined_df)} records")
        print(combined_df.head()) 
        combined_df.to_csv(output_file, index=False)

        saved_df = pd.read_csv(output_file)
        print(f"Re-read combined data saved to {output_file}")
        print(f"Total records in re-read combined file: {len(saved_df)}")
        print(saved_df.head()) 
    else:
        print("No CSV files found in the directory.")

def createDirectories(intervals_directory_path, preprocessed_data_directory_path):
    if not os.path.exists(intervals_directory_path):
        os.makedirs(intervals_directory_path)
        print(f"Directory '{intervals_directory_path}' created.")
    else:
        print(f"Directory '{intervals_directory_path}' already exists.")

    if not os.path.exists(preprocessed_data_directory_path):
        os.makedirs(preprocessed_data_directory_path)
        print(f"Directory '{preprocessed_data_directory_path}' created.")
    else:
        print(f"Directory '{preprocessed_data_directory_path}' already exists.")


def load_data(file_path, specified_record=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    df['record_name'] = df['record_name'].astype(str)
    if specified_record:
        df = df[df['record_name'] == str(specified_record)]
    if 'num_AFIB_annotations' in df.columns:
        actual_afib_annotations = df['num_AFIB_annotations'].values
    else:
        actual_afib_annotations = None
    
    return df, actual_afib_annotations

def preprocess_data(df):
  if df.empty:
      print(df)
      raise ValueError("No data left after filtering. Adjust the filter criteria.")

  features = ['hrv_sdnn', 'hrv_rmssd', "hrv_mean", 'cv', "heart_rate_std", "heart_rate_mean", "sd1", "sd2"]
  scaler = StandardScaler()
  df[features] = scaler.fit_transform(df[features])
  df['sampling_rate'] = df['sampling_rate'].astype(int)

  return df, df[features], df[features].join(df['sampling_rate'])

# prediction method used for LSTM, CNN models
def predict(model, features):
  features = np.array(features).reshape((features.shape[0], 1, features.shape[1]))  # LSTM and CNN
  predictions = model.predict(features)
  return np.argmax(predictions, axis=1)

total_N_annotations = 0
total_AFIB_annotations = 0
last_annotation = None
last_annotation_type = None

class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = keras.models.load_model('./LSTM_model.keras',
                                              custom_objects={
                                                  'bce_dice_weighted_loss': bce_dice_weighted_loss_wrapper(0.5, 0.5),
                                                  'dice_coef': dice_coef_wrapper()
                                              })
        self.intervals_directory_path = "./data/10_sec_intervals"
        self.preprocessed_data_directory_path = "./data/preprocessed_data"

    def evaluate(self, signal_reader: SignalReader):
        # Preprocessing
        record_path = signal_reader.record_path # Path to specific record
        record_name = os.path.basename(record_path)  # Extract the record name

        createDirectories(self.intervals_directory_path, 
                          self.preprocessed_data_directory_path)

        # for record_name in records:
        print("Record path:", record_path)
        print("Record name: ", record_name)
        features = process_ecg_record(record_path, record_name)

        df = pd.DataFrame(features)
        file_name = self.intervals_directory_path + "/" + record_name + "_features.csv"
        # Save the DataFrame to a CSV file
        df.to_csv(file_name, index=False)
        
        output_file = self.preprocessed_data_directory_path + "/afdb_data.csv"
        load_and_combine_data(self.intervals_directory_path, output_file)

        # Predictions
        df, actual_afib_annotations = load_data(output_file)
        df, features, features_sample = preprocess_data(df)
        predictions = predict(self._model, features)

        # Postprocess
        expanded_predictions = []
        for pred in predictions:
            expanded_predictions.extend([pred] * 10)
        predictions = np.array(expanded_predictions)

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), predictions)


if __name__ == '__main__':
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader('./val_db/6.csv')
    record_eval.evaluate(signal_reader)
