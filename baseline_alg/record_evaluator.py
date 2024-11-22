import os

import numpy as np
import keras
import wfdb.processing
import tensorflow as tf
from signal_reader import SignalReader

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


class RecordEvaluator:
    def __init__(self, dest_dir):
        self._dest_dir = dest_dir
        self._model = keras.models.load_model('./model_09.keras',
                                              custom_objects={
                                                  'bce_dice_weighted_loss': bce_dice_weighted_loss_wrapper(0.5, 0.5),
                                                  'dice_coef': dice_coef_wrapper()
                                              })

    def evaluate(self, signal_reader: SignalReader):
        signal = signal_reader.read_signal()
        fs = signal_reader.read_fs()

        xqrs = wfdb.processing.XQRS(sig=signal[:, 0], fs=fs)
        xqrs.detect()
        qrs_inds = xqrs.qrs_inds
        rr = wfdb.processing.calc_rr(qrs_inds, fs=fs, min_rr=None, max_rr=None, qrs_units='samples',
                                     rr_units='seconds')

        input_rr_samples = 30
        batch_size = 64
        qrs_af_probabs = np.zeros(shape=(len(qrs_inds), ), dtype=np.float32)
        qrs_af_overlap = np.zeros(shape=(len(qrs_inds), ), dtype=np.float32)

        pred_step = input_rr_samples // 3

        batch = np.zeros(shape=(batch_size, input_rr_samples, 1), dtype=np.float32)
        batch_idx = 0
        rr_indices_history = []
        for rr_idx in range(0, rr.shape[0]-input_rr_samples, pred_step):
            snippet = rr[rr_idx:rr_idx + input_rr_samples]
            rr_indices_history.append([rr_idx, rr_idx + input_rr_samples])
            snippet = snippet[..., np.newaxis]
            batch[batch_idx] = snippet
            batch_idx += 1

            if batch_idx == batch_size:
                results = self._model(batch)
                for j in range(batch_idx):
                    rr_from, rr_to = rr_indices_history[j]
                    qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
                    qrs_af_overlap[rr_from: rr_to] += 1.0

                batch_idx = 0
                rr_indices_history = []

        if batch_idx > 0:
            results = self._model(batch)
            for j in range(batch_idx):
                rr_from, rr_to = rr_indices_history[j]
                qrs_af_probabs[rr_from: rr_to] += results[j, :, 0]
                qrs_af_overlap[rr_from: rr_to] += 1.0

        qrs_af_overlap[qrs_af_overlap == 0.0] = 1.0
        qrs_af_probabs /= qrs_af_overlap
        qrs_af_preds = np.round(qrs_af_probabs)

        pred = np.zeros([len(signal), ], dtype=np.float32)

        for qrs_idx in range(len(rr)):
            pred[qrs_inds[qrs_idx]: qrs_inds[qrs_idx+1]] = qrs_af_preds[qrs_idx]

        code = signal_reader.get_code()
        np.save(os.path.join(self._dest_dir, f'{code}'), pred)


if __name__ == '__main__':
    record_eval = RecordEvaluator('./')
    signal_reader = SignalReader('./val_db/6.csv')
    record_eval.evaluate(signal_reader)
