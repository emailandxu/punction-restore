# coding=utf-8
# Copyright 2020 Beijing BluePulse Corp.
# Created by Zhang Guanqun on 2020/6/5


import difflib
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import os
import tensorflow as tf
from scipy.fftpack import fft
from tensorflow.keras import backend as K
from typing import Union, List
import re
import unicodedata


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_shape_invariants(tensor):
    shapes = shape_list(tensor)
    return tf.TensorShape([i if isinstance(i, int) else None for i in shapes])


def normalize_signal(signal: np.ndarray):
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def preprocess_paths(paths: Union[List, str]):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None


def get_reduced_length(length, reduction_factor):
    return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)


def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# word error rate
def get_edit_distance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


# ctc decoder
def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text


# draw loss pic
def plot_metric(history, metric, pic_file_name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(pic_file_name)


# against LAS loop decoding
def text_no_repeat(s):
    repeat_times = 0
    repeat_pattern = ''
    for i in range(1, len(s) // 2):
        pos = i
        if s[0 - 2 * pos:0 - pos] == s[0 - i:]:
            tmp_repeat_pattern = s[0 - i:]
            tmp_repeat_times = 1
            while pos * (tmp_repeat_times + 2) <= len(s) \
                    and s[0 - pos * (tmp_repeat_times + 2):0 - pos * (tmp_repeat_times + 1)] == s[0 - i:]:
                tmp_repeat_times += 1
            if tmp_repeat_times * len(tmp_repeat_pattern) > repeat_times * len(repeat_pattern):
                repeat_times = tmp_repeat_times
                repeat_pattern = tmp_repeat_pattern
    # print(repeat_pattern, '*', repeat_times)
    if len(repeat_pattern) != 1:
        s = s[:0 - repeat_times * len(repeat_pattern)] if repeat_times > 0 else s
    # print(s)
    return s


# convert audio file to feature
def extract_audio_feature(audio_file_path):
    audio, _ = librosa.load(audio_file_path, sr=16000)

    # normalize signal
    signal = np.asfortranarray(audio)
    signal = normalize_signal(signal)
    signal = preemphasis(signal, 0.97)

    # mel filter spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                     sr=16000,
                                                     # stride=10ms
                                                     hop_length=160,
                                                     # window=25ms,
                                                     win_length=400,
                                                     # length of the FFT window
                                                     n_fft=512,
                                                     # output feature dim
                                                     n_mels=80,
                                                     fmax=8000)

    # log mel spectrogram
    mel_spectrogram_log = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=80.0)

    # transpose shape to (frame_num, 80)
    mel_spectrogram_t = np.swapaxes(mel_spectrogram_log, 0, 1)

    # normalize audio feature
    feature = normalize_audio_feature(mel_spectrogram_t)
    return feature


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z)
    w = re.sub(r"[^a-zA-Z'-]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w
