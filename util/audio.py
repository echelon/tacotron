import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
import scipy
from hparams import hparams

# KEEP
def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))

#KEEP
def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)

#KEEP
def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)

# KEEP
def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  return spectrogram
  # IT LOOKS LIKE THIS REMOVES GRIFFIN-LIM: 
  #S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
  #return _griffin_lim_tensorflow(tf.pow(S, hparams.power))

# KEEP
def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


# - KEEP
def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)

# - KEEP
def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


# - KEEP
def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


# - KEEP
def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

# - KEEP
def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

# - KEEP
def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

# - KEEP
def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

# - KEEP
def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

# - KEEP
def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

# - KEEP
def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

# - KEEP
def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
