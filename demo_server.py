import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
from pprint import pprint

import math
import struct
import io

import numpy as np
import tensorflow as tf

from librosa import effects
from models.tacotron import Tacotron
from text import text_to_sequence
from util import audio

from PIL import Image, ImageDraw

def show_image(wav):
    dimensions = wav.shape
    rows = dimensions[0]
    cols = dimensions[1]

    pprint('>>> Generating image')
    image = Image.new('RGB', dimensions)
    pixels = image.load()

    minimum = wav[0,0]
    maximum = wav[0,0]

    for x in range(0, rows):
        for y in range(0, cols):
            if wav[x,y] > maximum:
                maximum = wav[x,y]
            if wav[x,y] < minimum:
                minimum = wav[x,y]

    print('Minimum: ' + str(minimum))
    print('Maximum: ' + str(maximum))

    if maximum == minimum:
        maximum = 1 # nb: to prevent division by zero below

    for x in range(0, rows):
        for y in range(0, cols):
            v = wav[x,y]
            scaled = int((v - minimum) / (maximum - minimum) * 255)
            pixels[x, y] = (scaled, scaled, scaled)

    image.show()

def mel_to_wavernn_vocoder_input(wav, output_filename):
    dimensions = wav.shape
    rows = dimensions[0]
    cols = dimensions[1]
    pprint('>>> wavernn_vocoder - rows: {}, cols: {}'.format(rows, cols))

    with open(output_filename, 'wb') as f:
        # NB: Swapped. Rows comes first in the input format.
        f.write(struct.pack('i', cols))
        f.write(struct.pack('i', rows))

        # NB: Swapped iteration order.
        for y in range(0, cols):
            for x in range(0, rows):
                value = wav[x,y]
                value = value * 4.0 # Attempt to scale the values.
                f.write(struct.pack('f', value))

class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = Tacotron(hparams)
      self.model.initialize(inputs, input_lengths)
      pprint('>>> Model Linear Ouputs:')
      pprint(self.model.linear_outputs[0])
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    pprint('Text: ' + text)
    #pprint('Seq')
    #pprint(seq)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    pprint(self.wav_output)
    pprint('>>> Getting wav')
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)

    pprint('>>> Gotten wav')
    pprint(wav)
    pprint(wav.shape)

    show_image(wav)

    pprint('>>> Resizing wav')
    downsized_dimensions = (wav.shape[0], 80)
    new_wav = np.zeros(shape=downsized_dimensions, dtype=np.float32)

    rows = downsized_dimensions[0]
    cols = downsized_dimensions[1]

    old_cols = wav.shape[1]
    new_cols = new_wav.shape[1]

    pprint('Rows: {}, Cols: {}'.format(rows, cols))

    for x in range(0, rows):
        for y in range(0, cols):
            # Naive impl:
            # new_wav[x,y] = wav[x,y]

            y_start = math.ceil( y * ( old_cols / new_cols ) )
            y_end = math.ceil( ( y + 1 ) * ( old_cols / new_cols ) )

            accum = 0.0
            for z in range(y_start, y_end):
                accum += wav[x, z]

            accum = accum / (y_end - y_start)
            new_wav[x,y] = accum


    wav = new_wav.copy()
    pprint(wav.shape)

    #pprint('>>> Modify wav with inv_preemphasis and find_endpoint')
    #wav = audio.inv_preemphasis(wav)
    ## The audio is typically ~13 seconds unless truncated:
    #endpoint = audio.find_endpoint(wav)
    #pprint('Endpoint: {}'.format(endpoint))
    #wav = wav[:endpoint]

    #pprint('>>> Transposing wav')
    #wav = wav.transpose()

    pprint('>>> Modified wav')
    pprint(wav)
    pprint(wav.shape)

    show_image(wav)

    pprint('>>Saving wavernn vocoder-formatted mel')
    mel_to_wavernn_vocoder_input(wav, 'wavernn_vocoder_mel_input.mel')

    # NB: cannot call resize() below since matrix doesn't "own" its data
    #wav = wav.copy()
    #pprint('>>> NUMPY RESIZE')
    #wav.resize((1000, 80))
    #pprint('>>> Resized wav')
    #pprint(wav)
    #pprint(wav.shape)

    # Save mel spectrogram
    filename = 'spectrogram.npy'
    pprint('>>> Saving spectrogram as: ' + filename)
    np.save(filename, wav)
    np.save('no_pickle_' + filename, wav, allow_pickle=False)

    #show_image(wav)

    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(req.params.get('text'))
    #pprint(res.data)
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  print(hparams_debug_string())
  synthesizer.load(args.checkpoint)
  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
  synthesizer.load(os.environ['CHECKPOINT'])
