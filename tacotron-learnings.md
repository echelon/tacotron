Tacotron Learnings
==================

Example Sentences
-----------------

1. testing generation of audio from my own instance of taco tron machine learning
2. donald trump is the president of the united states of america
3. the best selling nintendo video game franchise is pokemon pocket monsters
4. hi hello there goodbye now I am man bear pig fighter of evil and defender of democracy
5. Japan is an island country in East Asia Located in the Pacific Ocean5
6. 123456 dollars at 12:00 pm
7. Exiting due to exception: Failed to get convolution algorithm
8. I want to be the very best, Like no one ever was, To catch them is my real test, To train them is my cause!

Installing Tensorflow, CUDA, etc.
---------------------------------
Tacotron won't work because Tensorflow won't. It requires CUDA 9.0, not 9.1 on Ubuntu 18.04 LTS (as of 2019-02-11).

After trying and failing for a long time, I found this guide extremely helpful:

https://yangcha.github.io/CUDA90/

Force installing these (with a hammer)

Sometimes `sudo dpkg -i --force-overwrite` when things won't install over old versions.

Once this works, verify by `python3` -> `import tensorflow` working.

Graphics might mess up.

1. Open Ubuntu's "Software and Updates" -> "Additional Drivers"
2. Select "Using NVIDIA driver metapackage from nvidia-driver-390 (proprietary, tested)" and reboot. 

Running Tacotron
----------------

- Downloaded ljspeech dataset
- preprocess.py is smooth.

Exiting due to exception: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.

https://github.com/tensorflow/tensorflow/issues/24828

pip3 install --upgrade tensorflow-gpu==1.8.0

But this causes,

Something is wrong with the numpy installation. While importing we detected an older version of numpy in

```
mv /home/bt/.local/lib/python3.6/site-packages/numpy ~/.bad_numpy
pip3 install --upgrade numpy
```

Then,

```
bt@halide:~/dev/3rd/Tacotron-2$ pip3 uninstall "tensorflow==1.7.*"

  Successfully uninstalled tensorflow-1.12.0

```

import tensorflow works again

training fails with errors.
preprocess again fails.

git clone again (to start fresh) since it's hard to tell where state is being saved. (large gitignore)
install requirements.txt
move dataset to new, clean project
re-preprocess (no errors)

train.py still fails,

```
WARNING:tensorflow:From /home/bt/.local/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py:497: calling conv1d (from t
ensorflow.python.ops.nn_ops) with data_format=NHWC is deprecated and will be removed in a future version.
Instructions for updating:                                                                    
`NHWC` for data_format is deprecated, use `NWC` instead                                               
Traceback (most recent call last):                                        
  File "train.py", line 138, in <module>                                    
    main()                                                                          
  File "train.py", line 132, in main                                                        
    train(args, log_dir, hparams)                                                   
  File "train.py", line 52, in train                                                      
    checkpoint = tacotron_train(args, log_dir, hparams)                         
  File "/home/bt/dev/3rd/Tacotron-2/tacotron/train.py", line 399, in tacotron_train     
    return train(log_dir, args, hparams)                                          
  File "/home/bt/dev/3rd/Tacotron-2/tacotron/train.py", line 156, in train                
    model, stats = model_train_mode(args, feeder, hparams, global_step)         
  File "/home/bt/dev/3rd/Tacotron-2/tacotron/train.py", line 87, in model_train_mode    
    is_training=True, split_infos=feeder.split_infos)                         
  File "/home/bt/dev/3rd/Tacotron-2/tacotron/models/tacotron.py", line 170, in initialize
    CustomDecoder(decoder_cell, self.helper, decoder_init_state),                                    
  File "/home/bt/dev/3rd/Tacotron-2/tacotron/models/custom_decoder.py", line 43, in __init__                 
    rnn_cell_impl.assert_like_rnncell(type(cell), cell)                                  
AttributeError: module 'tensorflow.python.ops.rnn_cell_impl' has no attribute 'assert_like_rnncell'
```

Versions,

```
bt@halide:~/dev/3rd/Tacotron-2$ ls -l /usr/local/cuda-9.0/lib64/libcud*
-rw-r--r-- 1 root root 638208 Sep  2  2017 /usr/local/cuda-9.0/lib64/libcudadevrt.a
lrwxrwxrwx 1 root root     16 Sep  2  2017 /usr/local/cuda-9.0/lib64/libcudart.so -> libcudart.so.9.0
lrwxrwxrwx 1 root root     20 Sep  2  2017 /usr/local/cuda-9.0/lib64/libcudart.so.9.0 -> libcudart.so.9.0.176
-rw-r--r-- 1 root root 442392 Sep  2  2017 /usr/local/cuda-9.0/lib64/libcudart.so.9.0.176
-rw-r--r-- 1 root root 830686 Sep  2  2017 /usr/local/cuda-9.0/lib64/libcudart_static.a

python -c "import tensorflow; print(tensorflow.__version__)"
1.7.1
```

System Audio
------------

DVI -> Monitor -> Audio Jack Speaker works in WIndows, but not Ubuntu
HDMI -> Monitor -> Audio Jack works in Linux, but it's difficult to configure:

1. Maybe some magic in alamixer? Unmute things.
2. pavucontrol (bulk of magic) 
  1. Configuration : Built-in Audio -> off
  2. Configuration : GP102 HDMI Audio Controller -> Digital Stereo (HDMI 2) Output
  3. Output Devices : HDMI / DisplayPort 2 (plugged in)

Unclamp audio from 100% to get rid of static

tacotron (github: keithito/tacotron)
----------------

prepare works ! 
training works !
this one does stuff.

python train.py --base_dir=/home/bt/dev/3rd/tacotron

```
Step 995     [3.684 sec/step, loss=0.16315, avg_loss=0.17145]
Step 996     [3.687 sec/step, loss=0.16898, avg_loss=0.17140]
Step 997     [3.683 sec/step, loss=0.17208, avg_loss=0.17137]
Step 998     [3.663 sec/step, loss=0.17030, avg_loss=0.17133]
Step 999     [3.660 sec/step, loss=0.17126, avg_loss=0.17129]
Step 1000    [3.691 sec/step, loss=0.17046, avg_loss=0.17128]
Writing summary at step: 1000
Saving checkpoint to: /home/bt/dev/3rd/tacotron/logs-tacotron/model.ckpt-1000
Saving audio and alignment...
/home/bt/.local/lib/python3.6/site-packages/librosa/util/utils.py:1725: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  if np.issubdtype(x.dtype, float) or np.issubdtype(x.dtype, complex):
Input: she testified that she told him, quote,~__
Step 1001    [3.702 sec/step, loss=0.17063, avg_loss=0.17124]
```

 python demo_server.py --checkpoint=/home/bt/dev/3rd/tacotron/logs-tacotron/model.ckpt-1000

After 1000 iterations, it sounds like an alien machine attacking. Gross.

I'm assuming the other model did 20,180,906 iterations instead.

Also, I need the 3x speed boost of TCMalloc (trying now)

sudo apt-get install libtcmalloc-minimal4

Also, there's this fork:

https://github.com/MycroftAI/mimic2

Also, 

sudo apt-get install google-perftools

Then,

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train.py --base_dir=/home/bt/dev/3rd/tacotron

Might need to build Tensorflow from source. Not getting the 1sec/iter reported

Restoring,

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train.py --base_dir=/home/bt/dev/3rd/tacotron --restore_step=1000

As it turns out, GPU might not be used per:

- `nvidia-smi` shows no GPU load
- `htop` shows CPU load

https://stackoverflow.com/a/52627105 reports that even with tensorflow-gpu, cuda and cudnn must be installed

```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176

cat /usr/local/cuda/version.txt
CUDA Version 9.0.176
```


Even,

```
find /usr | grep cudnn
/usr/share/lintian/overrides/libcudnn7
/usr/share/lintian/overrides/libcudnn7-dev
/usr/share/doc/libcudnn7
/usr/share/doc/libcudnn7/copyright
/usr/share/doc/libcudnn7/NVIDIA_SLA_cuDNN_Support.txt
/usr/share/doc/libcudnn7/changelog.Debian.gz
/usr/share/doc/libcudnn7-dev
/usr/share/doc/libcudnn7-dev/copyright
/usr/share/doc/libcudnn7-dev/changelog.Debian.gz
/usr/lib/x86_64-linux-gnu/libcudnn.so.7.0.5
/usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a
/usr/lib/x86_64-linux-gnu/libcudnn.so.7
/usr/lib/x86_64-linux-gnu/libcudnn_static.a
/usr/lib/x86_64-linux-gnu/libcudnn.so
/usr/include/cudnn.h
/usr/include/x86_64-linux-gnu/cudnn_v7.h
```

Maybe try: https://github.com/tensorflow/tensorflow/issues/12388#issuecomment-452650768

https://stackoverflow.com/questions/41402409/tensorflow-doesnt-seem-to-see-my-gpu

Yep, fails:

>>> tf.test.gpu_device_name()
''

Going to try reinstalling tensorflow

> pip uninstall tensorflow
>  Successfully uninstalled tensorflow-1.7.1

> pip uninstall tensorflow-gpu
>  Successfully uninstalled tensorflow-gpu-1.12.0

Numpy broke.

> pip uninstall numpy
>   Successfully uninstalled numpy-1.16.1


THis shit is broooke

  Successfully uninstalled tensorflow-gpu-1.12.0

TRY THESE:

> mv /home/bt/.local/lib/python3.6/site-packages/tensorflow ~/.bad_tensorflow
> mv /home/bt/.local/lib/python3.6/site-packages/tensorflow_gpu-1.8.0.dist-info/ ~/.badtensorflow_gpu
> pip3 install --upgrade tensorflow-gpu==1.8.0
> pip3 install --upgrade numpy

python -c "import tensorflow as tf" -- WORKS

AND NOW IT SEES MY GPU! HURRAH!

```
>>> import tensorflow as tf
>>> tf.test.gpu_device_name()
2019-02-12 03:09:04.596398: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-02-12 03:09:04.877553: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.6705
pciBusID: 0000:65:00.0
totalMemory: 10.91GiB freeMemory: 10.39GiB
2019-02-12 03:09:04.877584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2019-02-12 03:09:05.077669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-02-12 03:09:05.077705: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2019-02-12 03:09:05.077713: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2019-02-12 03:09:05.077899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/device:GPU:0 with 10058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:65:00.0, compute capability: 6.1)
'/device:GPU:0'
```

Resuming training, hopefully not at 3 - 6 seconds per step

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train.py --base_dir=/home/bt/dev/3rd/tacotron --restore_step=1000


WAHAHAHA,

```
nvidia-smi 
Tue Feb 12 03:12:39 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.77                 Driver Version: 390.77                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:65:00.0  On |                  N/A |
| 39%   62C    P2    99W / 250W |  10803MiB / 11175MiB |     74%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1410      G   /usr/lib/xorg/Xorg                            20MiB |
|    0      1449      G   /usr/bin/gnome-shell                          49MiB |
|    0      1738      G   /usr/lib/xorg/Xorg                           128MiB |
|    0      1894      G   /usr/bin/gnome-shell                         168MiB |
|    0     29705      C   python3                                    10431MiB |
+-----------------------------------------------------------------------------+
```

Double WAHAHAHA,

```
Step 1094    [1.904 sec/step, loss=0.16965, avg_loss=0.16972]
Step 1095    [1.902 sec/step, loss=0.16756, avg_loss=0.16970]
Step 1096    [1.901 sec/step, loss=0.16741, avg_loss=0.16967]
Step 1097    [1.894 sec/step, loss=0.17295, avg_loss=0.16971]
Step 1098    [1.903 sec/step, loss=0.16909, avg_loss=0.16970]
Step 1099    [1.910 sec/step, loss=0.16443, avg_loss=0.16965]
Step 1100    [1.908 sec/step, loss=0.16872, avg_loss=0.16964]
```

Better than the old values:

```
Step 1001    [7.826 sec/step, loss=0.17147, avg_loss=0.17147]
Step 1002    [4.815 sec/step, loss=0.16660, avg_loss=0.16903]
Step 1003    [4.702 sec/step, loss=0.17020, avg_loss=0.16942]
Step 1004    [5.016 sec/step, loss=0.16975, avg_loss=0.16951]
Step 1005    [5.035 sec/step, loss=0.17291, avg_loss=0.17019]
Step 1006    [4.685 sec/step, loss=0.17192, avg_loss=0.17047]
Step 1007    [4.493 sec/step, loss=0.17012, avg_loss=0.17042]
Step 1008    [4.379 sec/step, loss=0.17148, avg_loss=0.17056]
Step 1009    [4.535 sec/step, loss=0.16747, avg_loss=0.17021]
Step 1010    [4.287 sec/step, loss=0.16556, avg_loss=0.16975]
Step 1011    [4.408 sec/step, loss=0.17196, avg_loss=0.16995]
Step 1012    [4.446 sec/step, loss=0.17125, avg_loss=0.17006]
```

But can probably still be improved.,

Microphone and Audacity Setup
-----------------------------

Mic -> (cable to input "1") -> Zoom H6 -> (3.1 mic jack) -> tower top jack

pavucontrol -> 
1. built-in audio -> analog sterio input (unplugged)
2. input devices -> front microphone (unplugged)

This gets it recording waveforms in audacity

Audacity input/output GUI:
- ALSA
- HDA INtel PCH Generic Analog - I think hw0,0 front mic 0
- 2 (stero)  recording channels
- HDA NVidia: HDMI 1


Use USB mode to interface H6 instead:

https://www.sweetwater.com/sweetcare/articles/using-the-zoom-h6-as-an-audio-interface/

Volume level is ~4 or 5 on audacity's mic input slider

## 2019-02-12: Training my own data

python preprocess.py --dataset=btspeech --base_dir=/home/bt/dev/3rd/personal/tacotron

tacotron failed to preprocess, so I reinstalled requirements.txt. Works.

train failed due to "shapes" erorr,

https://github.com/keithito/tacotron/issues/47

set hparams max_iters

### Prepare:

python preprocess.py --dataset=btspeech --base_dir=/home/bt/dev/3rd/personal/tacotron

### Train:

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train.py --base_dir=/home/bt/dev/3rd/personal/tacotron --hparams="max_iters=400" --restore_step=11000

### Serve: 

`python demo_server.py --checkpoint=/home/bt/dev/3rd/personal/tacotron/logs-tacotron/model.ckpt-11000`


#### Training Attempt "1":

- 7:08 of btaudio (45 wav files)

Sounds awful.

#### Training Attempt "2":

- 11:51 of btaudio (70 wav files)

Wrote 70 utterances, 56930 frames (0.20 hours)
Max input length:  328
Max output length: 1534

