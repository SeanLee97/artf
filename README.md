<p align="center">/ artf /</p>

<p align="center">A lightweight tensorflow library.</p>

<p align="center">轻量级tensorflow库</p>

## Overview
```
project
│   README.md
│   LICENSE    
│
└───artf
│   │   __init__.py   # default functions lib
│   │   conv.py       # convolution interface 
|   |   highway.py    # highway network interface
|   |   loss.py       # loss functions
|   |   rnn.py        # rnn interface
│   │
│   └───attention
│       │   __init__.py    # default attention functions
│       │   multihead_attention.py
│       │   ...
│   
└───test    # modules test
    │   ...
```

## Modules

### conv
provide a unified interface for convolution operation.

#### usage
```python
from artf.conv import Conv

# print the helper doc
Conv.helper()

# define a conv instance
conv = Conv(activation=None, kernel_size=1, bias=None)
# convolution operation with the defined instance
output = conv(inputs, output_size, scope='conv_encoder', reuse=None)
```

### highway
implement the highway network, support 3 types kernels:
- fcn: fully-connect
- fcn3d: fully-connect (it performs better than the fcn if the input tensor`s dimension eq 3)
- conv: convolution

#### usage
```python
from artf.highway import Highway

# print the helper doc
Highway.helper()

highway = Highway(activation=None, kernel='conv', num_layers=2, dropout=0.0)
output = highway(inputs, scope='highway', reuse=None)
```

### rnn
provide a unified multilayers-rnn interface, support 2 kernels:
- lstm: return 3 values(concat_ouputs, last_c, last_hidden) if the kernel is lstm
- gru: return 2 values(concat_outputs, last_c) if the kernel is gru

it contains 4 classes:
- RNN: single-directional rnn
- BiRNN: binary-directional rnn
- CudnnRNN: single-directional rnn, faster than RNN, only support GPU to **train or test**
- BiCudnnRNN: binary-directional rnn, faster than BiRNN, only support GPU to **train or test**

#### usage
because the 4 classes have the same way to use, I only give an example.
```python
from artf.rnn import BiRNN

# print the helper doc
BiRNN.helper()

# kernel is lstm
rnn = BiRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm')
output, last_c, last_hidden = rnn(inputs,
                                  seq_len=None,
                                  batch_first=True,
                                  scope='bidirection_rnn',
                                  reuse=None)

# kernel is gru
rnn = BiRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='gru')
output, last_c = rnn(inputs,
                     seq_len=None,
                     batch_first=True,
                     scope='bidirection_rnn',
                     reuse=None)
```

### loss
implement some loss functions:

- spread_loss(labels, activations, margin)
- cross_entropy(y, logits)
- margin_loss(y, logits)
- bin_focal_loss(y, logits, weights=None, alpha=0.5, gamma=2): for binary labels
- focal_loss(y, logits, gamma=2, epsilon=1e-10): for multiply labels

#### usage
```python
import artf.loss as loss

l = loss.focal_loss(true_label, logits)
```

### attention
