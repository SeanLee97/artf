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

#### bi_attention
```python
import artf.attention as attention

p2p, q2p = attentoin.ai_attention(p_enc, q_enc,
                                  p_mask, q_mask,
                                  kernel='bilinear', dropout=0.0)
```

#### dot_attention
```python
import artf.attention as attention
V_t, alpha = dot_attention(self, Q, K, V):
```

#### multihead attention
```python3
from artf.attention.multihead_attention import MultiheadAttention

# print help doc
MultiheadAttention.helper()
attention = MultiheadAttention(num_heads=8, dropout=0.0, )
output = attention(query, key, values, num_units=None,
                   query_mask=None, value_mask=None, residual=True,
                   scope="multihead_attention", reuse=None)
```

### transformer
Implement some modules in Transformer

#### encoder
```python
from artf.transformer import Encoder

transEnc = Encoder(num_heads=8,
                   num_blocks=4,
                   activation=tf.nn.relu,
                   dropout=0.0,
                   bias=False)
output = transEnc(inputs, num_units,
                  input_mask=None,
                  scope='transformer_encoder',
                  reuse=None)
```
#### decoder
```python
from artf.transformer import Decoder

transDec = Decoder(num_heads=8,
                               num_blocks=4,
                               activation=tf.nn.relu,
                               dropout=0.0,
                               bias=False)
output = transDec(inputs, encoder, num_units,
                  input_mask=None,
                  encoder_mask=None,
                  scope='transformer_encoder',
                  reuse=None)
```

### qanet
Implement some modules in QANet

#### residual block
```python
from artf.qanet import ResidualBlock

# print help doc
ResidualBlock.helper()

residual = ResidualBlock(num_heads=2,
                         num_blocks=4,
                         num_conv_layers=2,
                         activation=tf.nn.relu,
                         dropout=0.0,
                         bias=True)
kernel_size = 5
outputs = residual(inputs, kernel_size)

```

## Reference
- [transformer](https://github.com/Kyubyong/transformer)
- [QANet](https://github.com/NLPLearn/QANet)
