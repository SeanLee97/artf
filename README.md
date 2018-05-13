# artf
a lightweight tensorflow library.

## functions
* mask
```python3
artf.mask(inputs, seq_len=None, mode='mul', mask_value=-1e12, name='mask', reuse=None)
```
* dense
```python3
artf.dense(inputs, ouput_size, bias=True, seq_len=None, name='dense', reuse=None)
```
* layer_norm
```python3
artf.layer_norm(inputs, size=None, epsilon=1e-6, name='layer_norm', reuse=None)
```
* group_norm
```python3
artf.group_norm(inputs, size=None, num_groups=8, epsilon=1e-5, name='group_norm', reuse=None)
```
* postion_embedding
```python3
artf.nn.postion_embdding(inputs, position_dim)
```
* glu
```python3
artf.nn.glu(x)
```
* leaky_relu
```python3
artf.nn.leaky_relu(x)
```
* conv
```python3
artf.nn.conv(inputs, out_size, bias=None, activation=None, kernel_size=1, name='conv', reuse=None)
```
## attention
```python3
artf.attention.multihead_attention(queries, keys, values, num_heads=8, num_units=None, bias=False,
                        dense_kernel=None, residual=False, Q_len=None, V_len=None, dropout=0.0,
                        name='multi_head_attnetion', reuse=None, is_training=False)
```
## highway
```python3
from artf.highway import Highway
highway = Highway(kernel='conv', size, name='highway', dropout, reuse=None)
#highway = Highway(kernel='fcn', size, name='highway', dropout, reuse=None)
#highway = Highway(kernel='fcn3d', size, name='highway', dropout, reuse=None)
inputs = highway(inputs)
```

## residual_block
```python3
inputs = artf.nn.residual_block(inputs,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 5,
                num_filters = self.config.hidden_dim,
                num_heads = self.config.num_heads,
                seq_len = inputs_len, # mask if seq_len not None 
                #seq_len = None,  # dont mask
                name = "residual_block",
                dropout = self.dropout)
```
