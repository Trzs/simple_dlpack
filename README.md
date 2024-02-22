# simple_dlpack

Simple example script that shows how to use [DLPack](https://dmlc.github.io/dlpack/latest/) to make C++ arrays accessible for Numpy, Pytorch & Co.

To build it, run:

```
git clone https://github.com/Trzs/simple_dlpack.git
cd simple_dlpack
git clone https://github.com/pybind/pybind11.git
mkdir build
cd build
cmake ..
make
```

How to use it:
```
>>> import simple_dlpack
>>> import torch
>>> import numpy as np
>>> A = simple_dlpack.simple_array()
>>> x = torch.from_dlpack(A)
>>> x[2] = 51.25
>>> A.print()
C = [ 0 0 51.25 0 0 0 0 0 0 0 0 0 ]
>>> A.set(1, 3.1145)
>>> print(x)
tensor([ 0.0000,  3.1145, 51.2500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000], dtype=torch.float64)
>>> y = np.from_dlpack(A)
>>> y
array([ 0.    ,  3.1145, 51.25  ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ])
>>> # But numpy is read-only!
>>> y[5] = 1.23
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: assignment destination is read-only
>>>
```
