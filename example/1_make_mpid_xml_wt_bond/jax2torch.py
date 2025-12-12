#!/usr/bin/env python
import sys
import numpy as np
import pickle
import torch
import warnings
warnings.filterwarnings("ignore")
import jax.numpy as jnp
from collections.abc import Mapping


# 递归转换函数
def jax_to_torch(obj):
    if isinstance(obj, Mapping):
        return {k: jax_to_torch(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [jax_to_torch(v) for v in obj]
    elif isinstance(obj, jnp.ndarray):
        return torch.from_numpy(np.array(obj, dtype=np.float32))
    elif np.isscalar(obj):  # 处理 float32、int32 等标量
        return torch.tensor(obj, dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

if __name__ == '__main__':
    params_sgnn_jax = pickle.load(open("params_sgnn_992.pickle", "rb"))
    # print(params_sgnn_jax)

    pytorch_params = jax_to_torch(params_sgnn)
    # print(f"pytorch_params: {pytorch_params}")

    with open("params_sgnn_torch.pickle", "wb") as f:
        pickle.dump(pytorch_params, f)
