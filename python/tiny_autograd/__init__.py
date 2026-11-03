"""tiny_autograd —— 把 C++ 计算图引擎（Tensor / FCLayer / Net）暴露成 Python API。

底层是同一份 C++ 源码编出来的 DLL，通过 C ABI + ctypes 调用，
不依赖 pybind11，也不需要 MSVC。
"""

import ctypes
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# libgomp / libgcc 是 OpenMP 与 C++ 运行时的依赖，随包一起分发
for _dep in ("libdl.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll", "libgomp-1.dll"):
    _p = os.path.join(_DIR, _dep)
    if os.path.exists(_p):
        try:
            ctypes.WinDLL(_p)
        except OSError:
            pass

_LIB_PATH = os.path.join(_DIR, "tiny_autograd.dll")
if not os.path.exists(_LIB_PATH):  # pragma: no cover
    raise ImportError(
        "找不到 tiny_autograd.dll，请先构建：python setup.py build_ext "
        "（或 pip install .）"
    )

_lib = ctypes.CDLL(_LIB_PATH)

_lib.ta_seed.restype = None
_lib.ta_seed.argtypes = [ctypes.c_uint]

_lib.ta_last_error.restype = ctypes.c_char_p
_lib.ta_last_error.argtypes = []

_lib.ta_create.restype = ctypes.c_void_p
_lib.ta_create.argtypes = [ctypes.c_int]

_lib.ta_destroy.restype = None
_lib.ta_destroy.argtypes = [ctypes.c_void_p]

_lib.ta_add_layer.restype = ctypes.c_int
_lib.ta_add_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.ta_set_loss.restype = ctypes.c_int
_lib.ta_set_loss.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.ta_init_params.restype = None
_lib.ta_init_params.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]

_lib.ta_set_backward.restype = ctypes.c_int
_lib.ta_set_backward.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.ta_in_dim.restype = ctypes.c_int
_lib.ta_in_dim.argtypes = [ctypes.c_void_p]

_lib.ta_out_dim.restype = ctypes.c_int
_lib.ta_out_dim.argtypes = [ctypes.c_void_p]

_lib.ta_train_batch.restype = ctypes.c_float
_lib.ta_train_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

_lib.ta_sgd.restype = None
_lib.ta_sgd.argtypes = [ctypes.c_void_p, ctypes.c_float]

_lib.ta_zero_grad.restype = None
_lib.ta_zero_grad.argtypes = [ctypes.c_void_p]

_lib.ta_forward_probs.restype = ctypes.c_int
_lib.ta_forward_probs.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]

_ACTIVATIONS = {"none": 0, "relu": 1}
_LOSSES = {"mse": 0, "cross_entropy": 1, "ce": 1}
_BACKWARD = {"stack": 0, "memo": 1}


def _error() -> str:
    raw = _lib.ta_last_error()
    return raw.decode("utf-8", "replace") if raw else "unknown error"


def _as_f32(a) -> np.ndarray:
    arr = np.ascontiguousarray(a, dtype=np.float32)
    return arr


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def manual_seed(seed: int) -> None:
    """固定 C++ 侧的随机数发生器，让参数初始化可复现。"""
    _lib.ta_seed(int(seed))


class Net:
    """一个全连接网络。batch 决定一次前向处理多少样本，构建后不可变。"""

    def __init__(self, batch: int = 1):
        self._h = _lib.ta_create(int(batch))
        if not self._h:
            raise RuntimeError(_error())
        self._batch = int(batch)
        self._built = False

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.ta_destroy(h)
            self._h = None

    def __repr__(self):
        return f"<tiny_autograd.Net batch={self._batch} in={self.in_dim} out={self.out_dim}>"

    def _check(self, rc: int):
        if rc != 0:
            raise RuntimeError(_error())

    def add_layer(self, num_inputs: int, num_outputs: int, activation: str = "relu") -> "Net":
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation 只支持 {list(_ACTIVATIONS)}")
        self._check(_lib.ta_add_layer(self._h, num_inputs, num_outputs, _ACTIVATIONS[activation]))
        self._built = True
        return self

    def set_loss(self, loss: str = "cross_entropy") -> "Net":
        if loss not in _LOSSES:
            raise ValueError(f"loss 只支持 {list(_LOSSES)}")
        self._check(_lib.ta_set_loss(self._h, _LOSSES[loss]))
        return self

    def init_parameters(self, mean: float = 0.0, std: float = 0.02) -> "Net":
        _lib.ta_init_params(self._h, mean, std)
        return self

    def set_backward(self, mode: str = "stack") -> "Net":
        if mode not in _BACKWARD:
            raise ValueError(f"mode 只支持 {list(_BACKWARD)}")
        self._check(_lib.ta_set_backward(self._h, _BACKWARD[mode]))
        return self

    @property
    def in_dim(self) -> int:
        return _lib.ta_in_dim(self._h)

    @property
    def out_dim(self) -> int:
        return _lib.ta_out_dim(self._h)

    def train_batch(self, x, y) -> float:
        """一次前向 + 反向，返回这一批的平均损失。样本数必须等于 batch。"""
        xa = _as_f32(x)
        ya = _as_f32(y)
        if xa.shape[0] != self._batch:
            raise ValueError(f"样本数 {xa.shape[0]} 与 batch {self._batch} 不一致")
        loss = _lib.ta_train_batch(self._h, _ptr(xa), _ptr(ya), int(xa.shape[0]))
        if loss < 0:
            raise RuntimeError(_error())
        return float(loss)

    def sgd(self, lr: float) -> None:
        _lib.ta_sgd(self._h, float(lr))

    def zero_grad(self) -> None:
        _lib.ta_zero_grad(self._h)

    def predict_proba(self, x) -> np.ndarray:
        """前向并返回 softmax 概率（需要 loss="cross_entropy"）。

        图是按固定 batch 建的，所以按 batch 切块前向。
        """
        xa = _as_f32(x)
        n = xa.shape[0]
        out = np.zeros((n, self.out_dim), dtype=np.float32)
        for off in range(0, n, self._batch):
            chunk = np.ascontiguousarray(xa[off: off + self._batch])
            m = chunk.shape[0]
            buf = np.zeros((m, self.out_dim), dtype=np.float32)
            self._check(_lib.ta_forward_probs(self._h, _ptr(chunk), m, _ptr(buf)))
            out[off: off + m] = buf
        return out

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def train(
    net: Net,
    x,
    y,
    epochs: int = 1,
    lr: float = 0.01,
    shuffle: bool = True,
    seed: int = 0,
    verbose: bool = False,
):
    """按 batch 切片训练，返回每个 epoch 的平均损失。"""
    xa = _as_f32(x)
    ya = _as_f32(y)
    n = xa.shape[0]
    batch = net._batch
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        total = 0.0
        steps = 0
        for off in range(0, n - batch + 1, batch):
            idx = order[off: off + batch]
            loss = net.train_batch(xa[idx], ya[idx])
            net.sgd(lr)
            net.zero_grad()
            total += loss
            steps += 1
        avg = total / max(steps, 1)
        history.append(avg)
        if verbose:
            print(f"epoch {epoch + 1}: loss={avg:.6f}")
    return history


__all__ = ["Net", "train", "manual_seed"]
