# tiny_autograd (Python 封装)

把仓库根目录的 C++ 计算图引擎（`Tensor` / `FCLayer` / `Net`）通过 **C ABI + ctypes**
暴露成 Python 包。不走 pybind11，所以不需要 MSVC，MinGW 的 g++ 就能编。

## 构建与安装

```bash
cd python
pip install .            # 构建期调用 g++ 编出 tiny_autograd.dll
# 或只编原生库，便于本地调试：
python setup.py build_native
```

依赖：`numpy`；编译需要 `g++`（默认取 PATH 里的，可用 `CXX=...` 覆盖）。
编出来的 DLL 会连带 `libgomp-1.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll` /
`libdl.dll` 一起放进包里，所以安装后自包含。

## 用法

```python
import numpy as np
import tiny_autograd as ta

ta.manual_seed(0)                                    # 参数初始化可复现
net = (ta.Net(batch=128)                             # batch 决定一次前向的行数
         .add_layer(784, 256, "relu")
         .add_layer(256, 10, "none")
         .set_loss("cross_entropy")
         .set_backward("memo"))                      # "stack" | "memo"
net.init_parameters(0.0, 0.02)

ta.train(net, x_train, y_onehot, epochs=5, lr=0.12)
print(net.predict(x_test))
```

跑真实 MNIST：

```bash
python examples/train_mnist.py --train 8000 --epochs 12 --lr 0.12 --seed 7
```

## 注意

- 图是**静态**的，行数在 `ta.Net(batch=...)` 时固定；`train_batch` 的样本数必须等于 batch。
- `predict_proba` 会自动按 batch 切块前向，所以喂任意样本数都行。
- `predict_proba` 需要 `loss="cross_entropy"`（否则 softmax 不会被计算）。
- 目前只暴露了全连接网络；注意力/算子级的图 API 还没做 Python 绑定。
