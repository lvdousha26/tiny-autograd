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

## 稀疏自编码器支持

引擎里新增了 `TOPK` 算子（逐行 ReLU 后只保留前 k 个正值）和隐层 L1 惩罚，
所以同一个 `Net` 可以直接拿来训 SAE，不必再手写梯度：

```python
import numpy as np
import tiny_autograd as ta

ta.manual_seed(0)
net = (ta.Net(batch=256)
         .add_topk_layer(64, 128, 4)     # 编码器：每行恰好 4 个特征点火
         .add_layer(128, 64, "none")    # 解码器：线性重构
         .set_loss("mse")               # 目标 y 就是输入 x
         .set_backward("memo"))
net.init_parameters(0.0, 0.1)
net.set_sparse_penalty(0.05, 0)         # 可选：给隐层加 L1（TopK 时一般不用）

net.train_batch(x[:256], x[:256])       # 一次前向 + 反向，返回平均损失
net.sgd(0.05)
net.zero_grad()

z = net.forward_hidden(x)               # 稀疏编码，任意样本数
x_hat = net.forward_output(x)            # 重构（原始输出，不做 softmax）
W_dec = net.get_weights(1)              # (d_in, n_features)，列即学到的方向
```

`forward_layer` / `forward_output` / `forward_hidden` 返回**原始**激活，
`predict_proba` 则返回 softmax 概率（只适用于交叉熵）。

相关接口：`add_topk_layer`、`set_sparse_penalty`、`layer_count`、`layer_dims`、
`forward_layer`、`forward_output`、`forward_hidden`、`get_weights` / `set_weights`、
`get_bias` / `set_bias`、`get_weight_grad`（读梯度做交叉校验）。

C++ 侧的 `sparse_demo.cpp` 对 TOPK 反传和 L1 梯度注入都做了中心差分校验。

## 注意

- 图是**静态**的，行数在 `ta.Net(batch=...)` 时固定；`train_batch` 的样本数必须等于 batch。
- `predict_proba` / `forward_layer` 会自动按 batch 切块前向，所以喂任意样本数都行。
- `predict_proba` 需要 `loss="cross_entropy"`（否则 softmax 不会被计算）。
- 目前只暴露了全连接网络 + TopK / LN 等算子；任意构图级别的图 API 还没做 Python 绑定。
