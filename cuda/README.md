# CUDA 版训练器

把 `main.cpp` 里的 `MNIST_DATA_mlp`（784 -> 256 -> 10，softmax + 交叉熵，朴素 SGD）
用一组手写 kernel 重写了一遍，用来回答一个问题：**同一套数学搬到 GPU 上，到底快多少、卡在哪**。

只用 cudart，不碰 cuBLAS / cuDNN —— 这样每一分性能都能归因到自己写的那几段代码上。

## 环境

| 组件 | 位置 / 版本 |
| --- | --- |
| GPU | RTX 4060 Laptop（SM 8.9，24 SM，8 GB，驱动 531.61） |
| nvcc | conda 环境 `E:\cpp_project\.cuda-env`，CUDA 12.6 |
| 宿主编译器 | VS2022 社区版 `cl.exe`（19.44） |
| Windows SDK | `E:\Windows Kits\10`（10.0.26100.0） |

nvcc 在 Windows 上只能用 MSVC 当宿主编译器，而且必须能同时找到 VC 头文件和
Windows SDK。本机 DevShell 探测 SDK 时用的注册表查询被安全策略拦掉了，所以
`build.ps1` 里手动补了 `INCLUDE` / `LIB`：

```powershell
pwsh -File cuda/build.ps1
```

（conda 装 nvcc 的命令，方便换机器复现：
`conda create -p .cuda-env -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6`）

## 跑

```bash
./cuda_mlp.exe                                  # 默认 10 epoch / batch 256 / lr 0.3
./cuda_mlp.exe --epochs 20 --batch 128 --lr 0.3
./cuda_mlp.exe --bench-gemm                     # 只测 GEMM 吞吐
python torch_baseline.py                        # PyTorch 对照（CPU + GPU）
```

数据仍读 `../MMISTdataset`，所以要在 `cuda/` 目录下执行。

## 实现了什么

| kernel | 作用 |
| --- | --- |
| `gemm_kernel<TA,TB>` | 64x64 分块、k 步长 16、每线程 4x4 微块；支持 A/B 转置 |
| `bias_relu_kernel` | 按列广播加偏置，可选 ReLU |
| `softmax_ce_kernel` | softmax + 交叉熵，同时产出 `dZ = (p - onehot) / batch` |
| `relu_mask_kernel` | 反传时按 `Z > 0` 屏蔽 |
| `col_sum_acc_kernel` | 偏置梯度（按列求和） |
| `sgd_kernel` | 参数更新 |
| `gather_batch_kernel` | 按设备端保存的 shuffle 顺序取批 |

一次训练步 = 取批 + 5 个 GEMM + 若干逐元素 kernel，共 11 次 launch。

## 优化过程（每一步都量了）

1. **朴素 32x32 分块、一线程一个输出**：GEMM 实测 200~290 GFLOPS（受笔记本降频影响，
   同一次运行里前后能差 3 倍，所以下面所有对比都是同一轮里紧挨着跑的）。
2. **换 64x64 分块 + 4x4 微块**（共享内存访存/计算比从 2:1 降到 1:2）→ 430~820 GFLOPS。
   这一步顺带修掉两个真 bug：标签没跟着 shuffle 走、偏置梯度 `+=` 后没清零。
3. **去掉每步两次 pageable H2D 拷贝**：WDDM 下这玩意儿单次几十到几百微秒，改成每个 epoch
   只传一次顺序数组，取批交给 GPU。
4. **CUDA Graph 回放**（关键的一步）：每步 11 次 launch，在 WDDM 上单次约三四十微秒，
   合计 ~0.4 ms/步，比 GEMM 本身还贵。把整步录成图之后每步只剩 1 次 `cudaGraphLaunch`。
   batch=256 时 epoch 从 362 ms 降到 150 ms，**2.4 倍**。

## 结果

MNIST，10 epoch：

| 实现 | batch | 训练耗时 | test acc |
| --- | --- | --- | --- |
| 手写 CUDA（含图回放） | 256 | **1.58 s** | 96.99% |
| 手写 CUDA | 1024 | 1.02 s | 86.6% |
| PyTorch + cuBLAS（GPU） | 1024 | 0.59 s | 94.36% |
| PyTorch（CPU） | 256 | 2.63 s | 97.41% |
| 本仓库 C++ 引擎（OpenMP 16 线程） | 64 | 11.70 s | 97.99% |

GEMM 单点吞吐（`--bench-gemm`）：

| 形状 | 时间 | GFLOPS |
| --- | --- | --- |
| 1024x256x784（前向 X·W1） | 0.50 ms | 824 |
| 784x256x1024（反向 dW1 = Xᵀ·dZ1） | 0.68 ms | 607 |
| 256x256x784（batch=256 前向） | 0.24 ms | 428 |
| 2048³（看这张卡的天花板） | 16.9 ms | 1015 |

CPU 侧同一个 matvec 热点循环大约是 47.9 GFLOPS，所以 **GEMM 这一层 GPU 快 10~20 倍**，
但端到端只快 7 倍出头。

## 结论

- 小模型上 GPU 的收益被**固定开销**吃掉：batch=256 时一步只有 51M 次乘加，算完只要几十微秒，
  而 11 次 kernel launch 就要 0.4 ms。这就是 CUDA Graph 带来 2.4 倍的原因。
- 手写 GEMM 离 cuBLAS 还有 3~5 倍差距（1015 GFLOPS vs 4060 上 cuBLAS 的几 TFLOPS），
  形状越小差距越大。想要再快，要么换 cuBLAS，要么把 micro-tile 提到 8x8 并做双缓冲。
- 准确率略低于 CPU 版（96.99% vs 97.99%）主要是更新步数少（batch 256 只有 234 步/epoch，
  CPU 版 batch 64 是 937 步/epoch），不是数值精度问题 —— 同一配置下 loss 曲线形状一致。
