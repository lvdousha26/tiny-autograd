"""PyTorch 对照基线：同样的 784-256-10 MLP、同样的数据、同样的 SGD。

用途有两个：
1. 正确性对账 —— 手写 CUDA kernel（cuda_mlp.cu）和 PyTorch 在同一配置下应该收敛到
   同一个准确率量级，差得多就说明 kernel 有问题。
2. 性能上限 —— cuBLAS + cuDNN 这套高度优化的实现能跑多快，可以当作「手写 kernel
   离天花板还有多远」的参照。

用法：
    python torch_baseline.py              # CPU 和 GPU 都跑
    python torch_baseline.py --device cuda
"""

import argparse
import struct
import time

import torch


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, path
        buf = f.read(n * rows * cols)
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
    return t.view(n, rows * cols).float() / 255.0


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, path
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone().long()


def load(data_dir):
    xtr = read_idx_images(f"{data_dir}/train-images-idx3-ubyte")
    ytr = read_idx_labels(f"{data_dir}/train-labels-idx1-ubyte")
    xte = read_idx_images(f"{data_dir}/t10k-images-idx3-ubyte")
    yte = read_idx_labels(f"{data_dir}/t10k-labels-idx1-ubyte")
    return xtr, ytr, xte, yte


def build(device):
    torch.manual_seed(20251104)
    net = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )
    # 和 C++ / CUDA 版一致的初始化：uniform(-0.02, 0.02)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.02, 0.02)
            torch.nn.init.zeros_(m.bias)
    return net.to(device)


def run(device, batch, lr, epochs, data):
    xtr, ytr, xte, yte = data
    net = build(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    xtr_d, ytr_d = xtr.to(device), ytr.to(device)
    xte_d, yte_d = xte.to(device), yte.to(device)

    n = xtr_d.shape[0]
    steps = n // batch
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    last_loss = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(steps):
            idx = perm[s * batch : (s + 1) * batch]
            xb, yb = xtr_d[idx], ytr_d[idx]
            logits = net(xb)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = loss.item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    net.eval()
    with torch.no_grad():
        correct = 0
        for off in range(0, xte_d.shape[0], 1000):
            pred = net(xte_d[off : off + 1000]).argmax(dim=1)
            correct += (pred == yte_d[off : off + 1000]).sum().item()
        test_acc = correct / xte_d.shape[0]

        tr_correct = 0
        for off in range(0, n, 1000):
            pred = net(xtr_d[off : off + 1000]).argmax(dim=1)
            tr_correct += (pred == ytr_d[off : off + 1000]).sum().item()
        train_acc = tr_correct / n

    print(
        f"[{device}] batch={batch:4d} lr={lr} epochs={epochs}  "
        f"loss={last_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
        f"train={train_s * 1000:.1f} ms"
    )
    return train_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../MMISTdataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    args = ap.parse_args()

    data = load(args.data)
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    devices = []
    if args.device in ("cpu", "both"):
        devices.append(torch.device("cpu"))
    if args.device in ("cuda", "both") and torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        if dev.type == "cuda":
            run(dev, 256, 0.3, args.epochs, data)
            run(dev, 1024, 0.3, args.epochs, data)
        else:
            run(dev, 64, 0.3, args.epochs, data)
            run(dev, 256, 0.3, args.epochs, data)


if __name__ == "__main__":
    main()
