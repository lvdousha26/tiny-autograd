"""用 Python 包的 API 训练 MNIST（数据是仓库里现成的 IDX 文件）。

    python examples/train_mnist.py --data ../../MMISTdataset --train 8000 --epochs 5
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tiny_autograd as ta  # noqa: E402


def read_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0


def read_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def one_hot(y, classes=10):
    return np.eye(classes, dtype=np.float32)[y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "..", "MMISTdataset"))
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--backward", default="memo", choices=["stack", "memo"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ta.manual_seed(args.seed)

    x_train = read_images(os.path.join(args.data, "train-images-idx3-ubyte"))[: args.train]
    y_train = read_labels(os.path.join(args.data, "train-labels-idx1-ubyte"))[: args.train]
    x_test = read_images(os.path.join(args.data, "t10k-images-idx3-ubyte"))[: args.test]
    y_test = read_labels(os.path.join(args.data, "t10k-labels-idx1-ubyte"))[: args.test]

    # 打乱训练集
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]

    net = (
        ta.Net(batch=args.batch)
        .add_layer(784, args.hidden, "relu")
        .add_layer(args.hidden, 10, "none")
        .set_loss("cross_entropy")
        .set_backward(args.backward)
    )
    net.init_parameters(0.0, 0.02)

    t0 = time.time()
    history = ta.train(net, x_train, one_hot(y_train), epochs=args.epochs, lr=args.lr, seed=0)
    elapsed = time.time() - t0

    n = (len(x_train) // args.batch) * args.batch
    train_acc = (net.predict(x_train[:n]) == y_train[:n]).mean()
    test_acc = (net.predict(x_test) == y_test).mean()

    for i, loss in enumerate(history, 1):
        print(f"epoch {i:2d}  loss={loss:.6f}")
    print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  {elapsed:.2f}s "
          f"({len(x_train) * args.epochs / elapsed / 1000:.1f}k samples/s)")


if __name__ == "__main__":
    main()
