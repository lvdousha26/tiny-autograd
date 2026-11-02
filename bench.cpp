// A/B benchmark harness for the CppMLP framework.
// Trains an MLP (784-256-10) on a subset of MNIST and reports test accuracy.
// --buggy-loop reproduces the original training-loop bug
// (remain_data_num -= batch_size placed inside the inner k loop).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Tensor.h"
#include "Net.h"
#include "utils.h"
#include "include/mnist/mnist_reader.hpp"

struct Dataset
{
	std::vector<std::vector<float>> train_x, train_y, test_x, test_y;
};

static Dataset load_mnist(int train_n, int test_n)
{
	auto ds = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("../MMISTdataset");

	Dataset out;
	out.train_x.resize(train_n);
	out.train_y.assign(train_n, std::vector<float>(10, 0.0f));
	out.test_x.resize(test_n);
	out.test_y.assign(test_n, std::vector<float>(10, 0.0f));

	std::vector<int> idx(ds.training_images.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::mt19937 pick(12345);
	std::shuffle(idx.begin(), idx.end(), pick);
	for (int i = 0; i < train_n; ++i)
	{
		int s = idx[i];
		out.train_x[i] = ds.training_images[s];
		for (auto& v : out.train_x[i]) v /= 255.0f;
		out.train_y[i][ds.training_labels[s]] = 1.0f;
	}

	idx.resize(ds.test_images.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::shuffle(idx.begin(), idx.end(), pick);
	for (int i = 0; i < test_n; ++i)
	{
		int s = idx[i];
		out.test_x[i] = ds.test_images[s];
		for (auto& v : out.test_x[i]) v /= 255.0f;
		out.test_y[i][ds.test_labels[s]] = 1.0f;
	}
	return out;
}

static float evaluate(Net& nn, const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y)
{
	int correct = 0;
	for (size_t i = 0; i < X.size(); ++i)
	{
		nn.forward(X[i], Y[i]);
		const std::vector<float>& p = nn.get_y_hat_softmax();
		int pred = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
		int truth = static_cast<int>(std::max_element(Y[i].begin(), Y[i].end()) - Y[i].begin());
		if (pred == truth) ++correct;
	}
	return static_cast<float>(correct) / static_cast<float>(X.size());
}

int main(int argc, char** argv)
{
	int train_n = 8000, test_n = 2000, epochs = 3, batch = 10;
	float lr = 0.001f;
	bool buggy = false;
	for (int i = 1; i < argc; ++i)
	{
		if (std::strcmp(argv[i], "--buggy-loop") == 0) buggy = true;
		else if (std::strcmp(argv[i], "--train") == 0) train_n = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--test") == 0) test_n = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--epochs") == 0) epochs = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--lr") == 0) lr = std::stof(argv[++i]);
	}

	auto t0 = std::chrono::steady_clock::now();
	Dataset d = load_mnist(train_n, test_n);
	std::cerr << "[setup] train=" << train_n << " test=" << test_n
			  << " epochs=" << epochs << " batch=" << batch << " lr=" << lr
			  << " buggy_loop=" << (buggy ? 1 : 0) << std::endl;

	rand_generator().seed(20251102);

	Net nn;
	nn.add_layer(784, 256, FCLayer::ActivationFun::RELU);
	nn.add_layer(256, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<int> index_vec(train_n);
	std::iota(index_vec.begin(), index_vec.end(), 0);

	long long samples_seen = 0;
	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		std::shuffle(index_vec.begin(), index_vec.end(), rand_generator());
		float loss_sum = 0.0f;
		int remain = train_n;
		int pos = 0;
		while (remain > 0 && pos < train_n)
		{
			std::vector<std::vector<float>> bx, by;
			if (buggy)
			{
				for (int k = 0; k < std::min(batch, remain); ++k)
				{
					int i = index_vec[pos];
					bx.emplace_back(d.train_x[i]);
					by.emplace_back(d.train_y[i]);
					++pos;
					remain -= batch;
				}
			}
			else
			{
				int cur = std::min(batch, remain);
				bx.reserve(cur);
				by.reserve(cur);
				for (int k = 0; k < cur; ++k)
				{
					int i = index_vec[pos];
					bx.emplace_back(d.train_x[i]);
					by.emplace_back(d.train_y[i]);
					++pos;
				}
				remain -= cur;
			}
			if (bx.empty()) break;
			samples_seen += static_cast<long long>(bx.size());
			loss_sum += nn.train(bx, by);
			nn.SGD(lr);
			nn.zero_grad();
		}
		float train_acc = evaluate(nn, d.train_x, d.train_y);
		float test_acc = evaluate(nn, d.test_x, d.test_y);
		std::cout << "epoch " << (epoch + 1)
				  << "  samples=" << samples_seen
				  << "  loss=" << loss_sum / static_cast<float>(train_n)
				  << "  train_acc=" << train_acc
				  << "  test_acc=" << test_acc << std::endl;
	}

	auto t1 = std::chrono::steady_clock::now();
	double sec = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "RESULT samples_seen=" << samples_seen
			  << " test_acc=" << evaluate(nn, d.test_x, d.test_y)
			  << " sec=" << sec << std::endl;
	return 0;
}
