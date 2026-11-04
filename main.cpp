#include <iostream>
#include <random>
#include <numeric>
#include <chrono>

#include "Tensor.h"
#include "Net.h"
#include "utils.h"
#include "include/mnist/mnist_reader.hpp"

struct MNIST_DATA_set
{
	std::vector<std::vector<float>> training_images; ///< The training images
	std::vector<std::vector<float>> test_images;     ///< The test images
	std::vector<std::vector<float>> training_labels; ///< The training labels
	std::vector<std::vector<float>> test_labels;     ///< The test labels
};

MNIST_DATA_set read_MNIST_DATA()
{
	std::string MNIST_DATA_LOCATION = "../MMISTdataset";
	mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
		mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

	//std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
	//std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
	//std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	//std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	std::for_each(dataset.training_images.begin(), dataset.training_images.end(), [](auto& m)
	{
		std::for_each(m.begin(), m.end(), [](auto& v)
		{
			v = v / 255.0f;
		});
	});

	std::for_each(dataset.test_images.begin(), dataset.test_images.end(), [](auto& m)
	{
		std::for_each(m.begin(), m.end(), [](auto& v)
		{
			v = v / 255.0f;
		});
	});

	int train_num = dataset.training_images.size();
	int test_num = dataset.test_images.size();

	std::vector<std::vector<float>> training_labels(train_num, std::vector<float>(10, 0));
	for (int i = 0; i < train_num; ++i)
	{
		training_labels[i][dataset.training_labels[i]] = 1;
	}
	std::vector<std::vector<float>> test_labels(test_num, std::vector<float>(10, 0));
	for (int i = 0; i < test_num; ++i)
	{
		test_labels[i][dataset.test_labels[i]] = 1;
	}

	return { dataset.training_images, dataset.test_images, training_labels, test_labels };
}

// 线性回归
// ---------------------------------------------------------------------------
// 批量训练 / 评估
// 梯度按批大小取均值，所以 lr 是「每样本」尺度：旧版把一批的梯度直接求和，
// 等效学习率是 lr * batch。这里改成均值，等效学习率固定为 lr 本身。
// ---------------------------------------------------------------------------

// 训练一个 epoch：样本数按 batch 向下取整，尾部不足一批的样本跳过
static float train_one_epoch(Net& nn,
	const std::vector<std::vector<float>>& xs,
	const std::vector<std::vector<float>>& ys,
	const std::vector<int>& order, int batch, float learn_rate)
{
	std::vector<std::vector<float>> bx;
	std::vector<std::vector<float>> by;
	bx.reserve(batch);
	by.reserve(batch);

	float loss_sum = 0.0f;
	int steps = 0;
	const int total = static_cast<int>(order.size());
	for (int off = 0; off + batch <= total; off += batch)
	{
		bx.clear();
		by.clear();
		for (int k = 0; k < batch; ++k)
		{
			const int i = order[off + k];
			bx.emplace_back(xs[i]);
			by.emplace_back(ys[i]);
		}
		loss_sum += nn.train_batch(bx, by) * static_cast<float>(batch);
		nn.SGD(learn_rate);
		nn.zero_grad();
		++steps;
	}
	return steps > 0 ? loss_sum / static_cast<float>(steps * batch) : 0.0f;
}

// 分类评估：返回准确率，avg_loss 为按样本平均的损失
static float evaluate_classification(Net& nn,
	const std::vector<std::vector<float>>& xs,
	const std::vector<std::vector<float>>& ys, int batch, float& avg_loss)
{
	float loss_sum = 0.0f;
	int correct = 0;
	int seen = 0;
	const int total = static_cast<int>(xs.size());
	for (int off = 0; off + batch <= total; off += batch)
	{
		const std::vector<std::vector<float>> bx(xs.begin() + off, xs.begin() + off + batch);
		const std::vector<std::vector<float>> by(ys.begin() + off, ys.begin() + off + batch);
		loss_sum += nn.forward_batch(bx, by) * static_cast<float>(batch);
		for (int r = 0; r < batch; ++r)
		{
			const std::vector<float>& p = nn.get_y_hat_softmax(r);
			const int pred = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
			const int truth = static_cast<int>(std::max_element(by[r].begin(), by[r].end()) - by[r].begin());
			correct += (pred == truth) ? 1 : 0;
			++seen;
		}
	}
	avg_loss = seen > 0 ? loss_sum / static_cast<float>(seen) : 0.0f;
	return seen > 0 ? static_cast<float>(correct) / static_cast<float>(seen) : 0.0f;
}

// 回归评估：|y_hat - y| < tol 记为命中
static float evaluate_regression(Net& nn,
	const std::vector<std::vector<float>>& xs,
	const std::vector<std::vector<float>>& ys, int batch, float tol, float& avg_loss)
{
	float loss_sum = 0.0f;
	int hit = 0;
	int seen = 0;
	const int total = static_cast<int>(xs.size());
	for (int off = 0; off + batch <= total; off += batch)
	{
		const std::vector<std::vector<float>> bx(xs.begin() + off, xs.begin() + off + batch);
		const std::vector<std::vector<float>> by(ys.begin() + off, ys.begin() + off + batch);
		loss_sum += nn.forward_batch(bx, by) * static_cast<float>(batch);
		for (int r = 0; r < batch; ++r)
		{
			const float y_hat = nn.get_y_hat(r)[0];
			hit += (std::fabs(y_hat - by[r][0]) < tol) ? 1 : 0;
			++seen;
		}
	}
	avg_loss = seen > 0 ? loss_sum / static_cast<float>(seen) : 0.0f;
	return seen > 0 ? static_cast<float>(hit) / static_cast<float>(seen) : 0.0f;
}

// 线性回归
void linear_regression()
{
	// 均匀分布 U(0, 10)
	std::uniform_real_distribution<float> u_dist(0.0f, 10.0f);
	// 正态分布 N(0, 1)
	std::normal_distribution<float> n_dist(0.0f, 1.0f);

	int train_num = 40000;
	int test_num = 10000;

	std::vector<std::vector<float>> training_datas(train_num);
	std::vector<std::vector<float>> training_labels(train_num);
	std::vector<std::vector<float>> test_datas(test_num);
	std::vector<std::vector<float>> test_labels(test_num);

	float true_w1 = 2.0;
	float true_w2 = 3.0;
	float true_b = 7.0;
	for (int i = 0; i < train_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		// 按真实参数生成带噪声的标签
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		training_datas[i].emplace_back(x1);
		training_datas[i].emplace_back(x2);
		training_labels[i].emplace_back(y);
	}

	for (int i = 0; i < test_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		test_datas[i].emplace_back(x1);
		test_datas[i].emplace_back(x2);
		test_labels[i].emplace_back(y);
	}

	const int batch_size = 10;
	const float lr = 0.01f; // 梯度已按批取均值，等价于旧版的 0.001 * batch

	// 构建网络：2 -> 1，无激活
	Net nn(batch_size);
	nn.add_layer(2, 1, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::MEAN_SQUARED);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	for (int epoch = 0; epoch < 20; ++epoch)
	{
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());
		const float loss = train_one_epoch(nn, training_datas, training_labels, data_index_vec, batch_size, lr);
		nn.printParameters();
		std::cout << "loss: " << loss << std::endl;
	}

	float train_loss = 0.0f;
	float test_loss = 0.0f;
	const float train_acc = evaluate_regression(nn, training_datas, training_labels, batch_size, 2.0f, train_loss);
	const float test_acc = evaluate_regression(nn, test_datas, test_labels, batch_size, 2.0f, test_loss);
	std::cout << "train_acc: " << train_acc << ", train_loss: " << train_loss << std::endl;
	std::cout << "test_acc: " << test_acc << ", test_loss: " << test_loss << std::endl;
}

// 单层 softmax 回归：784 -> 10
void MNIST_DATA_softmax()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	const int batch_size = 64;
	const float lr = 0.1f;

	Net nn(batch_size);
	nn.add_layer(784, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<int> data_index_vec(dataset.training_images.size(), 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	for (int epoch = 0; epoch < 10; ++epoch)
	{
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());
		const float loss = train_one_epoch(nn, dataset.training_images, dataset.training_labels,
										   data_index_vec, batch_size, lr);
		std::cout << "loss: " << loss << std::endl;
	}

	float train_loss = 0.0f;
	float test_loss = 0.0f;
	const float train_acc = evaluate_classification(nn, dataset.training_images, dataset.training_labels,
													batch_size, train_loss);
	const float test_acc = evaluate_classification(nn, dataset.test_images, dataset.test_labels,
												   batch_size, test_loss);
	std::cout << "train_acc: " << train_acc << ", train_loss: " << train_loss << std::endl;
	std::cout << "test_acc: " << test_acc << ", test_loss: " << test_loss << std::endl;
}

// 识别手写数字（单隐层 MLP）：784 -> 256 -> 10
void MNIST_DATA_mlp()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	const int batch_size = 64;
	const float lr = 0.3f; // 均值梯度下的等效学习率（旧版是求和梯度，0.001*10）

	Net nn(batch_size);
	nn.add_layer(784, 256, FCLayer::ActivationFun::RELU);
	nn.add_layer(256, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<int> data_index_vec(dataset.training_images.size(), 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	for (int epoch = 0; epoch < 10; ++epoch)
	{
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());
		const float loss = train_one_epoch(nn, dataset.training_images, dataset.training_labels,
										   data_index_vec, batch_size, lr);
		std::cout << "loss: " << loss << std::endl;
	}

	float train_loss = 0.0f;
	float test_loss = 0.0f;
	const float train_acc = evaluate_classification(nn, dataset.training_images, dataset.training_labels,
													batch_size, train_loss);
	const float test_acc = evaluate_classification(nn, dataset.test_images, dataset.test_labels,
												   batch_size, test_loss);
	std::cout << "train_acc: " << train_acc << ", train_loss: " << train_loss << std::endl;
	std::cout << "test_acc: " << test_acc << ", test_loss: " << test_loss << std::endl;
}
// 跑一个 demo 并统计墙钟耗时（批量 + OpenMP 的收益要看得见）
static void run_demo(const char* name, void (*fn)())
{
	std::cout << "============== " << name << " ==============" << std::endl;
	const auto t0 = std::chrono::steady_clock::now();
	fn();
	const auto t1 = std::chrono::steady_clock::now();
	const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	std::cout << "[time] " << name << ": " << ms << " ms" << std::endl;
}

int main()
{
	run_demo("linear_regression", linear_regression);
	run_demo("MNIST_softmax", MNIST_DATA_softmax);
	run_demo("MNIST_mlp", MNIST_DATA_mlp);
	std::cout.flush();
	return 0;
}
