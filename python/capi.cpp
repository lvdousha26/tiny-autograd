// C ABI 导出层：供 Python ctypes 调用，异常一律在内部转成错误码
#include <cstring>
#include <string>
#include <vector>

#include "../Graph.h"
#include "../Net.h"
#include "../utils.h"

#ifdef _WIN32
#define TA_API __declspec(dllexport)
#else
#define TA_API
#endif

namespace
{
	struct Handle
	{
		Net net;
		int batch = 1;
		int in_dim = 0;
		int out_dim = 0;
		bool loss_set = false;
		explicit Handle(int b) : net(b), batch(b) {}
	};

	std::string g_error;

	void set_error(const char* msg)
	{
		g_error = msg;
	}

	// 把 (n x cols) 的平铺数组切成按行的 vector<vector<float>>
	std::vector<std::vector<float>> to_rows(const float* data, int n, int cols)
	{
		std::vector<std::vector<float>> rows(n, std::vector<float>(cols));
		for (int i = 0; i < n; ++i)
		{
			std::memcpy(rows[i].data(), data + static_cast<size_t>(i) * cols, sizeof(float) * cols);
		}
		return rows;
	}
}

extern "C"
{
	// 固定随机数发生器，便于复现
	TA_API void ta_seed(unsigned int seed)
	{
		rand_generator().seed(seed);
	}

	TA_API const char* ta_last_error()
	{
		return g_error.c_str();
	}

	TA_API void* ta_create(int batch)
	{
		g_error.clear();
		if (batch < 1)
		{
			set_error("batch must be >= 1");
			return nullptr;
		}
		return new Handle(batch);
	}

	TA_API void ta_destroy(void* h)
	{
		delete static_cast<Handle*>(h);
	}

	TA_API int ta_add_layer(void* h, int num_inputs, int num_outputs, int activation)
	{
		auto* handle = static_cast<Handle*>(h);
		if (handle == nullptr)
		{
			set_error("null handle");
			return -1;
		}
		if (handle->loss_set)
		{
			set_error("add_layer must be called before set_loss");
			return -1;
		}
		const auto fn = activation == 0 ? FCLayer::ActivationFun::NONE : FCLayer::ActivationFun::RELU;
		try
		{
			handle->net.add_layer(num_inputs, num_outputs, fn);
		}
		catch (const std::exception& e)
		{
			set_error(e.what());
			return -1;
		}
		if (handle->in_dim == 0)
		{
			handle->in_dim = num_inputs;
		}
		handle->out_dim = num_outputs;
		return 0;
	}

	TA_API int ta_set_loss(void* h, int loss)
	{
		auto* handle = static_cast<Handle*>(h);
		if (handle == nullptr || handle->in_dim == 0)
		{
			set_error("build the network first");
			return -1;
		}
		handle->net.set_loss_function(loss == 0 ? Net::LossFun::MEAN_SQUARED : Net::LossFun::CROSS_ENTROPY);
		handle->loss_set = true;
		return 0;
	}

	TA_API void ta_init_params(void* h, float mean, float stddev)
	{
		static_cast<Handle*>(h)->net.init_parameters(mean, stddev);
	}

	TA_API int ta_set_backward(void* h, int mode)
	{
		static_cast<Handle*>(h)->net.set_backward_mode(mode == 0 ? Net::STACK_DFS : Net::MEMO_SEARCH);
		return 0;
	}

	TA_API int ta_in_dim(void* h)
	{
		return static_cast<Handle*>(h)->in_dim;
	}

	TA_API int ta_out_dim(void* h)
	{
		return static_cast<Handle*>(h)->out_dim;
	}

	TA_API float ta_train_batch(void* h, const float* x, const float* y, int n)
	{
		auto* handle = static_cast<Handle*>(h);
		if (n != handle->batch)
		{
			set_error("sample count must equal the batch size given to ta_create");
			return -1.0f;
		}
		const auto xs = to_rows(x, n, handle->in_dim);
		const auto ys = to_rows(y, n, handle->out_dim);
		return handle->net.train_batch(xs, ys);
	}

	TA_API void ta_sgd(void* h, float lr)
	{
		static_cast<Handle*>(h)->net.SGD(lr);
	}

	TA_API void ta_zero_grad(void* h)
	{
		static_cast<Handle*>(h)->net.zero_grad();
	}

	// 前向并把每行 softmax 概率写进 out（调用前需设置交叉熵损失）
	TA_API int ta_forward_probs(void* h, const float* x, int n, float* out)
	{
		auto* handle = static_cast<Handle*>(h);
		if (n > handle->batch)
		{
			set_error("n exceeds batch size");
			return -1;
		}
		std::vector<std::vector<float>> xs(n, std::vector<float>(handle->in_dim, 0.0f));
		for (int i = 0; i < n; ++i)
		{
			std::memcpy(xs[i].data(), x + static_cast<size_t>(i) * handle->in_dim, sizeof(float) * handle->in_dim);
		}
		// 补齐到 batch 行，避免图里的行数不匹配
		for (int i = n; i < handle->batch; ++i)
		{
			xs.push_back(std::vector<float>(handle->in_dim, 0.0f));
		}
		const auto ys = std::vector<std::vector<float>>(handle->batch, std::vector<float>(handle->out_dim, 0.0f));
		handle->net.forward_batch(xs, ys);
		for (int i = 0; i < n; ++i)
		{
			const std::vector<float>& p = handle->net.get_y_hat_softmax(i);
			std::memcpy(out + static_cast<size_t>(i) * handle->out_dim, p.data(), sizeof(float) * handle->out_dim);
		}
		return 0;
	}
}
