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
		bool dynamic = true;   // 行数变化时自动重建图（保留参数）
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

	// 图是固定 batch 的静态图：不足 batch 行时用 0 补齐再前向
	std::vector<std::vector<float>> padded_input(const float* data, int n, int cols, int batch)
	{
		std::vector<std::vector<float>> rows(batch, std::vector<float>(cols, 0.0f));
		for (int i = 0; i < n; ++i)
		{
			std::memcpy(rows[i].data(), data + static_cast<size_t>(i) * cols, sizeof(float) * cols);
		}
		return rows;
	}

	// 让图适配 n 行：dynamic 模式下按 n 重建图（参数不丢），否则要求行数一致
	bool adapt(Handle* h, int n)
	{
		const int rows = h->net.get_batch_rows();
		if (n == rows)
		{
			return true;
		}
		if (!h->dynamic || n < 1)
		{
			set_error("sample count differs from the graph batch (static mode)");
			return false;
		}
		h->net.resize_batch(n);
		return true;
	}

	Tensor* layer_tensor(Handle* h, int idx, int which, const char** err)
	{
		if (idx < 0 || idx >= h->net.layer_count())
		{
			*err = "layer index out of range";
			return nullptr;
		}
		switch (which)
		{
		case 0: return h->net.layer_outputs(idx);
		case 1: return h->net.layer_weights(idx);
		case 2: return h->net.layer_bias(idx);
		default: return h->net.layer_inputs(idx);
		}
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
		static const FCLayer::ActivationFun kActs[] = {
			FCLayer::NONE, FCLayer::RELU, FCLayer::SIGMOID, FCLayer::TANH,
			FCLayer::LEAKY_RELU, FCLayer::ELU, FCLayer::GELU, FCLayer::SILU
		};
		if (activation < 0 || activation > 7)
		{
			set_error("unknown activation code");
			return -1;
		}
		const auto fn = kActs[activation];
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
		if (!adapt(handle, n))
		{
			return -1.0f;
		}
		const auto xs = to_rows(x, n, handle->in_dim);
		const auto ys = to_rows(y, n, handle->out_dim);
		return handle->net.train_batch(xs, ys);
	}

	TA_API int ta_set_dynamic(void* h, int on)
	{
		static_cast<Handle*>(h)->dynamic = (on != 0);
		return 0;
	}

	TA_API int ta_batch(void* h)
	{
		return static_cast<Handle*>(h)->net.get_batch_rows();
	}

	// 显式按 n 行重建图（参数保留），不跑前向
	TA_API int ta_resize_batch(void* h, int n)
	{
		auto* handle = static_cast<Handle*>(h);
		if (n < 1)
		{
			set_error("n must be >= 1");
			return -1;
		}
		handle->net.resize_batch(n);
		return 0;
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
		if (n < 1)
		{
			set_error("n must be >= 1");
			return -1;
		}
		if (n > handle->net.get_batch_rows() && !adapt(handle, n))
		{
			return -1;
		}
		const int rows = handle->net.get_batch_rows();
		const auto xs = padded_input(x, n, handle->in_dim, rows);
		const auto ys = std::vector<std::vector<float>>(
			rows, std::vector<float>(handle->out_dim, 0.0f));
		handle->net.forward_batch(xs, ys);
		for (int i = 0; i < n; ++i)
		{
			const std::vector<float>& p = handle->net.get_y_hat_softmax(i);
			std::memcpy(out + static_cast<size_t>(i) * handle->out_dim, p.data(), sizeof(float) * handle->out_dim);
		}
		return 0;
	}

	// ---- 稀疏自编码器需要的额外接口 -------------------------------------

	// 添加一层 TopK 激活：每行只保留前 k 个正值
	TA_API int ta_add_topk_layer(void* h, int num_inputs, int num_outputs, int k)
	{
		auto* handle = static_cast<Handle*>(h);
		if (handle == nullptr || handle->loss_set)
		{
			set_error("add_layer must be called before set_loss");
			return -1;
		}
		if (k < 1 || k > num_outputs)
		{
			set_error("top_k must be in [1, num_outputs]");
			return -1;
		}
		handle->net.add_layer(num_inputs, num_outputs, FCLayer::TOPK, k);
		if (handle->in_dim == 0)
		{
			handle->in_dim = num_inputs;
		}
		handle->out_dim = num_outputs;
		return 0;
	}

	// 给第 layer 层的输出加 L1 稀疏惩罚（layer=0 通常是编码器隐层）
	TA_API int ta_set_sparse_penalty(void* h, float coeff, int layer)
	{
		auto* handle = static_cast<Handle*>(h);
		if (handle == nullptr || layer < 0 || layer >= handle->net.layer_count())
		{
			set_error("layer index out of range");
			return -1;
		}
		handle->net.set_sparse_penalty(coeff, layer);
		return 0;
	}

	TA_API int ta_layer_count(void* h)
	{
		return static_cast<Handle*>(h)->net.layer_count();
	}

	// which: 0=outputs 1=weights 2=bias 3=inputs；写回 (rows, cols)
	TA_API int ta_layer_dims(void* h, int layer, int which, int* rows, int* cols)
	{
		const char* err = nullptr;
		Tensor* t = layer_tensor(static_cast<Handle*>(h), layer, which, &err);
		if (t == nullptr)
		{
			set_error(err);
			return -1;
		}
		*rows = t->get_row_dimension();
		*cols = t->get_col_dimension();
		return 0;
	}

	// 前向并返回指定层的输出（原始值，不做 softmax），n 可以小于 batch
	TA_API int ta_forward_layer(void* h, const float* x, int n, int layer, float* out)
	{
		auto* handle = static_cast<Handle*>(h);
		if (n < 1)
		{
			set_error("n must be >= 1");
			return -1;
		}
		if (!handle->loss_set)
		{
			set_error("call set_loss before forward_layer");
			return -1;
		}
		if (n > handle->net.get_batch_rows() && !adapt(handle, n))
		{
			return -1;
		}
		const int rows = handle->net.get_batch_rows();
		const char* err = nullptr;
		Tensor* t = layer_tensor(handle, layer, 0, &err);
		if (t == nullptr)
		{
			set_error(err);
			return -1;
		}
		const auto xs = padded_input(x, n, handle->in_dim, rows);
		const auto ys = std::vector<std::vector<float>>(
			rows, std::vector<float>(handle->out_dim, 0.0f));
		handle->net.forward_batch(xs, ys);
		const int cols = t->get_col_dimension();
		for (int i = 0; i < n; ++i)
		{
			std::memcpy(out + static_cast<size_t>(i) * cols,
				t->get_data()[i].data(), sizeof(float) * cols);
		}
		return 0;
	}

	// 读/写某一层的权重或偏置（按行主序平铺）
	TA_API int ta_get_layer_array(void* h, int layer, int which, float* out)
	{
		const char* err = nullptr;
		Tensor* t = layer_tensor(static_cast<Handle*>(h), layer, which, &err);
		if (t == nullptr)
		{
			set_error(err);
			return -1;
		}
		const int cols = t->get_col_dimension();
		for (int i = 0; i < t->get_row_dimension(); ++i)
		{
			std::memcpy(out + static_cast<size_t>(i) * cols,
				t->get_data()[i].data(), sizeof(float) * cols);
		}
		return 0;
	}

	TA_API int ta_set_layer_array(void* h, int layer, int which, const float* data)
	{
		const char* err = nullptr;
		Tensor* t = layer_tensor(static_cast<Handle*>(h), layer, which, &err);
		if (t == nullptr)
		{
			set_error(err);
			return -1;
		}
		const int cols = t->get_col_dimension();
		for (int i = 0; i < t->get_row_dimension(); ++i)
		{
			std::memcpy(t->get_data()[i].data(),
				data + static_cast<size_t>(i) * cols, sizeof(float) * cols);
		}
		return 0;
	}

	// 读某一层权重的梯度（用于和 NumPy 参考实现做交叉校验）
	TA_API int ta_get_weight_grad(void* h, int layer, float* out)
	{
		const char* err = nullptr;
		Tensor* t = layer_tensor(static_cast<Handle*>(h), layer, 1, &err);
		if (t == nullptr)
		{
			set_error(err);
			return -1;
		}
		const auto& g = t->get_grad();
		const int cols = t->get_col_dimension();
		for (int i = 0; i < t->get_row_dimension(); ++i)
		{
			std::memcpy(out + static_cast<size_t>(i) * cols,
				g[i].data(), sizeof(float) * cols);
		}
		return 0;
	}
}
