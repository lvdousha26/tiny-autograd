//

//

#ifndef _TENSOR_H_
#define _TENSOR_H_
#include <vector>
#include <algorithm>
class Tensor
{

 public:

	// 计算图的一条反向边：node 在第 slot 个输入位置使用了本张量
	struct Consumer
	{
		Tensor* node;
		int slot;
	};
	// 计算操作类型枚举
	enum CalculateFun {
		LEAF,               // 叶子节点（输入或参数）
		VECTOR_ADD_VECTOR,  // 向量加法
		VECTOR_SUB_VECTOR,  // 向量减法
		VECTOR_EXP,         // 指数运算
		VECTOR_LOG,         // 对数运算
		VECTOR_RELU,        // ReLU激活
		MATRIX_MUL_VECTOR,  // 矩阵乘向量
		VECTOR_ASSIGN,      // 向量赋值
		VECTOR_CROSSENTROPY,// 交叉熵损失
		VECTOR_SQUARED,     // 均方误差损失
		VECTOR_MUL_VECTOR,  // 逐元素相乘
		MATRIX_MUL_MATRIX,  // 矩阵乘矩阵
		TRANSPOSE,          // 转置
		SOFTMAX,             // 按行 softmax
		VECTOR_SIGMOID,     // sigmoid 激活
		VECTOR_TANH,        // tanh 激活
		VECTOR_LEAKY_RELU,  // leaky ReLU，负半轴斜率为 alpha
		VECTOR_ELU,         // ELU，负半轴斜率为 alpha
		VECTOR_GELU,        // GELU（精确 erf 形式）
		VECTOR_SILU,        // SiLU / Swish
		LAYER_NORM          // 逐行层归一化（不含仿射，gamma/beta 由 MUL/ADD 节点承担）
	};

	// 构造函数
	explicit Tensor(int row_dim, int col_dim, bool require_grad = false);

	Tensor(const Tensor&) = delete;
	Tensor& operator=(const Tensor&) = delete;

	// 反向传播：计算左子节点的梯度
	void backward_calculate_left_grad();

	// 反向传播：计算左子节点的梯度
	void backward_calculate_right_grad();

	// 前向计算：根据操作类型计算当前节点的值
	void forward_calculate_data();

	// 获取和设置数据、梯度
	[[nodiscard]] const std::vector<std::vector<float>>& get_grad() const;
	[[nodiscard]] std::vector<std::vector<float>>& get_data();
	[[nodiscard]] const std::vector<std::vector<float>>& get_data() const;
	[[nodiscard]] const std::vector<float>& get_softmax_data() const;
	[[nodiscard]] const std::vector<std::vector<float>>& get_softmax_batch() const;
	[[nodiscard]] const std::vector<Consumer>& get_consumers() const;
	void set_data(const std::vector<float>& value);

	// 批量写入：一次填满所有行（行数需与 row_dimension 一致）
	void set_batch_data(const std::vector<std::vector<float>>& value);

	// 清零梯度
	void zero_grad();

	// 梯度设为 1，根节点用
	void one_grad();

	[[nodiscard]] Tensor* get_left() const;
	[[nodiscard]] Tensor* get_right() const;
	void set_left(Tensor* l);
	void set_right(Tensor* r);
	void set_calculate_fn(CalculateFun fn);
	// leaky ReLU / ELU 的负半轴斜率
	void set_alpha(float value);
	[[nodiscard]] float get_alpha() const;
	[[nodiscard]] int get_row_dimension() const;
	[[nodiscard]] int get_col_dimension() const;
	[[nodiscard]] bool is_require_grad() const;
	[[nodiscard]] CalculateFun get_calculate_fn() const;
 private:
	std::vector<std::vector<float>> data; 	// 数据存储
	std::vector<std::vector<float>> soft_max_data; 		// Softmax结果，用交叉熵做损失函数，反向求导时用到，其他不用
	std::vector<std::vector<float>> grad;	// 梯度存储
	std::vector<Consumer> consumers;		// 反向边：谁用了我（记忆化搜索使用）
	int row_dimension;						// 维度信息
	int col_dimension;						// 维度信息
	Tensor* left = nullptr;					// 左子节点
	Tensor* right = nullptr;				// 右子节点
	bool require_grad = false;
	float alpha = 0.0f;
	CalculateFun calculate_fn = LEAF;		// 是否需要计算梯度
};

inline const std::vector<std::vector<float>>& Tensor::get_grad() const
{
	return grad;
}

inline std::vector<std::vector<float>>& Tensor::get_data()
{
	return data;
}

inline const std::vector<std::vector<float>>& Tensor::get_data() const
{
	return data;
}

inline const std::vector<float>& Tensor::get_softmax_data() const
{
	return soft_max_data[0];
}

inline void Tensor::set_data(const std::vector<float>& value)
{
	std::copy(value.begin(), value.end(), data[0].begin());
}

inline const std::vector<std::vector<float>>& Tensor::get_softmax_batch() const
{
	return soft_max_data;
}

inline void Tensor::set_batch_data(const std::vector<std::vector<float>>& value)
{
	for (int r = 0; r < row_dimension; ++r)
	{
		std::copy(value[r].begin(), value[r].end(), data[r].begin());
	}
}

inline void Tensor::zero_grad()
{
	for(auto& it:grad )
	{
		std::fill(it.begin(), it.end(), 0.0f);
	}
}

inline void Tensor::one_grad()
{
	for(auto& it:grad )
	{
		std::fill(it.begin(), it.end(), 1.0f);
	}
}

inline Tensor* Tensor::get_left() const
{
	return left;
}

inline Tensor* Tensor::get_right() const
{
	return right;
}

inline void Tensor::set_left(Tensor* l)
{
	left = l;
	if (l != nullptr)
	{
		l->consumers.push_back({this, 0});
	}
}
inline void Tensor::set_right(Tensor* r)
{
	right = r;
	if (r != nullptr)
	{
		r->consumers.push_back({this, 1});
	}
}
inline void Tensor::set_calculate_fn(CalculateFun fn)
{
	calculate_fn = fn;
}

inline void Tensor::set_alpha(float value)
{
	alpha = value;
}

inline float Tensor::get_alpha() const
{
	return alpha;
}
inline const std::vector<Tensor::Consumer>& Tensor::get_consumers() const
{
	return consumers;
}

inline int Tensor::get_row_dimension() const
{
	return row_dimension;
}
inline int Tensor::get_col_dimension() const
{
	return col_dimension;
}

inline bool Tensor::is_require_grad() const
{
	return require_grad;
}

inline Tensor::CalculateFun Tensor::get_calculate_fn() const
{
	return calculate_fn;
}
#endif //_TENSOR_H_
