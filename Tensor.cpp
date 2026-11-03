//

//
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Tensor.h"
Tensor::Tensor(int row_dim, int col_dim, bool require_grad)
	: row_dimension(row_dim), col_dimension(col_dim), require_grad(require_grad)
{
	data.resize(row_dimension);
	for (auto& it : data)
	{
		it.resize(col_dimension);
	}

	soft_max_data.assign(row_dimension, std::vector<float>(col_dimension, 0.0f));

	grad.resize(row_dimension);
	for (auto& it : grad)
	{
		it.resize(col_dimension);
	}
}

namespace
{
	// 广播规则：行数为 1 的张量按行复用（偏置、单行输入等）
	inline int row_of(int rows, int r)
	{
		return rows == 1 ? 0 : r;
	}
}

void Tensor::backward_calculate_left_grad()
{
	switch (this->calculate_fn)
	{
	case CalculateFun::VECTOR_ADD_VECTOR:
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i];
			}
		}
		break;
	case CalculateFun::VECTOR_EXP:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& ld = left->data[r];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i] * std::exp(ld[i]);
			}
		}
		break;
	case CalculateFun::VECTOR_LOG:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& ld = left->data[r];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i] / ld[i];
			}
		}
		break;
	case CalculateFun::VECTOR_RELU:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& ld = left->data[r];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i] * (ld[i] < 0.0f ? 0.0f : 1.0f);
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_VECTOR:
		// 左操作数是权重 (out x in)：dW[i][j] += sum_r dY[r][i] * X[r][j]
		{
			const int in = left->col_dimension;
#pragma omp parallel for schedule(static) if (row_dimension >= 4)
			for (int i = 0; i < col_dimension; ++i)
			{
				float* gw = left->grad[i].data();
				for (int r = 0; r < row_dimension; ++r)
				{
					const float gv = this->grad[r][i];
					const float* xr = right->data[row_of(right->row_dimension, r)].data();
					for (int j = 0; j < in; ++j)
					{
						gw[j] += gv * xr[j];
					}
				}
			}
		}
		break;
	case CalculateFun::VECTOR_ASSIGN:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i];
			}
		}
		break;
	case CalculateFun::VECTOR_MUL_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			for (int i = 0; i < col_dimension; ++i)
			{
				gl[i] += g[i] * y[i];
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_MATRIX:
		// dL[i][p] += sum_j dC[i][j] * R[p][j]
		for (int i = 0; i < row_dimension; ++i)
		{
			float* gl = left->grad[i].data();
			const float* grow = this->grad[i].data();
			for (int p = 0; p < left->col_dimension; ++p)
			{
				const float* rrow = right->data[p].data();
				float acc = 0.0f;
				for (int j = 0; j < col_dimension; ++j)
				{
					acc += grow[j] * rrow[j];
				}
				gl[p] += acc;
			}
		}
		break;
	case CalculateFun::TRANSPOSE:
		for (int i = 0; i < left->row_dimension; ++i)
		{
			for (int j = 0; j < left->col_dimension; ++j)
			{
				left->grad[i][j] += this->grad[j][i];
			}
		}
		break;
	case CalculateFun::SOFTMAX:
		// dy = y * (g - sum_k g[k] * y[k])
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gl = left->grad[row_of(left->row_dimension, r)];
			const std::vector<float>& y = data[r];
			const std::vector<float>& gr = this->grad[r];
			float dot = 0.0f;
			for (int j = 0; j < col_dimension; ++j)
			{
				dot += gr[j] * y[j];
			}
			for (int j = 0; j < col_dimension; ++j)
			{
				gl[j] += y[j] * (gr[j] - dot);
			}
		}
		break;
	case CalculateFun::VECTOR_CROSSENTROPY:
	{
		const int rows = left->row_dimension;
		const float scale = this->grad[0][0] / static_cast<float>(rows);
		for (int r = 0; r < rows; ++r)
		{
			std::vector<float>& gl = left->grad[r];
			const std::vector<float>& sm = left->soft_max_data[r];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			for (int j = 0; j < left->col_dimension; ++j)
			{
				gl[j] += scale * (sm[j] - y[j]);
			}
		}
		break;
	}
	case CalculateFun::VECTOR_SQUARED:
	{
		const int rows = left->row_dimension;
		const float scale = this->grad[0][0] / static_cast<float>(rows);
		for (int r = 0; r < rows; ++r)
		{
			std::vector<float>& gl = left->grad[r];
			const std::vector<float>& ld = left->data[r];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			for (int j = 0; j < left->col_dimension; ++j)
			{
				gl[j] += scale * (ld[j] - y[j]);
			}
		}
		break;
	}
	case LEAF:
		break;
	}
}

void Tensor::backward_calculate_right_grad()
{
	switch (this->calculate_fn)
	{
	case CalculateFun::VECTOR_ADD_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gr = right->grad[row_of(right->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gr[i] += g[i];
			}
		}
		break;
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gr = right->grad[row_of(right->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				gr[i] -= g[i];
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_VECTOR:
		// 右操作数是输入 (B x in)：dX[r][j] += sum_i dY[r][i] * W[i][j]
		// 按 i 外层累加到 dX 行上：权重顺序读，dX 常驻缓存
		{
			const int in = right->col_dimension;
#pragma omp parallel for schedule(static) if (row_dimension >= 4)
			for (int r = 0; r < row_dimension; ++r)
			{
				float* gx = right->grad[row_of(right->row_dimension, r)].data();
				const float* g = this->grad[r].data();
				for (int i = 0; i < col_dimension; ++i)
				{
					const float gv = g[i];
					const float* w = left->data[i].data();
					for (int j = 0; j < in; ++j)
					{
						gx[j] += gv * w[j];
					}
				}
			}
		}
		break;
	case CalculateFun::VECTOR_EXP:
	case CalculateFun::VECTOR_LOG:
	case CalculateFun::VECTOR_RELU:
	case CalculateFun::VECTOR_ASSIGN:
	case CalculateFun::VECTOR_MUL_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			std::vector<float>& gr = right->grad[row_of(right->row_dimension, r)];
			const std::vector<float>& g = this->grad[r];
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			for (int i = 0; i < col_dimension; ++i)
			{
				gr[i] += g[i] * x[i];
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_MATRIX:
		// dR[p][j] += sum_i L[i][p] * dC[i][j]
		for (int p = 0; p < right->row_dimension; ++p)
		{
			float* gr = right->grad[p].data();
			for (int i = 0; i < row_dimension; ++i)
			{
				const float lv = left->data[i][p];
				const float* grow = this->grad[i].data();
				for (int j = 0; j < col_dimension; ++j)
				{
					gr[j] += lv * grow[j];
				}
			}
		}
		break;
	case CalculateFun::TRANSPOSE:
	case CalculateFun::SOFTMAX:
		break;
	case CalculateFun::VECTOR_CROSSENTROPY:
	case CalculateFun::VECTOR_SQUARED:
	case LEAF:
		break;
	}
}

void Tensor::forward_calculate_data()
{
	switch (calculate_fn)
	{
	case CalculateFun::LEAF:
		break;
	case CalculateFun::VECTOR_ADD_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = x[i] + y[i];
			}
		}
		break;
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = x[i] - y[i];
			}
		}
		break;
	case CalculateFun::VECTOR_EXP:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = std::exp(x[i]);
			}
		}
		break;
	case CalculateFun::VECTOR_LOG:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = std::log(x[i]);
			}
		}
		break;
	case CalculateFun::VECTOR_RELU:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = std::max(0.0f, x[i]);
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_VECTOR:
		// 权重 (out x in) 乘以输入 (B x in)，输出 (B x out)
		// i 外层（并行）、r 内层：一行权重只读一次就服务整个 batch
		{
			const int in = left->col_dimension;
			const int rows = row_dimension;
#pragma omp parallel for schedule(static) if (rows >= 4)
			for (int i = 0; i < col_dimension; ++i)
			{
				const float* w = left->data[i].data();
				for (int r = 0; r < rows; ++r)
				{
					const float* x = right->data[row_of(right->row_dimension, r)].data();
					float s0 = 0.0f;
					float s1 = 0.0f;
					float s2 = 0.0f;
					float s3 = 0.0f;
					int j = 0;
					for (; j + 4 <= in; j += 4)
					{
						s0 += w[j] * x[j];
						s1 += w[j + 1] * x[j + 1];
						s2 += w[j + 2] * x[j + 2];
						s3 += w[j + 3] * x[j + 3];
					}
					for (; j < in; ++j)
					{
						s0 += w[j] * x[j];
					}
					data[r][i] = (s0 + s1) + (s2 + s3);
				}
			}
		}
		break;	case CalculateFun::VECTOR_MUL_VECTOR:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = x[i] * y[i];
			}
		}
		break;
	case CalculateFun::MATRIX_MUL_MATRIX:
		// C = L(m x k) * R(k x n)；按 p 外层复用 L 的一个元素，写连续
		{
			const int m = row_dimension;
			const int n = col_dimension;
			const int k = left->col_dimension;
			for (int i = 0; i < m; ++i)
			{
				float* d = data[i].data();
				const float* lrow = left->data[i].data();
				for (int j = 0; j < n; ++j)
				{
					d[j] = 0.0f;
				}
				for (int p = 0; p < k; ++p)
				{
					const float lv = lrow[p];
					const float* rrow = right->data[p].data();
					for (int j = 0; j < n; ++j)
					{
						d[j] += lv * rrow[j];
					}
				}
			}
		}
		break;
	case CalculateFun::TRANSPOSE:
		for (int i = 0; i < left->row_dimension; ++i)
		{
			for (int j = 0; j < left->col_dimension; ++j)
			{
				data[j][i] = left->data[i][j];
			}
		}
		break;
	case CalculateFun::SOFTMAX:
		// 按行 softmax，先减行最大值保证数值稳定
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			std::vector<float>& d = data[r];
			float mx = x[0];
			for (int j = 1; j < col_dimension; ++j)
			{
				mx = std::max(mx, x[j]);
			}
			float sum = 0.0f;
			for (int j = 0; j < col_dimension; ++j)
			{
				d[j] = std::exp(x[j] - mx);
				sum += d[j];
			}
			for (int j = 0; j < col_dimension; ++j)
			{
				d[j] /= sum;
			}
		}
		break;
	case CalculateFun::VECTOR_CROSSENTROPY:
	{
		const int rows = left->row_dimension;
		const int classes = left->col_dimension;
		float total = 0.0f;
		for (int r = 0; r < rows; ++r)
		{
			std::vector<float>& sm = left->soft_max_data[r];
			const std::vector<float>& ld = left->data[r];
			float sum = 0.0f;
			for (int j = 0; j < classes; ++j)
			{
				sm[j] = std::exp(ld[j]);
				sum += sm[j];
			}
			for (int j = 0; j < classes; ++j)
			{
				sm[j] /= sum;
			}
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			for (int j = 0; j < classes; ++j)
			{
				if (y[j] == 1.0f)
				{
					total += -std::log(sm[j]);
					break;
				}
			}
		}
		data[0][0] = total / static_cast<float>(rows);
		break;
	}
	case CalculateFun::VECTOR_SQUARED:
	{
		const int rows = left->row_dimension;
		const int cols = left->col_dimension;
		float total = 0.0f;
		for (int r = 0; r < rows; ++r)
		{
			const std::vector<float>& x = left->data[r];
			const std::vector<float>& y = right->data[row_of(right->row_dimension, r)];
			for (int j = 0; j < cols; ++j)
			{
				total += (x[j] - y[j]) * (x[j] - y[j]);
			}
		}
		data[0][0] = total / 2.0f / static_cast<float>(rows);
		break;
	}
	case CalculateFun::VECTOR_ASSIGN:
		for (int r = 0; r < row_dimension; ++r)
		{
			const std::vector<float>& x = left->data[row_of(left->row_dimension, r)];
			std::vector<float>& d = data[r];
			for (int i = 0; i < col_dimension; ++i)
			{
				d[i] = x[i];
			}
		}
		break;
	}
}
