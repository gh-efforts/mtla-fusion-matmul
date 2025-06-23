#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

struct Point {
    size_t x;
    size_t y;
    uint8_t mask;
};

__global__ void mtla_matmul_kernel(
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    __nv_bfloat16 *out,
    size_t col_num,
    size_t row_num,
    const Point *points,
    size_t points_len
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= points_len) {
        return;
    }

    size_t point_idx = tid % points_len;
    size_t batch_idx = tid / points_len;

    const __nv_bfloat16 *a_offset = a + batch_idx * row_num * col_num;
    const __nv_bfloat16 *b_offset = b + batch_idx * row_num * col_num;
    __nv_bfloat16 *out_offset = out + batch_idx * row_num * col_num;

    const Point &point = points[point_idx];
    size_t x = point.x;
    size_t y = point.y;
    uint8_t mask = point.mask;

    __nv_bfloat16 *out_point = out_offset + y * row_num + x;
    const __nv_bfloat16 *row = a_offset + y * col_num;

    __nv_bfloat16 out_point_tmp = __float2bfloat16(0.0f);
    for (size_t i = 0; i < col_num; i++) {
        __nv_bfloat16 a_v = *(row + i);

        if (mask == 1) {
            __nv_bfloat16 b_v = b_offset[x * col_num + i];
            out_point_tmp += a_v * b_v;
        } else if (mask == 2) {
            __nv_bfloat16 b_v = b_offset[(x - 1) * col_num + i] +
                                b_offset[x * col_num + i];
            out_point_tmp += a_v * b_v;
        } else if (mask == 4) {
            __nv_bfloat16 b_v = b_offset[(x - 3) * col_num + i] +
                                b_offset[(x - 2) * col_num + i] +
                                b_offset[(x - 1) * col_num + i] +
                                b_offset[x * col_num + i];
            out_point_tmp += a_v * b_v;
        }
    }
    *out_point = out_point_tmp;
}

uint8_t find_mask(size_t x, size_t y, size_t window) {
    if (x > y) {
        return 0;
    }

    size_t x_num = x + 1;
    size_t y_num = y + 1;
    size_t new_window = window;

    if (y_num % 2 != 0) {
        new_window += 1;
    }

    if (y_num <= new_window) {
        return 1;
    }

    size_t not_in_window = y_num - new_window;

    if (not_in_window < x_num) {
        return 1;
    }

    if (x_num % 2 != 0) {
        return 0;
    }

    size_t not_in_vwindow = not_in_window - std::min(window, not_in_window);

    if (not_in_vwindow < x_num) {
        return 2;
    }

    if (not_in_vwindow < 4) {
        return 2;
    }

    if (x_num % 4 == 0) {
        return 4;
    } else {
        if (x_num == not_in_vwindow) {
            return 2;
        }
        return 0;
    }
}

std::vector<Point> gen_point_list(size_t mat_rows, size_t window) {
    std::vector<Point> threads;

    for (size_t y = 0; y < mat_rows; ++y) {
        for (size_t x = 0; x < mat_rows; ++x) {
            if (x > y) {
                break;
            }

            uint8_t mask = find_mask(x, y, window);

            if (mask != 0) {
                threads.push_back(Point{x, y, mask});
            }
        }
    }

    return threads;
}

void mtla_matmul(
    size_t a,
    size_t b,
    size_t out,
    size_t col_num,
    size_t row_num,
    size_t batch_size,
    size_t window,
    uint64_t stream_int
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);

    std::vector<Point> points = gen_point_list(row_num, window);
    size_t points_len = points.size() * batch_size;

    size_t thread = 256;
    size_t block = points_len / thread;
    if (points_len % thread != 0) {
        block += 1;
    }

    Point* d_points = nullptr;
    cudaMallocAsync(&d_points, points.size() * sizeof(Point), stream);
    cudaMemcpyAsync(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice, stream);
    mtla_matmul_kernel<<<block, thread, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out),
        col_num,
        row_num,
        d_points,
        points.size()
    );
    cudaFreeAsync(d_points, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mtla_matmul", &mtla_matmul, "mtla_matmul");
}