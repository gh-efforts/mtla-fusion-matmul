from torch.utils.cpp_extension import load
import os
import torch


if __name__ == '__main__':
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

    mtla = load(
        name="mtla",
        sources=["mtla.cu"],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        verbose=True
    )

    window = 6
    row = 24
    col = 4

    matQ = torch.ones(2, row, col, device="cuda", dtype=torch.bfloat16)
    matK = torch.ones(2, row, col, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(2, row, row, device="cuda", dtype=torch.bfloat16)

    mtla.mtla_matmul(matQ.data_ptr(), matK.data_ptr(), out.data_ptr(), col, row, 2, window, torch.cuda.current_stream(device=None).cuda_stream)

    out_fp32 = out.to(torch.float32)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=1000000,
                           threshold=10 ** 9, edgeitems=0)

    print(out_fp32)
