#include <cstdio>  //  用于print结果

__global__ void two_norm_kernel(const float *a,const float *b,float *c, int n, int m) {
    __shared__ float a_minus_b[64*64];  // 不支持可变数量的参数,矩阵维度最多为(4096,4096)

//     // 初始化a_minus_b
//     for(int row=threadIdx.x;row<n;row+=blockDim.x){
//         for(int col=0;col<m;col++){  // 循环展开
//             a_minus_b[row * 64 + col] = 0.0;
//         }
//     }

    // 异常、越界处理
    if(threadIdx.x==0)
        printf("after compute a-b:\n");
    for(int row=threadIdx.x;row<n;row+=blockDim.x){
        for(int col=0;col<m;col++){  // 循环展开
            float a_ij = a[row * m + col];
            float b_ij = b[row * m + col];
            a_minus_b[row * m + col] = (a_ij - b_ij) * (a_ij - b_ij);
            printf("tensor a-b,coord=(%d,%d), value=%f\n", row,col, a_minus_b[row * m + col]) ;
        }
    }

    __syncthreads();  // 等待整个a_minus_b被计算完毕

    // 将每行结果加到每行的第一个元素上
    if(threadIdx.x==0)
        printf("after add rows to first col\n");
    for(int row=threadIdx.x;row<n;row+=blockDim.x){
        for(int col=1;col<m;col++){  // 循环展开
            a_minus_b[row * m + 0] += a_minus_b[row * m + col];
        }
        printf("tensor a-b,row=%d, value=%f\n", row, a_minus_b[row * m + 0]);
    }

    __syncthreads();  // 等待结果都加到了第一行

    // 将所有结果加到第一行第一列元素上
	if(threadIdx.x == 0){
        for(int i=1;i<n;i++)
            a_minus_b[0] += a_minus_b[i * m + 0];
        c[0] = sqrtf(a_minus_b[0]);
    }
}

void launch_two_norm(const float *a,const float *b,float *c, int n, int m) {
    two_norm_kernel<<<1, 512>>>(a, b, c, n, m);
}