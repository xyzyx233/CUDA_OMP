
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
#include <memory.h>
#include "matrixop.cuh"


#define RR 190
#define M 256
#define N 256
#define BLOCK_SIZE 16
#define THREAD_NUM 256
#define imin(a, b)    (a<b ? a : b)
const int threadsPerBlock = 16;
const int blocksPerGrid = imin(32, (RR + threadsPerBlock - 1) / threadsPerBlock);

//CPU函数
//double** matrixturn(double** matrix, int height, int width) {
//	int i = 0, j;
//	double** turn = (double**)malloc(width * sizeof(double*));
//	if (!turn)
//		return NULL;
//	for (i = 0; i < width; i++)
//		turn[i] = (double*)malloc(height * sizeof(double));
//	for (i = 0; i < width; i++)
//	for (j = 0; j < height; j++)
//		turn[i][j] = matrix[j][i];
//	return turn;
//}
//double** matrixmultiplic(double** matrix1, int width, double** matrix2,
//	int height, int wid2) {
//	int i, j, k;
//	double** ans = NULL;
//	//double** ans = creatmatrix(height,width); 这种写法会报错 conflicting types for 在声明前使用
//	ans = (double**)malloc(height * sizeof(double*));
//	if (ans == NULL)
//		return NULL;
//	for (i = 0; i < height; i++) {
//		ans[i] = (double*)malloc(width * sizeof(double));
//		if (ans[i] == NULL)
//			return NULL;
//	}
//	//	矩阵乘法
//	for (i = 0; i < height; i++) {
//		for (j = 0; j < width; j++) {
//			ans[i][j] = 0;
//			for (k = 0; k < wid2; k++) {
//				ans[i][j] += matrix2[i][k] * matrix1[k][j];
//			}
//		}
//	}
//	return ans;
//}
//double** creatmatrix(int height, int width) {
//	double **pixel = (double**)malloc(height * sizeof(double*));
//	if (pixel == NULL) {
//		printf("It is out of memory1!\n");
//		return NULL;
//	}
//	int i, j;
//	for (i = 0; i < height; i++) {
//		pixel[i] = (double*)malloc(width * sizeof(double));
//		if (pixel[i] == NULL) {
//			printf("It is out of memory2! %d\n", i);
//			return NULL;
//		}
//	}
//	for (i = 0; i < height; i++) {
//		for (j = 0; j < width; j++) {
//			pixel[i][j] = 0;
//		}
//	}
//	return pixel;
//}

//GPU核函数
/*
赋值
*/
__global__ void result(float *result, float *aug_y, int *posarray, int i)
{
	int j;
	for (j = 0; j < N; j++)
	{
		result[j] = 0;
	}
	for (j = 0; j < i; j++)
	{
		result[posarray[j]] = aug_y[j];
	}
}
/*
数值相乘
*/
__global__ void valueMul(float *c, float *a, int i)
{
	(*c) = a[i] * a[i];
}
/*
开根号
*/
__global__ void valueSqrt(float *a, float *b, float *c)
{
	(*c) = (*b) / sqrt((*a));
}
/*
找最大值下标
*/
__global__ void d_find_max(int *idx, float *P, int n){
	float output[N];
	for (int i = 0;i < N;i++)
		output[i] = P[i];
	for (int i = 1; i<n; i++){
		if (P[i]>P[0]){
			P[0] = P[i];
			(*idx) = i;
		}
	}
	return;
}
/*
向量相减
*/
__global__ void vector_sub_vector(float *C, float *A, float *B, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		C[tid] = A[tid] - B[tid];
	}
}
/*
矩阵相乘
A：M*P，B：P*N
*/
__global__ static void matMult_gpu(float *A, float *B, float *C, int m, int p, int n)
{
	extern __shared__ float data[];
	int tid = threadIdx.x;
	int row = blockIdx.x;   //一个Row只能由同一个block的threads来进行计算
	int i, j;
	for (i = tid; i<p; i += blockDim.x){
		data[i] = A[row*p + i];
	}
	__syncthreads();

	for (j = tid; j<n; j += blockDim.x){
		float t = 0;
		float y = 0;
		for (i = 0; i<p; i++){
			float r;
			y -= data[i] * B[i*n + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		C[row*n + j] = t;
	}
}
/*
矩阵转置
*/
__global__ static void matrix_transpose(float *A_T, float *A, int hA, int wA)
{
	__shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

	unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if ((xIndex < wA) && (yIndex < hA))
	{
		unsigned int aIndex = yIndex * wA + xIndex;
		temp[threadIdx.y][threadIdx.x] = A[aIndex];
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if ((xIndex < hA) && (yIndex < wA))
	{
		unsigned int a_tIndex = yIndex * hA + xIndex;
		A_T[a_tIndex] = temp[threadIdx.x][threadIdx.y];
	}
}
/*
矩阵与向量相乘
A(aH, aW); B(aW, 1); C(aH, 1)
*/
__global__ static void matrix_x_vector(float *C, float *A, float *B, int wA)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = bid*blockDim.x + tid;
	float temp = 0.0;
	__syncthreads();

	if (offset<wA)
	{
		for (int i = 0; i < wA; i++)
		{
			temp += A[offset*wA + i] * B[i];
		}
		__syncthreads();
		C[offset] = temp;
	}
}

/*
向量点积
先求出两两相乘的乘积，再归约求所有乘积的和
*/
__global__ void vector_dot_product(float *C, float *A, float *B, int n)
{
	__shared__ float temp[BLOCK_SIZE];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	double result = 0.0;
	while (tid < n)
	{
		result += A[tid] * B[tid];
		tid += blockDim.x * gridDim.x;
	}
	temp[tempIndex] = result;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		C[blockIdx.x] = temp[0];
	}
}
//__global__ void vector_dot_sum(float *C)
//{
//	extern __shared__ float temp[];
//	temp[threadIdx.x] = C[threadIdx.x];
//	for (int s = 1; s < blockDim.x; s *= 2)
//	{
//		if (threadIdx.x % (2 * s) == 0)
//		{
//			temp[threadIdx.x] += temp[threadIdx.x + s];
//		}
//		__syncthreads();
//	}
//	if (threadIdx.x == 0)
//	{
//		C[0] = temp[0];
//	}
//}
__global__ void vector_dot_sum(float *C)
{
	float temp=0;
	for (int s = 0; s < RR; s ++)
	{
		temp += C[s];
	}
		C[0] = temp;
}
/********************************************************************
*  矩阵求逆
*********************************************************************/
/*cholesky分解 A = LDL^T
最后a中存放的是分解后的单位下三角矩阵L，
d中存放的是对角均为正数的对角矩阵
d_inversion中存放的是对角均为正数的对角矩阵的逆矩阵
*/
__global__ void cholesky(float *a, float *d, float *d_inversion, int n)
{
	extern __shared__ float result_sum[];
	int tid = threadIdx.x;
	float sum = 0.0;
	float sum_t;
	for (int i = 0; i<n; i++)
	{
		for (int j = 0; j<n; j++)
		{
			sum = a[j*n + i];  //第一项
			for (int k = tid; k<i; k += blockDim.x){
				result_sum[k] = a[i*n + k] * a[j*n + k] * d[k*n + k];
			}
			__syncthreads();
			sum_t = 0;
			for (int k = 0; k<i; k++){
				sum_t += result_sum[k];
			}
			sum -= sum_t;

			if (i == j)
			{
				d[i*n + i] = sum;
				d_inversion[i*n + i] = 1 / sum;
				a[i*n + j] = 1;
			}
			else if (j<i)
			{
				a[j*n + i] = 0;

			}
			else
			{
				a[j*n + i] = sum / d[i*n + i];
			}
		}
	}
}
/*单位下三角矩阵求逆
最后E中是A的逆矩阵
*/
__global__ void matrix_inversion(float *a, float *E, int n)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = tid + blockDim.x*bid;
	for (int i = 0; i<n; i++)
	{
		for (int k = i + 1; k<n; k++)
		{
			if (offset<n){
				E[k*n + offset] = E[k*n + offset] - E[i*n + offset] * a[k*n + i];
			}
		}
	}
}

/*矩阵转置
最后dev_a_transpose中是dev_E矩阵的转置矩阵
*/
__global__ static void matrix_trans(float *dev_E, float *dev_a_transpose, int n)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = tid + bid*blockDim.x;
	int row = offset / n;
	int column = offset % n;
	if (row<n && column<n){
		dev_a_transpose[row*n + column] = dev_E[column*n + row];
	}
}
/*矩阵求积
最后A_t中是A矩阵和B矩阵相乘的结果
*/
__global__ static void  matrix_multiply(float *A, float *B, float *A_t, int n){
	extern __shared__ float data[];
	int tid = threadIdx.x;
	int row = blockIdx.x;   //一个Row只能由同一个block的threads来进行计算
	int i, j;
	for (i = tid; i<n; i += blockDim.x){
		data[i] = A[row*n + i];
	}
	__syncthreads();
	for (j = tid; j<n; j += blockDim.x){
		float t = 0;
		float y = 0;
		for (i = 0; i<n; i++){
			float r;
			y -= data[i] * B[i*n + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		A_t[row*n + j] = t;
	}
}

/*定义WORD为两个字节的类型*/
typedef unsigned short WORD;
/*定义DWORD为e四个字节的类型*/
typedef unsigned long DWORD;
/*位图文件头*/
typedef struct BMP_FILE_HEADER {
	WORD bType; /* 文件标识符 */
	DWORD bSize; /* 文件的大小 */
	WORD bReserved1; /* 保留值,必须设置为0 */
	WORD bReserved2; /* 保留值,必须设置为0 */
	DWORD bOffset; /* 文件头的最后到图像数据位开始的偏移量 */
} BMPFILEHEADER;
/*位图信息头*/
typedef struct BMP_INFO {
	DWORD bInfoSize; /* 信息头的大小 */
	DWORD bWidth; /* 图像的宽度 */
	DWORD bHeight; /* 图像的高度 */
	WORD bPlanes; /* 图像的位面数 */
	WORD bBitCount; /* 每个像素的位数 */
	DWORD bCompression; /* 压缩类型 */
	DWORD bmpImageSize; /* 图像的大小,以字节为单位 */
	DWORD bXPelsPerMeter; /* 水平分辨率 */
	DWORD bYPelsPerMeter; /* 垂直分辨率 */
	DWORD bClrUsed; /* 使用的色彩数 */
	DWORD bClrImportant; /* 重要的颜色数 */
} BMPINF;
/*彩色表*/
typedef struct RGB_QUAD {
	WORD rgbBlue; /* 蓝色强度 */
	WORD rgbGreen; /* 绿色强度 */
	WORD rgbRed; /* 红色强度 */
	WORD rgbReversed; /* 保留值 */
} RGBQUAD;

double anss[M][256];
unsigned char ans1[M][256];
double ww1[M][256];
double wwturn1[M][256];
double tempx[M][256];
double pixel[M][N];
double simplingmatrix[RR][N];
double compressed_matrix[RR][N];
double ww[N][N], wwturn[N][N];
double temp1[RR][N];
float Simplingmatrix[RR * N], Compressed_matrix[RR * N];
float showmeans[256] = { -1 };
float showmemans[256 * 256] = { -1 };
float showmemans2[190 * 256] = { -1 };
int givemethevans[190] = { -1 };
int givemetheans=-1;

void testout(double** x, int height, int width, char*str);	//将指定矩阵输出至选中的文件中
void testoutx(double x[][N], int height, int width, char*str);	//将指定矩阵输出至选中的文件中
extern "C" void readbmp(FILE* fp, double pixel[][N]);	//读取图像
extern "C" void writebmp(FILE* fo, unsigned char pixel[][N], int height, int width);
void createsimplingmatrix(int wid, int height, double phi[][N]);	//生成观测矩阵
double error(double** piexl, double** ans, int height, int width);
void matrixmultiplicxx(double matrix1[][N], int width, double matrix2[][N],
	int height, int wid2, double ans[][N]) {
	int i, j, k;
	//	矩阵乘法
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			ans[i][j] = 0;
			for (k = 0; k < wid2; k++) {
				ans[i][j] += matrix2[i][k] * matrix1[k][j];
			}
		}
	}
}
void testoutcol(double* x, int height, char*str) {
	int i;
	FILE *fto = fopen(str, "wb");
	for (i = 0; i < height; i++) {
		fprintf(fto, "%f\t", x[i]);
	}
	fclose(fto);
}
double gaussrand();	//生成高斯矩阵

int main() {

	FILE *fp, *fo, *fp1;
	FILE *fp2, *fp3;
	BMPFILEHEADER fileHeader;
	BMPINF infoHeader;
	long width = 256, height = 256;
	int i, j;
	int temp[256 * 4];
	//double **pixel = NULL;
	//double **simplingmatrix = NULL;
	//double **compressed_matrix = NULL;

	for (i = 0;i < 256;i++)
		showmeans[i] = 0;
	double **ans = NULL;
	float ansx[M*N];
	//float* ansx;
	//ansx = (float*)malloc(M*N * sizeof(float));

	//printf("OK\n");
	//if ((fp = fopen("lena256.bmp", "rb")) == NULL) {
	//	printf("Cann't open lena256!\n");
	//	exit(0);
	//}
	//if ((fp1 = fopen("DWT.txt", "rb")) == NULL) {
	//	printf("Cann't open DWT.txt!\n");
	//	exit(0);
	//}
	//if ((fo = fopen("OK.bmp", "wb")) == NULL) {
	//	printf("Cann't open OK.bmp!\n");
	//	exit(0);
	//}
	////printf("OK\n");
	//fseek(fp, 0, 0);
	//fread(&fileHeader, sizeof(fileHeader), 1, fp);
	//fwrite(&fileHeader, sizeof(fileHeader), 1, fo);
	//fread(&infoHeader, sizeof(infoHeader), 1, fp);
	//fwrite(&infoHeader, sizeof(infoHeader), 1, fo);
	//fread(&temp, sizeof(unsigned int), 256, fp);
	//fwrite(&temp, sizeof(unsigned int), 256, fo);
	////printf("OK\n");
	//width = infoHeader.bWidth;
	//height = infoHeader.bHeight;
	////printf("OK\n");
	////creatmatrix(pixel,height, width);
	////if (pixel == NULL)
	////	return 0;
	//readbmp(fp, pixel);
	//printf("read OK");
	//createsimplingmatrix(RR, width, simplingmatrix);
	//matrixmultiplicxx(pixel, width, simplingmatrix, RR, width, compressed_matrix);
	//printf("OK\n");

	//creatmatrix(ww,M, N);
	//for (i = 0; i<M; i++)
	//for (j = 0; j<N; j++)
	//	fscanf(fp1, "%lf", &ww[i][j]);

	//matrixturn(ww, height, height, wwturn);
	//for (i = 0; i<M; i++)
	//for (j = 0; j < N; j++) {
	//	ww1[i][j] = ww[i][j];
	//	wwturn1[i][j] = wwturn[i][j];
	//}
	printf("OK\n");
	//matrixmultiplicxx(wwturn, width, compressed_matrix,
	//	RR, width,temp1);
	//for (i = 0;i < RR;i++) {
	//	for (j = 0;j < N;j++)
	//		compressed_matrix[i][j] = temp1[i][j];
	//}

	//matrixmultiplicxx(wwturn, width, simplingmatrix, RR, width, temp1);
	//for(i=0;i<RR;i++)
	//	for(j=0;j<N;j++)
	//simplingmatrix[i][j] = temp1[i][j];

	//ans = creatmatrix(height, width);
	if ((fp2 = fopen("s.txt", "rb")) == NULL) {
		printf("Cann't open DWT.txt!\n");
		exit(0);
	}

	for (i = 0; i<190; i++)
	for (j = 0; j<256; j++)
		fscanf(fp2, "%lf", &simplingmatrix[i][j]);
	if ((fp3 = fopen("c.txt", "rb")) == NULL) {
		printf("Cann't open DWT.txt!\n");
		exit(0);
	}
	for (i = 0; i<190; i++)
	for (j = 0; j<256; j++)
		fscanf(fp3, "%lf", &compressed_matrix[i][j]);
	//二维数组转一维数组
	//Simplingmatrix = (float*)malloc(RR * width * sizeof(float));
	//Compressed_matrix = (float*)malloc(RR * width * sizeof(float));
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < RR; j++)
		{
			Simplingmatrix[i * RR + j] = simplingmatrix[j][i];
		}
	}
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < RR; j++)
		{
			Compressed_matrix[i * RR+ j] = compressed_matrix[j][i];
		}
	}
	//显存分配并将已知数据传到GPU
	float *dev_Simpling, *dev_Compress;
	cudaMalloc((void**)&dev_Simpling, RR * width * sizeof(float));
	cudaMalloc((void**)&dev_Compress, RR * width * sizeof(float));
	cudaMemcpy(dev_Simpling, Simplingmatrix, RR * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Compress, Compressed_matrix, RR * width * sizeof(float), cudaMemcpyHostToDevice);

	float *dev_final_result;
	cudaMalloc((void**)&dev_final_result, N* sizeof(float));

	//中间变量显存分配，dev_Aug_t和dev_temp3为RR*256,其余256*256
	float *dev_Aug_t, *dev_Aug_tt, *dev_temp3, *dev_temp1, *dev_temp2;
	float *r_n_temp;
	cudaMalloc((void**)&r_n_temp, RR* sizeof(float));
	cudaMalloc((void**)&dev_Aug_t, RR * N * sizeof(float));
	cudaMalloc((void**)&dev_Aug_tt, N* N * sizeof(float));
	cudaMalloc((void**)&dev_temp3, RR * N * sizeof(float));
	cudaMalloc((void**)&dev_temp1, N* N * sizeof(float));
	cudaMalloc((void**)&dev_temp2, N* N * sizeof(float));
	float *judge2;
	judge2 = (float*)malloc(sizeof(float));
	//向量times*1
	float *dev_aug_y;
	//矩阵求逆中间变量声明与赋值
	float *a_transpose, *E, *A_t, *d_transpose, *a, *a_a, *a_t, *d;
	a_transpose = (float*)malloc(sizeof(float)*N*N);
	E = (float*)malloc(sizeof(float)*N*N);
	A_t = (float*)malloc(sizeof(float)*N*N);
	d_transpose = (float*)malloc(sizeof(float)*N*N);
	//a = (float*)malloc(sizeof(float)*N*N);
	//a_a = (float*)malloc(sizeof(float)*N*N);
	//a_t = (float*)malloc(sizeof(float)*N*N);
	d = (float*)malloc(sizeof(float)*N*N);
	float *dev_a, *dev_d, *dev_a_transpose, *dev_E, *dev_A_t, *dev_A_t_t, *dev_d_transpose;
	cudaMalloc((void**)&dev_a, N * N * sizeof(float));
	cudaMalloc((void**)&dev_d, N * N * sizeof(float));
	cudaMalloc((void**)&dev_a_transpose, N * N * sizeof(float));
	cudaMalloc((void**)&dev_E, N * N * sizeof(float));
	cudaMalloc((void**)&dev_A_t, N * N * sizeof(float));
	cudaMalloc((void**)&dev_A_t_t, N * N * sizeof(float));
	cudaMalloc((void**)&dev_d_transpose, N * N * sizeof(float));
	cudaMalloc((void**)&dev_aug_y, N * sizeof(float));
	int c = 0;
	//初始化对角矩阵
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<N; j++)
		{
			d[i*N + j] = 0;
		}
	}
	//初始化单位矩阵
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<N; j++)
		{
			if (i == j)
			{
				E[i*N + j] = 1;
			}
			else
			{
				E[i*N + j] = 0;
			}
		}
	}
	//初始化临时储存矩阵
	for (int i = 0; i<N; i++){
		for (int j = 0; j<N; j++)
		{
			A_t[i*N + j] = 0;

		}
	}
	cudaMemcpy(dev_d, d, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d_transpose, d, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_A_t, A_t, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_E, E, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_A_t_t, A_t, N * N * sizeof(float), cudaMemcpyHostToDevice);

	//最大值位置
	int *dev_pos;
	//最大位置对应数组
	int *dev_pos_array;
	//判断和标记结束
	int *dev_c;
	float *dev_judge1, *dev_judge2;
	//范数
	//float *dev_norm;
	//取列
	float *dev_column;
	float *dev_finishomp;
	float *dev_finish;
	float *dev_column_temp;
	float *dev_product_temp;
	float *dev_product;
	float *dev_r_n;
	float *dev_Simp_column;
	float zero[RR];
	float *dev_zero;
	for (i = 0; i < RR; i++)
	{
		zero[i] = 0;
	}
	cudaMalloc((void**)&dev_zero, RR * sizeof(float));
	cudaMalloc((void**)&dev_finishomp, M * N *sizeof(float));
	cudaMalloc((void**)&dev_finish, M * N *sizeof(float));
	cudaMemcpy(dev_zero, zero, RR*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_r_n, RR * sizeof(float));
	cudaMalloc((void**)&dev_Simp_column, RR * sizeof(float));
	cudaMalloc((void**)&dev_product_temp, RR * sizeof(float));
	cudaMalloc((void**)&dev_product, N * sizeof(float));
	cudaMalloc((void**)&dev_column, RR * sizeof(float));
	cudaMalloc((void**)&dev_column_temp, RR * sizeof(float));
	//cudaMalloc((void**)&dev_norm, sizeof(float));
	cudaMalloc((void**)&dev_pos, sizeof(int));
	cudaMalloc((void**)&dev_pos_array, RR * sizeof(int));
	cudaMalloc((void**)&dev_c, sizeof(int));
	cudaMalloc((void**)&dev_judge1, sizeof(float));
	cudaMalloc((void**)&dev_judge2, sizeof(float));
	for (i = 0;i < RR;i++)
		zero[i] = 0;
	printf("OMP start\n");
	int col;
	int times;
	for (i = 0; i < width; i++)
	{
		cudaMemcpy(dev_column, (dev_Compress + i * RR*sizeof(float)), RR*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_r_n, dev_column, RR * sizeof(float), cudaMemcpyDeviceToDevice);
		for (times = 1; times <= RR; times++)
		{
			for (col = 0; col < N; col++)
			{
				cudaMemcpy(dev_Simp_column, &dev_Simpling[col * RR], RR*sizeof(float), cudaMemcpyDeviceToDevice);
				vector_dot_product << <blocksPerGrid, threadsPerBlock >> >(dev_product_temp, dev_Simp_column, dev_r_n, RR);
				vector_dot_sum << <1, 1 >> >(dev_product_temp);
				cudaMemcpy(&dev_product[col], dev_product_temp, sizeof(float), cudaMemcpyDeviceToDevice);
			}
			d_find_max << <1, 1 >> >(dev_pos, dev_product, N);
			int pos = 0;
			cudaMemcpy(&pos, dev_pos, sizeof(int), cudaMemcpyDeviceToHost);

			cudaMemcpy(&dev_Aug_tt[(times - 1)*RR], &dev_Simpling[pos * RR], RR*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&dev_Simpling[pos*RR], dev_zero, RR*sizeof(float), cudaMemcpyDeviceToDevice);
			int ax = (times + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int bx = (RR + BLOCK_SIZE - 1) / BLOCK_SIZE;
			dim3 blocks(bx, ax);
			dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

			//cudaMemcpy(showmemans, dev_Aug_tt, N*N * sizeof(float), cudaMemcpyDeviceToHost);

			matrix_transpose << <blocks, threads >> >(dev_Aug_t, dev_Aug_tt, N, RR);

			matMult_gpu << < N, N, sizeof(float)*RR >> >(dev_Aug_tt, dev_Aug_t, dev_temp1, N, RR, N);

			cholesky << <1, THREAD_NUM, sizeof(float)*times >> >(dev_temp1, dev_d, dev_d_transpose, times);
			matrix_inversion << <(times + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> >(dev_a, dev_E, times);
			matrix_trans << <((times + THREAD_NUM - 1) / THREAD_NUM)*times, THREAD_NUM >> >(dev_E, dev_a_transpose, times);
			matrix_multiply << <times, THREAD_NUM, sizeof(float)*times >> >(dev_a_transpose, dev_d_transpose, dev_A_t, times);
			matrix_multiply << <times, THREAD_NUM, sizeof(float)*times >> >(dev_A_t, dev_E, dev_temp2, times);



			matMult_gpu << < N, RR, sizeof(float)*N >> >(dev_temp2, dev_Aug_tt, dev_temp3, N, N, RR);

			cudaMemcpy(showmemans, dev_temp2, N*N * sizeof(float), cudaMemcpyDeviceToHost);

			matrix_x_vector << < (N + THREAD_NUM - 1 / THREAD_NUM), THREAD_NUM >> >(dev_aug_y, dev_temp3, dev_column, RR);



			matrix_x_vector << < (RR + THREAD_NUM - 1 / THREAD_NUM), THREAD_NUM >> >(r_n_temp, dev_Aug_t, dev_aug_y,N);

			vector_sub_vector << <1, RR >> >(dev_r_n, dev_column, r_n_temp, RR);

			cudaMemcpy(&dev_pos_array[times - 1], dev_pos, sizeof(int), cudaMemcpyDeviceToDevice);

			valueMul << <1, 1 >> >(dev_judge1, dev_aug_y, (times - 1));
			vector_dot_product << <blocksPerGrid, threadsPerBlock >> >(dev_product_temp, dev_aug_y, dev_aug_y, N);
			vector_dot_sum << <1, 1 >> >(dev_product_temp);
			valueSqrt << <1, 1 >> >(dev_product_temp, dev_judge1, dev_judge2);
			cudaMemcpy(judge2, dev_judge2, sizeof(float), cudaMemcpyDeviceToHost);

			if ((*judge2) < 0.05)
			{
				c = times;
				break;
			}
		}
		result << <1, 1 >> >(dev_final_result, dev_aug_y, dev_pos_array, c);
		cudaMemcpy((dev_finishomp + i*N*sizeof(float)), dev_final_result, N*sizeof(float), cudaMemcpyDeviceToDevice);
	}
	printf("OMP end\n");
	int ax2 = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int bx2 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 blocks2(bx2, ax2);
	dim3 threads2(BLOCK_SIZE, BLOCK_SIZE);
	matrix_transpose << < blocks2, threads2 >> >(dev_finish, dev_finishomp, M, N);
	cudaMemcpy(ansx, dev_finish, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < M; i++)
	for (j = 0; j < N; j++)
		anss[i][j] = ansx[i*N + j];
	//anss[i][j] = ans[i][j];
	printf("preapare to out.txt\n");
	testoutx(anss, height, width, "ans.txt");
	////小波反变换
	//matrixmultiplicxx(anss, width, wwturn1, height, width, tempx);
	//matrixmultiplicxx(ww1, width, tempx, height, width, anss);
	//printf("小波变换完成\n");
	//for (i = 0; i < M; i++)
	//for (j = 0; j < N; j++)
	//	ans[i][j] = anss[i][j];
	//printf("ans 输出完成\n");

	//for (i = 0; i < M; i++) {
	//	for (j = 0; j < N; j++) {
	//		if (anss[i][j] > 255) {
	//			ans1[i][j] = 255;
	//		}
	//		else if (anss[i][j] < 0) {
	//			ans1[i][j] = 0;
	//		}
	//		else {
	//			ans1[i][j] = anss[i][j];
	//		}
	//	}
	//}
	//printf("准备输出图像\n");
	//writebmp(fo, ans1, height, width);
	return 0;
}
void testout(double** x, int height, int width, char*str) {
	int i, j;
	FILE *fto = fopen(str, "wb");
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fprintf(fto, "%f\t", x[i][j]);
		}
		fprintf(fto, "\n");
	}
	fclose(fto);
}
void testoutx(double x[][N], int height, int width, char*str) {
	int i, j;
	FILE *fto = fopen(str, "wb");
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fprintf(fto, "%f\t", x[i][j]);
		}
		fprintf(fto, "\n");
	}
	fclose(fto);
}
void readbmp(FILE* fp, double pixel[][N]) {
	int i, j;
	for (i = N - 1; i >= 0; i--) {
		for (j = 0; j < N; j++) {
			pixel[i][j] = (double)fgetc(fp);
		}
	}
	fclose(fp);
}
void createsimplingmatrix(int height, int wid, double phi[][N]) {
	int i, j;
	double x, p[RR][N];
	for (i = 0; i < height; i++)
	for (j = 0; j < wid; j++) {
		phi[i][j] = 0;
		p[i][j] = 0;
	}
	srand((int)time(NULL));
	for (i = 0; i < height; i++) {
		for (j = 0; j < wid; j++) {
			x = gaussrand();
			p[i][j] = x;
		}
	}
	for (i = 0; i < height; i++)
	for (j = 0; j < wid; j++) {
		phi[i][j] = p[i][j];
	}
}
double gaussrand() {
	static double U, V;
	static int phase = 0;
	double z;

	if (phase == 0)
	{
		U = (rand() + 1.1) / (RAND_MAX + 2.);
		V = rand() / (RAND_MAX + 1.);
		z = sqrt(-1 * log(U))* sin(2 * 3.141592654 * V);
	}
	else
	{
		z = sqrt(-2 * log(U)) * cos(2 * 3.141592654 * V);
	}

	phase = 1 - phase;
	return z;
}
void writebmp(FILE* fo, unsigned char pixel[][N], int height, int width) {
	int i, j;
	for (i = height - 1; i >= 0; i--) {
		for (j = 0; j < width; j++) {
			fwrite(&pixel[j][i], sizeof(unsigned char), 1, fo);
		}
	}
	fclose(fo);
}
double error(double** piexl, double** ans, int height, int width){
	double sum = 0, psnr;
	int i, j;
	for (i = 0; i < height; i++)
	for (j = 0; j < width; j++)
		sum += pow(fabs(ans[i][j] - piexl[i][j]), 2);
	psnr = 10 * log10(255 * 255 / (sum / height / width));
	return psnr;
}
