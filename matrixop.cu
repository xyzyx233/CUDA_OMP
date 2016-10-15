/*
 * matrixop.c
 *
 *  Created on: 2016年4月5日
 *      Author: asus
 */

#include <math.h>
#include  <stdio.h>
#include <stdlib.h>
#include "matrixop.cuh"

void matrixturn(double matrix[][BB], int height, int width,double turn[][BB]) {
	int i = 0, j;
	for (i = 0; i < width; i++)
		for (j = 0; j < height; j++)
			turn[i][j] = matrix[j][i];
}
double** matrixmultiplic(double** matrix1, int width, double** matrix2,
	int height, int wid2) {
	int i, j, k;
	double** ans = NULL;
	ans = (double**)malloc(height * sizeof(double*));
	if (ans == NULL)
		return NULL;
	for (i = 0; i < height; i++) {
		ans[i] = (double*)malloc(width * sizeof(double));
		if (ans[i] == NULL)
			return NULL;
	}
	//	矩阵乘法
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			ans[i][j] = 0;
			for (k = 0; k < wid2; k++) {
				ans[i][j] += matrix2[i][k] * matrix1[k][j];
			}
		}
	}
	return ans;
}
void creatmatrix(double **pixel,int height, int width) {
	pixel = (double**)malloc(height * sizeof(double*));
	if (pixel == NULL) {
		printf("It is out of memory 1!\n");
		return ;
	}
	int i, j;
	for (i = 0; i < height; i++) {
		pixel[i] = (double*)malloc(width * sizeof(double));
		if (pixel[i] == NULL) {
			printf("It is out of memory %d!\n", i + 1);
			return ;
		}
	}
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			pixel[i][j] = 0;
		}
	}
}