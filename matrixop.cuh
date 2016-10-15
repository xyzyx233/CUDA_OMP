/*
 * matrixop.h
 *
 *  Created on: 2016年4月5日
 *      Author: eric
 */
#pragma once

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define BB 256

void matrixturn(double matrix[][BB], int height, int width,double turn[][BB]);
double** matrixmultiplic(double** matrix1, int width, double** matrix2,
		int height,int wid2);	//后边的是矩阵相乘中前边的矩阵
void creatmatrix(double** pixel,int height, int width);	//生成指定大小的矩阵


