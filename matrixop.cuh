/*
 * matrixop.h
 *
 *  Created on: 2016��4��5��
 *      Author: eric
 */
#pragma once

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define BB 256

void matrixturn(double matrix[][BB], int height, int width,double turn[][BB]);
double** matrixmultiplic(double** matrix1, int width, double** matrix2,
		int height,int wid2);	//��ߵ��Ǿ��������ǰ�ߵľ���
void creatmatrix(double** pixel,int height, int width);	//����ָ����С�ľ���


