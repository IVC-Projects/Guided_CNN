/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef CALL_TENSORFLOW
#define CALL_TENSORFLOW

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#include "av1/common/blockd.h"
#include <stdio.h>
#include "av1/common/onyxc_int.h"
#include <Python.h>
#include <direct.h>

void TF_Init_Model(FRAME_TYPE frameType, int q_index);
void TF_Init_Models(FRAME_TYPE frameType, int q_index);
uint8_t **TF_Predict(uint8_t *ppp, int height, int width, int stride,
                     int q_index, int frame_type);
uint16_t **TF_Predict_hbd(uint16_t *ppp, int height, int width, int stride);
uint8_t **TF_Predict_block_buf(uint8_t *ppp, int height, int width,
                               int q_index);
void TF_Predict_block(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                      int cur_buf_width, int stride, int q_index);

void TF_return_block(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                     int cur_buf_width, int stride, int q_index);
#if CRLC_LF
uint8_t **TF_Predict_CRLC(uint8_t *dgr, uint8_t *src, CRLCInfo *ri,int height, int width,
                          int dgr_stride,int src_stride, int QP, int frameType);
#endif
uint16_t **TF_Predict_block_hbd(uint16_t *ppp, int cur_buf_height,
                                int cur_buf_width, int stride);

int init_python();
int finish_python();
uint8_t **call_tensorflow(uint8_t *ppp, int height, int width, int stride,
                          FRAME_TYPE frame_type);
void block_call_tensorflow(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                           int cur_buf_width, int stride, FRAME_TYPE frame_type,
                           int q_index);
uint16_t **call_tensorflow_hbd(uint16_t *ppp, int height, int width, int stride,
                               FRAME_TYPE frame_type);
uint16_t **block_call_tensorflow_hbd(uint16_t *ppp, int cur_buf_height,
                                     int cur_buf_width, int stride,
                                     FRAME_TYPE frame_type);
#ifdef __cplusplus
}
#endif
#endif  // CALL_TENSORFLOW
