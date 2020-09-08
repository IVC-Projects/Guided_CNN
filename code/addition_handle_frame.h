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

#ifndef ADDITIONHANDLE_FRAME
#define ADDITIONHANDLE_FRAME

#include <stdint.h>

#include "av1/common/blockd.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/resize.h"
#include "av1/common/call_tensorflow.h"
#include "av1/encoder/encoder.h"
#include "aom_dsp/psnr.h"
#include "config/aom_scale_rtcd.h"
#ifdef __cplusplus
extern "C" {
#endif

// Minimum base_qindex needed to run CNN_RESTORATION.
#define MIN_CNN_Q_INDEX 100

static INLINE int av1_use_cnn(const AV1_COMMON *cm) {
  return (cm->base_qindex > MIN_CNN_Q_INDEX) && !av1_superres_scaled(cm);
}
uint8_t *getYbuf(uint8_t *yPxl, int height, int width, int stride);

void addition_handle_frame(AV1_COMMON *cm);
void addition_handle_frameT(AV1_COMMON *cm, int frame_type);

void addition_handle_blocks(AV1_COMMON *cm);

void addition_handle_blocks_rdo(YV12_BUFFER_CONFIG *source_frame,
                                AV1_COMMON *cm);
void addition_handle_blocks_adp(AV1_COMMON *cm, double threshold_MAD,int block_size);
void addition_handle_blocks_adp_loop(YV12_BUFFER_CONFIG *source, AV1_COMMON *cm,
                                     int block_size);

void addition_handle_blocks_rdo_frame(YV12_BUFFER_CONFIG *source_frame,
                                      AV1_COMMON *cm);
void addition_handle_frame_rdo(YV12_BUFFER_CONFIG *source_frame,
                               AV1_COMMON *cm);
#if CRLC_LF
void addition_handle_blocks_CRLC(YV12_BUFFER_CONFIG *source_frame,
                                 AV1_COMMON *cm);
void addition_handle_frame_CRLC(YV12_BUFFER_CONFIG *source_frame, AV1_COMMON *cm);

#endif
uint8_t **blocks_to_cnn_secondly(uint8_t *pBuffer_y, int height, int width,
                                 int stride, FRAME_TYPE frame_type);

void save_frame_yuv(char *file_name, YV12_BUFFER_CONFIG *buf, int frame_number);
void save_buf_to_yuv(uint8_t **buf, int height, int width, char *file_name,int flag);
double computeMSE(uint8_t **data1, uint8_t **data2, int height, int width);
void block_rdo(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *filter,
               YV12_BUFFER_CONFIG *cnn);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ADDITIONHANDLE_FRAME
