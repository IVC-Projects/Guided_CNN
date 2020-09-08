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

#include "av1/common/call_tensorflow.h"
/*
        author：cgy
        date：2018/9/22
        param：
                ppp： 指向
   encoder.c文件中cm->frame_to_show->y_buffer的地址，即重建图像的首地址；
                height：序列的高；
                width：序列的宽；

*/

#define PY_DE(OBJ) \
  if ((OBJ)) Py_DECREF((OBJ))

PyObject *pMod = NULL;
PyObject *load_model = NULL;
PyObject *predict = NULL;
PyObject *pParm = NULL;
PyObject *pArgs = NULL;

void TF_Init_Model(FRAME_TYPE frameType, int QP) {
  if (pMod != NULL) return;

  Py_SetPythonHome(L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu");

  Py_Initialize();
  Py_SetPath(
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/Lib;"
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/Lib/site-packages;"
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/DLLs");

  if (!Py_IsInitialized()) {
    printf("Initialize failed!");
  }
  char *path = NULL;
  path = getcwd(NULL, 0);
  printf("current working directory : %s\n", path);
  free(path);

  pMod = PyImport_ImportModule("cnn_CRLC");
  if (!pMod) {
    printf("Import Module failed!\n");
  }

  load_model = PyObject_GetAttrString(pMod, "init");
  if (!load_model) {
    printf("Import load_model Function failed!\n");
  }

  predict = PyObject_GetAttrString(pMod, "predict");
  if (!predict) {
    printf("Import predict_I Function failed!\n");
  }
  pParm = Py_BuildValue("(ii)", frameType, QP);
  PyEval_CallObject(load_model, pParm);  // 导入预训练的模型
}

void TF_Init_Models(FRAME_TYPE frameType, int QP) {
  if (pMod != NULL) return;

  Py_SetPythonHome(L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu");
  // Py_SetPythonHome(L"C:/Users/77103/.conda/envs/tensorflow-cpu");
  Py_Initialize();
  Py_SetPath(
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/Lib;"
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/Lib/site-packages;"
      L"C:/ProgramData/Anaconda3/envs/tensorflow-140-gpu/DLLs");
  /* Py_SetPath(
       L"C:/Users/77103/.conda/envs/tensorflow-cpu/Lib;"
       L"C:/Users/77103/.conda/envs/tensorflow-cpu/Lib/site-packages;"
       L"C:/Users/77103/.conda/envs/tensorflow-cpu/DLLs");*/

  if (!Py_IsInitialized()) {
    printf("Initialize failed!");
  }
  char *path = NULL;
  path = getcwd(NULL, 0);
  printf("current working directory : %s\n", path);
  free(path);

  pMod = PyImport_ImportModule("cnn_CRLC_v2");  // cnn_CRLC_v2
  if (!pMod) {
    printf("Import Module failed!\n");
  }

  load_model = PyObject_GetAttrString(pMod, "init");
  if (!load_model) {
    printf("Import load_model Function failed!\n");
  }

  predict = PyObject_GetAttrString(pMod, "predict");
  if (!predict) {
    printf("Import predict_I Function failed!\n");
  }
  pParm = Py_BuildValue("(ii)", frameType, QP);
  PyEval_CallObject(load_model, pParm);  // 导入预训练的模型
}

uint8_t **TF_Predict(uint8_t *ppp, int height, int width, int stride, int QP,
                     int frameType) {
  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(3);
  PyObject **lists = new PyObject *[height];
  // FILE *fBCnn = fopen("D:/AOMedia/aom_org/test_result/beforeCNN.yuv", "wb");
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(lists[i],
                    Py_BuildValue("i", *(ppp + j)));  //转化为python对象
      // fwrite(ppp + j, 1, 1, fBCnn);
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  // fclose(fBCnn);
  PyTuple_SetItem(pArgs, 0, list);
  PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", QP));
  PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", frameType));

  PyObject *presult = NULL;
  presult = PyEval_CallObject(predict, pArgs);
  uint8_t **rePic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint8_t[width];
  }
  uint8_t s;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s);
      rePic[i][j] = s;
    }
  }

  return rePic;
}

#if CRLC_LF
uint8_t **TF_Predict_CRLC(uint8_t *dgr, uint8_t *src, CRLCInfo *ci, int height,
                          int width, int dgr_stride, int src_stride, int QP,
                          int frameType) {
  pArgs = PyTuple_New(4);
  PyObject *dgrList = PyList_New(height);
  PyObject *srcList = PyList_New(height);
  PyObject **dgrLists = new PyObject *[height];
  PyObject **srcLists = new PyObject *[height];

  // FILE *fBCnn = fopen("D:/AOMedia/aom_org/test_result/beforeCNN.yuv", "wb");
  for (int i = 0; i < height; i++) {
    dgrLists[i] = PyList_New(0);
    srcLists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(dgrLists[i],
                    Py_BuildValue("i", *(dgr + j)));  //转化为python对象
      PyList_Append(srcLists[i],
                    Py_BuildValue("i", *(src + j)));  //转化为python对象
      // fwrite(ppp + j, 1, 1, fBCnn);
    }
    PyList_SetItem(dgrList, i, dgrLists[i]);
    PyList_SetItem(srcList, i, srcLists[i]);
    dgr += dgr_stride;
    src += src_stride;
  }
  // fclose(fBCnn);
  PyTuple_SetItem(pArgs, 0, dgrList);
  PyTuple_SetItem(pArgs, 1, srcList);
  PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", QP));
  PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", frameType));

  PyObject *presult = NULL;
  presult = PyEval_CallObject(predict, pArgs);
  uint8_t **rePic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint8_t[width];
  }
  uint8_t s;

  PyObject *pListPic = nullptr;
  PyObject *pListA = nullptr;

  if (presult && PyTuple_Check(presult)) {
    // printf("11111111111111111111111111111111111111111111111111\n");
    pListPic = PyTuple_GetItem(presult, 0);
    pListA = PyTuple_GetItem(presult, 1);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        PyArg_Parse(PyList_GetItem(PyList_GetItem(pListPic, i), j), "B", &s);
        rePic[i][j] = s;
        //  printf("%d,", rePic[i][j]);
        // rePic[i][j] =*(dgr + j);
      }
      // dgr += stride;
    }

    int A = 0;
    int channels = 2;
    for (int i = 0; i < ci->units_per_tile * channels; i++) {
      PyArg_Parse(PyList_GetItem(pListA, i), "i", &A);
      ci->unit_info[i / channels].xqd[i % channels] = A;
  /*    printf(" idx:%d,A[%d]:%d\t\t\t", i / channels, i % channels,
             ci->unit_info[i / channels].xqd[i % channels] );*/
      // printf(" idx:%d,A[%d]:%d\t\t\t", i / channels, i % channels, A);
      // f (i == 0) ci->xqd[i] = clamp(s, -32, 31);
    }
  }

  // printf("\n");

  // PY_DE(dgrList);
  // PY_DE(dgrLists);
  // PY_DE(srcList);
  // PY_DE(srcLists);
  // PY_DE(presult);

  return rePic;
}
#endif

uint8_t **TF_Predict_block_buf(uint8_t *ppp, int height, int width,
                               FRAME_TYPE fType) {
  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);
  PyObject **lists = new PyObject *[height];
  // FILE *fBCnn = fopen("D:/AOMedia/aom_org/test_result/beforeCNN.yuv", "wb");
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(
          lists[i],
          Py_BuildValue("i", *(ppp + j + i * width)));  //转化为python对象
      // fwrite(ppp + j, 1, 1, fBCnn);
    }
    PyList_SetItem(list, i, lists[i]);
    // PyList_Append(list, lists[i]);
  }
  // fclose(fBCnn);
  PyTuple_SetItem(pArgs, 0, list);

  PyObject *presult = NULL;

  if (fType == 1)
    presult = PyEval_CallObject(predict, pArgs);
  else
    presult = PyEval_CallObject(predict, pArgs);
  uint8_t **rePic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint8_t[width];
  }
  uint8_t s;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s);
      rePic[i][j] = s;
    }
  }

  return rePic;
}

uint16_t **TF_Predict_hbd(uint16_t *ppp, int height, int width, int stride) {
  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);
  PyObject **lists = new PyObject *[height];

  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(lists[i],
                    Py_BuildValue("i", *(ppp + j)));  //转化为python对象
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }

  PyTuple_SetItem(pArgs, 0, list);

  PyObject *presult = NULL;

  presult = PyEval_CallObject(predict, pArgs);

  uint16_t **rePic = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint16_t[width];
  }
  uint16_t s;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
    }
  }

  return rePic;
}

void TF_Predict_block(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                      int cur_buf_width, int stride, int q_index) {
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(3);

  // PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(AOM_ROOT));

  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);

  // PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", q_index));

  PyObject *presult = NULL;
  /* if (frame_type == KEY_FRAME) {
     presult = PyEval_CallObject(predict, pArgs);
   } else {*/
  presult = PyEval_CallObject(predict, pArgs);
  //}

  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B",
                  &buf[i][j]);
    }
  }
  // Py_Finalize();//关闭python解释器
}

void TF_return_block(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                     int cur_buf_width, int stride, int q_index) {
  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      buf[i][j] = *(ppp + j);
    }
    ppp += stride;
  }
  // Py_Finalize();//关闭python解释器
}

uint16_t **TF_Predict_block_hbd(uint16_t *ppp, int cur_buf_height,
                                int cur_buf_width, int stride) {
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(1);
  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);
  PyObject *presult = NULL;
  // if (frame_type == KEY_FRAME) {
  //  presult = PyEval_CallObject(predict, pArgs);
  //} else {
  presult = PyEval_CallObject(predict, pArgs);
  //}

  uint16_t **rePic = new uint16_t *[cur_buf_height];
  for (int i = 0; i < cur_buf_height; i++) {
    rePic[i] = new uint16_t[cur_buf_width];
  }
  uint16_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      // PyList_GetItem(PyList_GetItem(presult, i), j) mean presult(i,j)
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);

  // Py_Finalize();
  return rePic;
}

int init_python() {
  if (!Py_IsInitialized()) {
    /*const char *mypath = AOM_ROOT
        "/av1/common:"
        "/home/chenjs/.conda/envs/tf2/lib:"
        "/home/chenjs/.conda/envs/tf2/lib/python3.6:"
        "/home/chenjs/.conda/envs/tf2/lib/python3.6/site-packages:"
        "/home/chenjs/.conda/envs/tf2/lib/python3.6/lib-dynload";*/
    wchar_t *wcsmypath = (wchar_t *)malloc(sizeof(wchar_t) * 1024);
    // mbstowcs(wcsmypath, mypath, strlen(mypath));
    Py_SetPath(wcsmypath);
    free(wcsmypath);
    /*
    Py_SetPath(
        AOM_ROOT
        L"/av1/common:"
        "/usr/lib:"
        "/usr/lib/python3.6:"
        "/usr/lib/python3.6/site-packages:"
        "/usr/lib/python3.6/lib-dynload");
        */

    Py_Initialize();

    if (!Py_IsInitialized()) {
      printf("Python init failed!\n");
      return -1;
    }
  }
  return 0;
}

int finish_python() {
  // if (Py_IsInitialized()) {
  //   return Py_FinalizeEx();
  // }
  return 0;
}

uint8_t **call_tensorflow(uint8_t *ppp, int height, int width, int stride,
                          FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }
  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  // import python
  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire python !\n");
  // 获得TensorFlow函数指针
  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }

  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);  //以元组方式传参
  PyObject **lists = new PyObject *[height];
  // stringstream ss;
  //将图像缓冲区的数据读到列表中
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(lists[i],
                    Py_BuildValue("i", *(ppp + j)));  //转化为python对象
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);  //"list" is the input image

  PyObject *presult = NULL;

  // printf("\nstart tensorflow!\n");
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);  //将pArgs参数传递到Python中
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);  //将pArgs参数传递到Python中
  }

  /*
      Py_ssize_t q = PyList_Size(presult);
      printf("%d", q);
      */

  //需要定义一个二维数组；
  uint8_t **rePic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint8_t[width];
  }
  uint8_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);
  return rePic;
}

uint16_t **call_tensorflow_hbd(uint16_t *ppp, int height, int width, int stride,
                               FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }
  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire python !\n");
  // 获得TensorFlow函数指针
  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }

  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);  //以元组方式传参
  PyObject **lists = new PyObject *[height];
  // stringstream ss;
  // Read the data from y buffer into the list
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(
          lists[i],
          Py_BuildValue("i", *(ppp + j)));  // Convert to Python objects
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);  //将列表赋给参数

  PyObject *presult = NULL;

  // printf("\nstart tensorflow!\n");
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);  //将pArgs参数传递到Python中
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);  //将pArgs参数传递到Python中
  }

  //需要定义一个二维数组；
  uint16_t **rePic = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint16_t[width];
  }
  uint16_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      // PyList_GetItem(PyList_GetItem(presult, i), j)意味着presult的(i,j)位置
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);

  // Py_Finalize();//关闭python解释器
  return rePic;
}

void block_call_tensorflow(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                           int cur_buf_width, int stride, FRAME_TYPE frame_type,
                           int q_index) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return;
  }
  pModule = PyImport_ImportModule("TEST");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return;
  }

  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return;
  }

  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return;
  }
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(3);

  // PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(AOM_ROOT));

  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 1, list);

  PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", q_index));

  PyObject *presult = NULL;
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);
  }

  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B",
                  &buf[i][j]);
    }
  }
  // Py_Finalize();//关闭python解释器
}

uint16_t **block_call_tensorflow_hbd(uint16_t *ppp, int cur_buf_height,
                                     int cur_buf_width, int stride,
                                     FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }

  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_B");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }

  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(1);
  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);
  PyObject *presult = NULL;
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);
  }

  uint16_t **rePic = new uint16_t *[cur_buf_height];
  for (int i = 0; i < cur_buf_height; i++) {
    rePic[i] = new uint16_t[cur_buf_width];
  }
  uint16_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      // PyList_GetItem(PyList_GetItem(presult, i), j) mean presult(i,j)
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);

  // Py_Finalize();
  return rePic;
}
