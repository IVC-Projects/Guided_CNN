import sys

if not hasattr(sys, 'argv'):
    sys.argv = ['']
import numpy.linalg as lg

from model.CRLC_v24 import crlc_model  as CRLC
from UTILS import *


np.set_printoptions(threshold=np.inf)
tplt1 = "{0:^30}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}"  # \t{4:^10}\t{5:^10}
tplt2 = "{0:^30}\t{1:^10}\t{2:^10}"

model_set = {
    '''论文所用模型'''
    "CRLC_C1_I_QP53":r"H:\KONG\train\\checkpoints\CRLC_I_Q37_09121801\CRLC_I_Q37_09121801_117_-43.53.ckpt",
    "CRLC4_C2_I_QP53":r"H:\KONG\train\\checkpoints\CRLC4_I_Q53_channel2_20010802\CRLC4_I_Q53_channel2_20010802_134_-43.27.ckpt",
    "CRLC4_I_QP53_C2_V3":r"H:\KONG\train\\checkpoints\CRLC4_I_QP53_C2_V3_2020011001\CRLC4_I_QP53_C2_V3_2020011001_118_-34.86.ckpt" ,
    "CRLC4_I_QP53_C2_V6":r"H:\KONG\train\\checkpoints\CRLC4_I_QP53_C2_V6_2020011101\CRLC4_I_QP53_C2_V6_2020011101_399_-34.00.ckpt",
    "CRLC4_I_QP32_C2_V6":r"H:\KONG\train\\checkpoints\CRLC4_I_QP32_C2_V6_2020011205\CRLC4_I_QP32_C2_V6_2020011205_314_-4.94.ckpt",
    "CRLC4_I_QP43_C2_V6":r"H:\KONG\train\\checkpoints\CRLCv6_I_QP43_C2_2020011501\CRLCv6_I_QP43_C2_2020011501_339_-14.27.ckpt",   #"H:\KONG\train\\checkpoints\CRLCv6_I_QP43_C2_2020011207\CRLCv6_I_QP53_C2_2020011207_399_-13.79.ckpt",
    "CRLC4_I_QP63_C2_V6":r"H:\KONG\train\\checkpoints\CRLCv6_I_Q63_channel2_20011208\CRLCv6_I_Q63_channel2_20011208_249_-56.56.ckpt",
    "CRLC4_I_QP53_C1_V6":r"H:\KONG\train\\checkpoints\CRLCv6_I_QP53_C1_2020011402\CRLCv6_I_QP53_C1_2020011402_399_-28.41.ckpt",
    "CRLC4_I_QP53_C3_V6":r"H:\KONG\train\\checkpoints\CRLCv6_I_QP53_C3_2020011402\CRLCv6_I_QP53_C3_2020011402_399_-31.91.ckpt",
    
    
    # '''测试模型：开deblock'''
    "CRLC4_I_QP53_C2_V6_db":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC4_I_QP53_C2_V6_20022001\CRLC4_I_QP53_C2_V6_20022001_281_-25.39.ckpt",
    "CRLCv6_I_QP47-55_C2_202103":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv6_I_QP47~55_C2_202103\CRLCv6_I_QP47~55_C2_202103_161_-22.26.ckpt",
    "CRLCv6_I_QP47-55_C2_202103_CNN1":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv6_I_QP47~55_C2_202103_CNN1\CRLCv6_I_QP47~55_C2_202103_CNN1_499_-14.08.ckpt",   #256 -20 5 -18 -3
    "CRLC4_I_allQP_C2_V6_20022502":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC4_I_allQP_C2_V6_20022502\CRLC4_I_allQP_C2_V6_20022502_136_-11.86.ckpt",             #512 -10 5 -8 7

    # inter+intra训练集
    "CRLC_v24_inter+intra20-32": r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC_v24_inter+intra20-32_20032701\CRLC_v24_inter+intra20-32_20032701_632_-7.57.ckpt",
    "CRLC_v24_inter+intra32-43": r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC_v24_inter+intra32-43_20032701\CRLC_v24_inter+intra32-43_20032701_630_-9.64.ckpt",
    "CRLC_v24_inter+intra43-53": r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC_v24_inter+intra43-53_20032601\CRLC_v24_inter+intra43-53_20032601_528_-7.28.ckpt",


    #intra训练集 v25
    "CRLCv25_I_QP17~26_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_I_QP17~26_C2\CRLCv24_I_QP17~26_C2_20032101_153_-1.30.ckpt",
    "CRLCv25_I_QP27~36_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_I_QP27~36_C2\CRLCv24_I_QP27~36_C2_20033001_180_-6.37.ckpt",
    "CRLCv25_I_QP37~46_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_I_QP37~46_C2\CRLCv24_I_QP37~46_C2_20032501_129_-9.98.ckpt",
    "CRLCv25_I_QP47~56_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_I_QP47~56_C2\CRLCv24_I_QP47~56_C2_20032501_174_-21.60.ckpt",




    #intra训练姐 v24
    "CRLCv24_I_QP7~16_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP7~16_C2\CRLCv24_real_I_QP7~16_C2_20040101_097_-0.49.ckpt",     #1024 0 15 -7 8
    "CRLCv24_I_QP17~26_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP17~26_C2\CRLCv24_real_I_QP17~26_C2_20040201_107_-1.42.ckpt",    #512 2 17 -3 12
    "CRLCv24_I_QP27~36_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP27~36_C2\CRLCv24_real_I_QP27~36_C2_20040701_165_-14.12.ckpt",    #8128 -28 -13 -7 8
    "CRLCv24_I_QP37~46_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP37~46_C2\CRLCv24_real_I_QP37~46_C2_20040301_178_-10.60.ckpt",   #512 -13 2 -13 2
    "CRLCv24_I_QP47~56_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP47~56_C2\CRLCv24_real_I_QP47~56_C2_20040301_194_-21.95.ckpt",    #1024 -15 0 -10 5
    "CRLCv24_I_QP57~66_C2":r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLCv24_real_I_QP57~66_C2\CRLCv24_real_I_QP57~66_C2_20040301_190_-39.37.ckpt"      #256 -9 6 -10 5

}

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 分配50%
config.gpu_options.allow_growth = True

global mCRLC
global cnn20,cnn32,cnn43
global cnn7, cnn17,cnn27,cnn37,cnn47, cnn57


def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = getWH(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = getWH(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            filesize = os.path.getsize(pair[0])
            picsize = getWH(pair[0])[0] * getWH(pair[0])[0] * 3 // 2
            numFrames = filesize // picsize
            # if numFrames ==1:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
            # else:
            #     while numFrames>0:
            #         or_imgY =getOneFrameY(pair[0])
            #         gt_imgY =getOneFrameY(pair[1])
            #         # normalize
            #         or_imgY = normalize(or_imgY)
            #
            #         or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            #         gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))
            #
            #         ## act as a placeholder
            #         or_imgCbCr = 0
            #         original_ycbcr.append([or_imgY, or_imgCbCr])
            #         gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Predict:
    input_tensor = None
    output_tensor = None
    model = None
    r = None
    gt = None
    R = None
    R_out = None

    def __init__(self, model, modelpath):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.model = model
        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            self.gt = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.r=tf.reshape(tf.subtract(self.gt,self.input_tensor),[1,tf.shape(self.input_tensor)[1]*tf.shape(self.input_tensor)[2],1])
            # self.R = tf.make_template('shared_model', self.model)(self.input_tensor)
            self.R = self.model(self.input_tensor)
            # R_ = tf.reshape(self.R, [1, tf.shape(self.input_tensor)[1] * tf.shape(self.input_tensor)[2], tf.shape(self.R)[-1]])
            # r_ = tf.reshape(tf.subtract(self.gt, self.input_tensor),
            #                 [1, tf.shape(self.input_tensor)[1] * tf.shape(self.input_tensor)[2], 1])
            #
            # self.A = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(R_, perm=[0, 2, 1]), R_)),
            #                         tf.transpose(R_, perm=[0, 2, 1])), r_)
            # self.A=tf.round(tf.multiply(tf.reshape(self.A,(1,tf.shape(self.R)[-1])),128))
            # A0=self.A
            # A1=self.A
            # A0 = tf.clip_by_value(A0, -8, 23)
            # A1 = tf.clip_by_value(A1, -16, 15)
            # self.A=[A0[0,0],A1[0,1]]
            # self.output_tensor = tf.add((tf.reduce_sum(tf.multiply(self.R, tf.multiply(self.A,1/128)),axis=3,keep_dims=True)), self.input_tensor)
            #self.output_tensor=self.R
            self.output_tensor = tf.multiply(self.R,255)
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph,config=config)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
                print(modelpath)

    def predict(self, fileOrDirX, fileOrDirY):
        imgX = None
        imgY = None

        if (isinstance(fileOrDirX, str)):
            original_ycbcr0, gt_y, fileName_list = prepare_test_data(fileOrDirX)
            original_ycbcr1, gt_y, fileName_list = prepare_test_data(fileOrDirY)
            imgX = original_ycbcr0[0][0]
            imgY = original_ycbcr1[0][0]

        elif type(fileOrDirX) is np.ndarray:

            imgX = fileOrDirX
            imgY = fileOrDirY

        elif (isinstance(fileOrDirX, list) and isinstance(fileOrDirY, list)):
            # print("model.predict", type(fileOrDirX), type(fileOrDirY))
            # print(len(fileOrDirX),len(fileOrDirX[0]))
            fileOrDirX = np.asarray(fileOrDirX, dtype='float32')
            fileOrDirY = np.asarray(fileOrDirY, dtype='float32')
            imgX = normalize(np.reshape(fileOrDirX, (1, len(fileOrDirX), len(fileOrDirX[0]), 1)))
            imgY = normalize(np.reshape(fileOrDirY, (1, len(fileOrDirY), len(fileOrDirY[0]), 1)))
            # imgX = np.reshape(fileOrDirX, (1, len(fileOrDirX), len(fileOrDirX[0]), 1))
            # imgY = np.reshape(fileOrDirY, (1, len(fileOrDirY), len(fileOrDirY[0]), 1))




        else:
            imgX = None
            imgY = None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = self.sess.run([self.output_tensor],
                                    feed_dict={self.input_tensor: imgX, self.gt: imgY})
                out = np.reshape(out, np.shape(out)[2:])
                return out




def init(sliceType, QP):
    # print("init !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    global mCRLC,cnn20,cnn32,cnn43,cnn17,cnn27,cnn37,cnn47,cnn7,cnn57
    QP=QP/4
    # if QP ==32:
    #     mCRLC = Predict(CRLC, model_set['CRLC4_I_QP32_C2_V6'])
    # elif QP ==43:
    #     mCRLC = Predict(CRLC, model_set['CRLC4_I_QP43_C2_V6'])
    # elif QP==53:
    #     mCRLC =Predict(CRLC,model_set['CRLC4_I_QP53_C2_V6_db'])   #CRLC4_I_QP53_C2_V6
    # elif QP >= 63:
    #     mCRLC = Predict(CRLC, model_set['CRLC4_I_QP63_C2_V6'])
    # mCRLC = Predict(CRLC, model_set['CRLCv6_I_QP47-55_C2_202103_CNN1'])
    #
    # cnn20 = Predict(CRLC, model_set["CRLC_v24_inter+intra20-32"])
    # cnn32 = Predict(CRLC, model_set["CRLC_v24_inter+intra32-43"])
    # cnn43 = Predict(CRLC, model_set["CRLC_v24_inter+intra43-53"])

    cnn7 =Predict(CRLC,model_set["CRLCv24_I_QP7~16_C2"])
    cnn17=Predict(CRLC,model_set["CRLCv24_I_QP17~26_C2"])
    cnn27=Predict(CRLC,model_set["CRLCv24_I_QP27~36_C2"])
    cnn37=Predict(CRLC,model_set["CRLCv24_I_QP37~46_C2"])
    cnn47=Predict(CRLC,model_set["CRLCv24_I_QP47~56_C2"])
    cnn57 = Predict(CRLC, model_set["CRLCv24_I_QP57~66_C2"])

def predict(dgr,src, QP, block_size=128):
    # print(psnr(dgr,src))
    QP = QP / 4

    if QP<17:
        scale = 1024  # 226144
        A0_min = 0
        A0_max = 15
        A1_min = -7
        A1_max = 8
    elif 17<=QP<27:
        scale = 512  # 226144
        A0_min = 2
        A0_max = 17
        A1_min = -3
        A1_max = 12
    elif 27<=QP<37:
        scale = 8128  # 226144
        A0_min = -28
        A0_max = -13
        A1_min = -7
        A1_max = 8
    elif 37<=QP<47:
        scale = 512  # 226144
        A0_min = -13
        A0_max = 2
        A1_min = -13
        A1_max = 2
    elif 47<=QP<57:
        scale = 1024  # 226144
        A0_min = -15
        A0_max = -0
        A1_min = -10
        A1_max = 5
    else:
        scale = 256  # 226144
        A0_min = -9
        A0_max = 6
        A1_min = -10
        A1_max = 5



    hei=np.shape(dgr)[0]
    wid=np.shape(dgr)[1]

    global mCRLC
    # block_size=128
    rows=math.ceil(float(hei)/block_size)
    clows=math.ceil(float(wid)/block_size)
    # R=mCRLC.predict(dgr,src)
    if QP  <17:
        R = cnn7.predict(dgr, src)
    elif 17 <= QP  <27:
        R = cnn17.predict(dgr, src)
    elif 27 <= QP < 37:
        R = cnn27.predict(dgr, src)
    elif 37 <= QP < 47:
        R = cnn37.predict(dgr, src)
    elif 47 <= QP < 57:
        R = cnn47.predict(dgr, src)
    else:
        R = cnn57.predict(dgr, src)


    dgr = np.asarray(dgr, dtype='float32')
    src = np.asarray(src, dtype='float32')
    # rec=np.zeros((np.shape(dgr)[0]+1,np.shape(dgr)[1]))
    rec = np.zeros(np.shape(src))
    A_list=[]
    for i in range(rows):
        for j in range(clows):
            start_row=end_row=start_clow=end_clow=0
            if i==rows-1:
                start_row=hei-block_size
                end_row=hei
            else :
                start_row=i*block_size
                end_row=(i+1)*block_size
            if j==clows-1:
                start_clow=wid-block_size
                end_clow=wid
            else :
                start_clow=j*block_size
                end_clow=(j+1)*block_size
            if hei<block_size:
                start_row=0
                end_row=hei
            if wid<block_size:
                start_clow=0
                end_clow=wid
            # print(start_row,end_row,start_clow,end_clow)
            sub_dgr=dgr[start_row:end_row,start_clow:end_clow]
            sub_src=src[start_row:end_row,start_clow:end_clow]
            sub_r=(sub_src-sub_dgr).flatten()
            sub_R=np.reshape(R[start_row:end_row,start_clow:end_clow,:],((end_clow-start_clow)*(end_row-start_row),np.shape(R)[-1]))
            A=lg.inv(sub_R.T.dot(sub_R)).dot(sub_R.T).dot(sub_r)*scale
            A=np.around(A)
            A[0]=np.clip(A[0],A0_min,A0_max)
            A[1]=np.clip(A[1],A1_min,A1_max)
            # print(A)
            A_list.append(A.astype('int'))
            sub_rec=sub_dgr+np.sum(np.multiply(R[start_row:end_row,start_clow:end_clow,:],A/scale),axis=2)

            rec[start_row:end_row,start_clow:end_clow]=sub_rec

    A_list = np.array(A_list).flatten()
    A_list = np.around(A_list)
    A_list = A_list.astype('int')
    A_list = A_list.tolist()
    # print(A_list,flush=True)

    rec = np.around(rec)
    rec = np.clip(rec, 0, 255)
    rec = rec.astype('int')
    rec = rec.tolist()
    # print("psnr",psnr(src,rec))
    # showImg(rec, r"H:\KONG\test_result\E20011601_pic\BasketballDrill_416x240_QP53_CRLC1.jpg")
    #
    # print(rec)
    return rec,A_list

def showImg(inp,name=r"D:/rec/compose.jpg"):
    h, w = inp[0], inp[1]
    tem = np.asarray(inp, dtype='uint8')
    # np.save(r"H:\KONG\cnn_2K%f" % time.time(),tem)
    tem = Image.fromarray(tem, 'L')
    tem.show()
    tem.save(name)


def test_all_ckpt(modelPath):
    global mCRLC
    # low_img = r"D:\konglingyi\Frame_1\av1_nofilter\QP53"
    low_img = r"H:\KONG\FRAME_!\AV!_deblock_nocdefLr\QP53"
    heigh_img = r"D:\konglingyi\Frame_1\ori"
    original_ycbcr, gt_y, fileName_list = prepare_test_data([low_img, heigh_img])
    total_imgs = len(fileName_list)

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    max_ckpt=0
    maxckpt_psnr=0
    for ckpt in ckptFiles:
        epoch = int(ckpt.split('.')[0].split('_')[-2])
        loss = int(ckpt.split('.')[0].split('_')[-1])
        #
        # if epoch !=147 :
        #     continue

        mCRLC = Predict(CRLC, os.path.join(modelPath, ckpt))
        # init(0,212)

        sum_img_psnr=0
        cur_ckpt_psnr=0
        img_index=[14,17,4,2,7,10,12,3,0,13,16,5,6,1,15,8,9,11]
        for i in img_index:
            # if i != 3:
            #     continue
            imgY = original_ycbcr[i][0]

            # print((imgY[0,0,0,:]))
            # exit()
            gtY = gt_y[i] if gt_y else 0
            # print(np.shape(imgY),np.shape(gtY))
            # showImg(denormalize(np.reshape(original_ycbcr[i][0], [480,832])))
            # showImg(np.reshape(gtY, [480,832]))
            # print(imgY.shape)



            # block_size=64
            # padding_size=8
            # sub_imgs=zip(divide_img(imgY,block_size,padding_size),divide_img((normalize(gtY)),block_size,padding_size))
            # recs=[]
            # for lowY,gY in sub_imgs:
            #     #print(type(predictor.predict(lowY, gY)))
            #     recs.append(predictor.predict(lowY, gY)[0])
            # rec=compose_img(imgY,recs,block_size,padding_size)
            # # print(psnr(np.reshape(denormalize(imgY), np.shape(rec)), np.reshape(gtY, np.shape(rec))))
            # cur_img_psnr = psnr(rec, np.reshape(gtY, np.shape(rec)))



            rec,_= predict(denormalize(imgY)[0,:,:,0].tolist(),gtY[0,:,:,0].tolist(),212,128)  # [:,:64,:64,:]
            cur_img_psnr = psnr(rec, np.reshape(gtY, np.shape(rec)))


            '''
            print(psnr(denormalize(np.reshape(imgY[:, :64, :64, :], [np.shape(imgY[:, :64, :64, :])[1],
                                                                     np.shape(imgY[:, :64, :64, :])[2]])),
                       np.reshape(gtY[:, :64, :64, :],
                                  [np.shape(imgY[:, :64, :64, :])[1], np.shape(imgY[:, :64, :64, :])[2]])))
            showImg(np.reshape(gtY[:, :64, :64, :], np.shape(rec)))
            cur_psnr[cnnTime] = psnr(rec, np.reshape(gtY[:, :64, :64, :], np.shape(rec)))
            '''
            # print(psnr(denormalize(np.reshape(imgY, [np.shape(imgY)[1], np.shape(imgY)[2]])),
            #            np.reshape(gtY, [np.shape(imgY)[1], np.shape(imgY)[2]])))

            sum_img_psnr =cur_img_psnr+sum_img_psnr
            print(tplt2.format(os.path.basename(fileName_list[i]), cur_img_psnr,psnr(denormalize(np.reshape(imgY, np.shape(rec))),np.reshape(gtY, np.shape(rec)))))
        cur_ckpt_psnr=sum_img_psnr/total_imgs
        if cur_ckpt_psnr>maxckpt_psnr:
            maxckpt_psnr=cur_ckpt_psnr
            max_ckpt=epoch
        print("cur_ckpt:",epoch, " agv_psnr: ", cur_ckpt_psnr,"max ckpt:",max_ckpt)





if __name__ == '__main__':


    # init(0,0)
    test_all_ckpt(r"D:\konglingyi\trainWork\train_resiMV\checkpoints\CRLC_v24_inter+intra43-53_20032601")
    # build_trainset()
