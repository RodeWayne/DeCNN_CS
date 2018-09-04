# -*- coding: utf-8 -*-
from PIL import Image
from DeCNN import train_data_prepare as tdp
from DeCNN import DeconvNN_CS_1 as De1
from DeCNN import showResult as sR
import os
import time


def save_Result_img(imgArrayList, oriFileDir, saveDir, nn, crop_strides, start):
    img1List = get_imgArrayList_from_nn(imgArrayList=imgArrayList, nn=nn)
    end = time.clock()
    print("Run time:", (end-start))
    print("average Run time:", ((end-start)/12))
    img1List_ori = tdp.down_dim(img1List)
    tdp.paste_from_crops_to_fileDir(img1List=img1List_ori, oriFileDir=oriFileDir, saveDir=saveDir,
                                    crop_strides=crop_strides)
    return


def get_imgArrayList_from_nn(imgArrayList, nn):
    reconstructList = []
    batch_num = int(len(imgArrayList)/De1.batch_size)
    for i in range(batch_num):
        img_arr_batch = De1.get_block_from_data(imgArrayList, data_batch_size=De1.batch_size, j=i)#反卷积网络用的相同函数
        reconstruct_batch = nn.getReconstruct(img_arr_batch, De1.ge_csphi(ge_new=False))
        for re_img_array in reconstruct_batch:
            reconstructList.append(re_img_array)
    print("len(reconstructList):", len(reconstructList))
    return reconstructList


def save_reconImg_from_nn(nn):
    dirs = "./Training_Data_L/test_images/noiseless/"
    saveDir = './Training_Data_L/re_test_images/noiseless/v1/mr_0_01_e1000/'
    imgList = tdp.eachFile(dirs)
    #print("last one:", imgList[len(imgList)-1])
    crop_strides = tdp.crops_width
    imgArrayList = tdp.crop_img(imgList, crop_strides=crop_strides)
    imgArrayList_up = tdp.up_dim(imgArrayList)
    start = time.clock()
    save_Result_img(imgArrayList=imgArrayList_up, oriFileDir=dirs, saveDir=saveDir, nn=nn, crop_strides=crop_strides, start = start)
    return


def cal_each_mean_PSNR(oriImgDir, reconImgDir):

    oriImgFullPaths = tdp.eachFile(oriImgDir)
    reconImgFullPaths = tdp.eachFile(reconImgDir)
    #oriImgList = []
    #reconImgList = []
    oridict = {}
    redict = {}
    for oriImgFullPath in oriImgFullPaths:
        oriImgName = os.path.split(oriImgFullPath)[1].split(".")[0]
        oridict[oriImgName] = oriImgFullPath
        '''
         a.append(os.path.split(oriImgFullPath)[1])
        oriImg = Image.open(oriImgFullPath)
        oriImgList.append(oriImg)
        '''
    for reconImgFullPath in reconImgFullPaths:
        reconImgName = os.path.split(reconImgFullPath)[1].split('.')[0]
        redict[reconImgName] = reconImgFullPath
        '''
        b.append(os.path.split(reconImgFullPath)[1])
        reconImg = Image.open(reconImgFullPath)
        reconImgList.append(reconImg)
        '''
    '''
    print("ori:", a)
    print("re:", b)
    '''
    PSNR_dict = {}
    totalPSNR = 0
    #for i in range(len(oriImgList)):
    for i in oridict:
        #print(i)
        oriImg = Image.open(oridict[i])
        reImg = Image.open(redict[i])
        somePSNR = sR.calc_PSNR(generate_img=reImg, original_img=oriImg)
        #somePSNR = sR.calc_PSNR(generate_img=reconImgList[i], original_img=oriImgList[i])
        #PSNR_dict[os.path.split(oriImgFullPaths[i])[1]] = somePSNR
        PSNR_dict[i] = somePSNR
        totalPSNR += somePSNR
    #meanPSNR = totalPSNR/len(oriImgList)
    meanPSNR = totalPSNR / len(oridict)
    print("PSNR of each pair:", PSNR_dict)
    print("meanPSNR:", meanPSNR)
    #print("without flinstone, meanPSNR:", (totalPSNR-PSNR_dict['flinstones'])/(len(oridict)-1))
    return

if __name__ == '__main__':
    #Decnn = De1.De_ops(init_op=False)
    #save_reconImg_from_nn(nn=Decnn)

    oriImgDir = './Training_Data_L/test_images/noiseless/'
    reconImgDir = './Training_Data_L/re_test_images/noiseless/v1/mr_0_01_e1000/'
    print(reconImgDir)
    cal_each_mean_PSNR(oriImgDir=oriImgDir, reconImgDir=reconImgDir)

    #文件夹中图片放到标准尺寸背景中
    # dir1 = './Training_Data_L/reconTrain/'
    # dir2 = './Training_Data_L/reconTrainSTS/'
    # tdp.paste_entireImg_to_standardSizeImg(imgDirPath=dir1, saveDirPath=dir2)