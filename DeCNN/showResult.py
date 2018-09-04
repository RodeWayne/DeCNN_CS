# -*- coding: utf-8 -*-
from PIL import Image
from DeCNN import train_data_prepare as tdp
import numpy as np
import math
from DeCNN import DeconvNN_CS_1 as De1


def get_Result_img(imgArrayList, original_img, nn, crop_strides):
    img1List = get_imgArrayList_from_nn(imgArrayList=imgArrayList, nn=nn)
    img1List_ori = tdp.down_dim(img1List)
    result_img = tdp.paste_from_crops(img1List=img1List_ori, img=original_img)#默认无重叠
    #result_img = tdp.paste_from_crops_3(img1List=img1List_ori, img=original_img, crop_strides=crop_strides)
    return result_img


def calc_PSNR(generate_img, original_img):
    g_img_array = np.asarray(generate_img).astype(np.float32)
    #print("g_img_array.shape:", g_img_array.shape)
    #print("g_img_array:", g_img_array)
    o_img_array = np.asarray(original_img).astype(np.float32)
    #print("o_img_array.shape:", o_img_array.shape)
    #print("o_img_array:", o_img_array)
    sub_img_array = g_img_array - o_img_array
    img_array_pow = sub_img_array**2
    img_array_sum = np.sum(img_array_pow)
    MSE = img_array_sum / (original_img.size[0] * original_img.size[1])
    PSNR = 10 * math.log10((255.0*255.0)/MSE)

    return PSNR


def calc_half_PSNR(generate_img, original_img):
    half_weight = generate_img.size[0]/2
    height = generate_img.size[1]
    left_generate_img = generate_img.crop((0, 0, half_weight, height))
    left_original_img = original_img.crop((0, 0, half_weight, height))
    left_generate_img.show()
    left_original_img.show()
    PSNR = calc_PSNR(left_generate_img, left_original_img)
    return PSNR


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


def show_from_De1():
    Decnn = De1.myDeconvNN_CS
    dirs = "./Training_Data_L/Test/Set1/"
    imgList = tdp.eachFile(dirs)
    img = Image.open(imgList[0])
    img.show()
    imgArrayList = tdp.crop_img(imgList)
    imgArrayList_up = tdp.up_dim(imgArrayList)
    print("len(imgArrayList):", len(imgArrayList))
    result_img = get_Result_img(imgArrayList=imgArrayList_up, original_img=img, nn=Decnn)
    result_img.show()
    PSNR = calc_PSNR(result_img, img)
    print("PSNR:", PSNR)


def show_img_from_nn(nn):
    dirs = "./Training_Data_L/Test/Set1/"
    imgList = tdp.eachFile(dirs)
    img = Image.open(imgList[0])
    img.show()
    crop_strides = tdp.crops_width
    imgArrayList = tdp.crop_img(imgList, crop_strides=crop_strides)
    imgArrayList_up = tdp.up_dim(imgArrayList)
    result_img = get_Result_img(imgArrayList=imgArrayList_up, original_img=img, nn=nn, crop_strides=crop_strides)
    result_img.show()
    #denoised_img_array = BM3D_denoise.ge_final(result_img)
    #print("enoised_img_array.shape:", denoised_img_array.shape)
    #print("denoised_img_array:", denoised_img_array)
    #denoised_img = Image.fromarray(denoised_img_array)
    #print("show the denoised image.")
    #denoised_img.show()
    PSNR_mine1 = calc_PSNR(result_img, img)
    #PSNR_mine2 = calc_PSNR(denoised_img, img)
    print("PSNR_mine1 between nondenoised_img and ori_img:", PSNR_mine1)
    #left_img_PSNR = calc_half_PSNR(result_img, img)
    #print("left_img_PSNR:", left_img_PSNR)
    #print("PSNR_mine2 between denoised_img and ori_img:", PSNR_mine2)
    #PSNR1 = PSNR.PSNR(result_img, img)#图片格式不同导致改函数计算报错
    #PSNR2 = PSNR.PSNR(denoised_img, img)
    #print("PSNR1 between nondenoised_img and ori_img:", PSNR1)
    #print("PSNR1 between denoised_img and ori_img:", PSNR2)
if __name__ == '__main__':
    Decnn = De1.De_ops(init_op=False)
    #autoencoder = mdae2.my_ops(init_op=True)
    show_img_from_nn(nn=Decnn)
    '''
    img1_name = './Training_Data_L/result_of_reconNet_paper/mr_0_25_e410/flinstones.png'
    img2_name = './Training_Data_L/test_images/flinstones.png'
    img1 = Image.open(img1_name)
    img2 = Image.open(img2_name)
    print(calc_PSNR(img1, img2))
    '''

