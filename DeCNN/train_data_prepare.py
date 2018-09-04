from PIL import Image
import os
import numpy as np
import math
import cv2

crops_width = 34
crops_heigth = 34


# crop_strides = 14 #训练时使用，力求和ReconNet保持一致，便于比较
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    imgDir = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        imgDir.append(child)
    return imgDir


def eachFile2(filepath):
    pathDir = os.listdir(filepath)
    imgDirDict = {}
    for allDir in pathDir:
        name = allDir.split('.')[0]
        name_num = int(name)
        imgDirDict[name_num] = os.path.join('%s%s' % (filepath, allDir))
    return imgDirDict



def convert_RGB_L(fileNameDir, newFilePath):
    '''

    :param fileNameDir:RGB images’ Dir
    :return: None(but fuction will save the L model images.)
    '''
    for fileName in fileNameDir:
        img = Image.open(fileName)
        img_L = img.convert("L")
        subFileName = os.path.split(fileName)[1]
        newFileFullPath = os.path.join('%s%s' % (newFilePath, subFileName))
        img_L.save(newFileFullPath)
    return


def crop_img(fileNameDir, crop_strides=14):
    img1Dir = []
    for img in fileNameDir:
        img = Image.open(img)
        width = img.size[0]
        height = img.size[1]
        a = 0
        while crops_width + a * crop_strides <= width:
            img1 = img.crop((a * crop_strides, img.size[1] - crops_heigth, crops_width + a * crop_strides, img.size[1]))
            img1_array = np.asarray(img1)
            img1Dir.append(img1_array)
            a += 1
        b = 0
        while crops_heigth + b * crop_strides <= height:
            img1 = img.crop((img.size[0] - crops_width, b * crop_strides, img.size[0], crops_heigth + b * crop_strides))
            img1_array = np.asarray(img1)
            img1Dir.append(img1_array)
            b += 1
        img1 = img.crop((img.size[0] - crops_width, img.size[1] - crops_heigth, img.size[0], img.size[1]))
        img1_array = np.asarray(img1)
        img1Dir.append(img1_array)

        a = 0
        while crops_width + a * crop_strides <= width:
            b = 0
            while crops_heigth + b * crop_strides <= height:
                img1 = img.crop((a * crop_strides, b * crop_strides,
                                 crops_width + a * crop_strides, crops_heigth + b * crop_strides))
                img1_array = np.asarray(img1)
                # print("img1_array.shape:", img1_array.shape)#(crop_strides, crop_strides)
                img1Dir.append(img1_array)
                b += 1
            a += 1
    return img1Dir


#保留切片后的小图片，用于D-AMP等方法的复原
def save_img_pieces(subimgArrayList):
    img_pieces_path = "./Training_Data_L/test_img_pieces/"
    i = 1
    for img1Array in subimgArrayList:
        subimgArray = img1Array.astype(np.uint8)
        subimg = Image.fromarray(subimgArray)
        path2 = str(i) + '.tif'
        savepath = img_pieces_path + path2
        #subimg.show()
        subimg.save(savepath)
        i += 1
    return


def up_dim(img1Dir):
    img1Dir_up = []
    for img1_array_ori in img1Dir:
        img1_array = img1_array_ori.reshape(crops_heigth, crops_width, 1)
        img1Dir_up.append(img1_array)
    return img1Dir_up


def down_dim(img1Dir_up):
    img1Dir_ori = []
    for img1_array in img1Dir_up:
        img_array_ori = img1_array.reshape(crops_heigth, crops_width)
        img1Dir_ori.append(img_array_ori)
    return img1Dir_ori


# crop_strides = crops_width 无重叠，不去格线
def paste_from_crops(img1List, img, crop_strides=crops_width):
    '''

    :param img1List: an input imgArrayList(Here it belongs to one image).
    :param img: the exact original image.
    :return: a image(Image obj) pasted by subImages converted from input.
    '''

    # image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    image = img.crop()
    print("image.mode, image.size:", image.mode, image.size)
    # image = Image.new(img.mode, img.size)
    subImgList = []
    for img_array in img1List:
        img_array1 = img_array.astype(np.uint8)
        subImg = Image.fromarray(img_array1)
        subImgList.append(subImg)
    print("subImg.mode, subImg.size:", subImg.mode, subImg.size)
    aup = int((img.size[0] - crops_width) / crop_strides)
    bup = int((img.size[1] - crops_heigth) / crop_strides)
    print("aup, bup:", aup, bup)
    # 先把前K个边缘处的图像块粘贴好
    a = 0
    while a <= aup:
        box = (a * crop_strides, img.size[1] - crops_heigth, crops_width + a * crop_strides, img.size[1])
        subImg = subImgList[a]
        image.paste(subImg, box)
        a += 1
    b = 0
    while b <= bup:
        box = (img.size[0] - crops_width, b * crop_strides, img.size[0], crops_heigth + b * crop_strides)
        subImg = subImgList[(a + b)]
        image.paste(subImg, box)
        b += 1
    box = (img.size[0] - crops_width, img.size[1] - crops_heigth, img.size[0], img.size[1])
    subImg = subImgList[(a + b)]
    image.paste(subImg, box)
    l = a + b + 1  # 已经粘贴了多少块子图
    # 贴非边缘的
    a = 0
    while a <= aup:
        b = 0
        while b <= bup:
            i = a * (bup + 1) + b + l
            box = (a * crop_strides, b * crop_strides, a * crop_strides + crops_width, b * crop_strides + crops_heigth)

            if i < len(subImgList):
                subImg = subImgList[i]
                image.paste(subImg, box)
            b += 1
        a += 1
    image_array = np.asarray(image)
    cv2.imwrite("mine1.bmp", image_array)
    return image

def paste_entireImg_to_standardSizeImg(imgDirPath, saveDirPath):
    imgFullPaths = eachFile(imgDirPath)
    for imgFullPath in imgFullPaths:
        img = Image.open(imgFullPath)
        back_img = Image.new('L', (512, 512), 0)
        box = (0, 0, img.size[0], img.size[1])
        back_img.paste(img, box)
        savePath = saveDirPath + os.path.split(imgFullPath)[1]
        back_img.save(savePath)
    return


#无重叠，不去格线,针对整个文件夹的图片并保存
def paste_from_crops_to_fileDir(img1List, oriFileDir, saveDir, crop_strides=crops_width):

    subImgList = []
    #for img_array in img1List:
    for img_array1 in img1List:
        #img_array1 = img_array.astype(np.uint8)
        subImg = Image.fromarray(img_array1)
        subImgList.append(subImg)
    # image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    imgDir = eachFile(filepath=oriFileDir)
    for imgFullPathName in imgDir:
        img = Image.open(imgFullPathName)
        image = img.crop()
        print("image.mode, image.size:", image.mode, image.size)
        # image = Image.new(img.mode, img.size)
        aup = int((img.size[0] - crops_width) / crop_strides)
        bup = int((img.size[1] - crops_heigth) / crop_strides)
        print("len(subImgList):", len(subImgList))
        print("aup, bup:", aup, bup)
        # 先把前K个边缘处的图像块粘贴好
        a = 0
        while a <= aup:
            box = (a * crop_strides, img.size[1] - crops_heigth, crops_width + a * crop_strides, img.size[1])
            if a >= len(subImgList):
                return
            subImg = subImgList[a]
            image.paste(subImg, box)
            a += 1
        b = 0
        while b <= bup:
            box = (img.size[0] - crops_width, b * crop_strides, img.size[0], crops_heigth + b * crop_strides)
            if a+b >= len(subImgList):
                return
            subImg = subImgList[(a + b)]
            image.paste(subImg, box)
            b += 1
        box = (img.size[0] - crops_width, img.size[1] - crops_heigth, img.size[0], img.size[1])
        subImg = subImgList[(a + b)]
        image.paste(subImg, box)
        l = a + b + 1  # 已经粘贴了多少块子图
        # 贴非边缘的
        a = 0
        while a <= aup:
            b = 0
            while b <= bup:
                i = a * (bup + 1) + b + l
                box = (a * crop_strides, b * crop_strides, a * crop_strides + crops_width, b * crop_strides + crops_heigth)

                if i >= len(subImgList):
                    return
                subImg = subImgList[i]
                image.paste(subImg, box)
                b += 1
            a += 1
        savepath = saveDir + os.path.split(imgFullPathName)[1]
        #image.save(savepath)
        cv2.imwrite(savepath, np.asarray(image))#这里比先astype(np.uint8)再image.save()效果更好

        j = 0
        print("i=", i)
        while j <= i:
            subImgList.remove(subImgList[0])
            j += 1
        #image_array = np.asarray(image)
        #cv2.imwrite("mine1.bmp", image_array)
    return


# 当strides=crops_width/2时，用下面方式粘帖并用平均法初步去格线
# 实验证明该方法让效果不生反而急降
def paste_from_crops_2(img1List, img, crop_strides=int(crops_width / 2)):
    '''

    :param img1List: an input imgArrayList(Here it belongs to one image).
    :param img: the exact original image.
    :return: a image(Image obj) pasted by subImages converted from input.
    '''

    # image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    image = img.crop()
    print("image.mode, image.size:", image.mode, image.size)
    # image = Image.new(img.mode, img.size)
    subImgArrayList = []
    subImgList = []
    for img_array in img1List:
        ssubImgArrayList = []
        img_array1 = img_array.astype(np.uint8)
        i = 0
        while (i + 1) * crop_strides <= crops_heigth:
            j = 0
            while (j + 1) * crop_strides <= crops_width:
                ssub_img_array1 = img_array1[i * crop_strides:(i + 1) * crop_strides,
                                  j * crop_strides:(j + 1) * crop_strides]
                ssubImgArrayList.append(ssub_img_array1)
                j += 1
            i += 1
        subImgArrayList.append(ssubImgArrayList)
        subImg = Image.fromarray(img_array1)
        subImgList.append(subImg)
    # print("subImg.mode, subImg.size:", subImg.mode, subImg.size)
    aup = int((img.size[0] - crops_width) / crop_strides)
    bup = int((img.size[1] - crops_heigth) / crop_strides)
    print("aup, bup:", aup, bup)
    # 先把前K个边缘处的图像块粘贴好
    a = 0
    while a <= aup:
        box = (a * crop_strides, img.size[1] - crops_heigth, crops_width + a * crop_strides, img.size[1])
        subImg = subImgList[a]
        image.paste(subImg, box)
        a += 1
    b = 0
    while b <= bup:
        box = (img.size[0] - crops_width, b * crop_strides, img.size[0], crops_heigth + b * crop_strides)
        subImg = subImgList[(a + b)]
        image.paste(subImg, box)
        b += 1
    box = (img.size[0] - crops_width, img.size[1] - crops_heigth, img.size[0], img.size[1])
    subImg = subImgList[(a + b)]
    image.paste(subImg, box)
    l = a + b + 1  # 已经粘贴了多少块子图
    # 贴非边缘部分，伴随平均法去格线
    x = int(img.size[0] / crop_strides)
    y = int(img.size[1] / crop_strides)
    a = 0
    while a <= (x - 1):
        b = 0
        while b <= (y - 1):
            if a == 0:
                if b == 0:
                    c = l
                    newssubImageArray = subImgArrayList[c][0]
                elif b == y - 1:
                    c = b - 1 + l
                    newssubImageArray = subImgArrayList[c][2]
                else:
                    c = l + b - 1
                    newssubImageArray = (subImgArrayList[c][2] + subImgArrayList[c + 1][0]) / 2
            elif a == x - 1:
                if b == 0:
                    c = l + (a - 1) * (y - 1) + b
                    newssubImageArray = subImgArrayList[c][1]
                elif b == y - 1:
                    c = l + (a - 1) * (y - 1) + b - 1
                    newssubImageArray = subImgArrayList[c][3]
                else:
                    c = l + (a - 1) * (y - 1) + b - 1
                    newssubImageArray = (subImgArrayList[c][3] + subImgArrayList[c + 1][1]) / 2
            else:
                if b == 0:
                    c = l + (a - 1) * (y - 1)
                    newssubImageArray = (subImgArrayList[c][1] + subImgArrayList[c + y - 1][0]) / 2
                elif b == y - 1:
                    c = l + (a - 1) * (y - 1) + b - 1
                    newssubImageArray = (subImgArrayList[c][3] + subImgArrayList[c + y - 1][2]) / 2
                else:
                    c = (a - 1) * (y - 1) + b - 1 + l
                    newssubImageArray = (subImgArrayList[c][3] + subImgArrayList[c + 1][1] + subImgArrayList[c + y - 1][
                        2] +
                                         subImgArrayList[c + y][0]) / 4
            ssubImgArray1 = newssubImageArray.astype(np.uint8)
            ssubImg = Image.fromarray(ssubImgArray1)
            box = (a * crop_strides, b * crop_strides, (a + 1) * crop_strides, (b + 1) * crop_strides)
            image.paste(ssubImg, box)
            b += 1
        a += 1
    image_array = np.asarray(image)
    cv2.imwrite("mine2.bmp", image_array)
    return image


# crop_strides != crops_width, 但是不用平均法去格线
def paste_from_crops_3(img1List, img, crop_strides):
    '''

    :param img1List: an input imgArrayList(Here it belongs to one image).
    :param img: the exact original image.
    :return: a image(Image obj) pasted by subImages converted from input.
    '''

    # image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    image = img.crop()
    print("image.mode, image.size:", image.mode, image.size)
    # image = Image.new(img.mode, img.size)
    subImgList = []
    for img_array in img1List:
        img_array1 = img_array.astype(np.uint8)
        subImg = Image.fromarray(img_array1)
        subImgList.append(subImg)
    print("subImg.mode, subImg.size:", subImg.mode, subImg.size)
    aup = int((img.size[0] - crops_width) / crop_strides)
    bup = int((img.size[1] - crops_heigth) / crop_strides)
    print("aup, bup:", aup, bup)
    # 先把前K个边缘处的图像块粘贴好
    a = 0
    while a <= aup:
        box = (a * crop_strides, img.size[1] - crops_heigth, crops_width + a * crop_strides, img.size[1])
        subImg = subImgList[a]
        image.paste(subImg, box)
        a += 1
    b = 0
    while b <= bup:
        box = (img.size[0] - crops_width, b * crop_strides, img.size[0], crops_heigth + b * crop_strides)
        subImg = subImgList[(a + b)]
        image.paste(subImg, box)
        b += 1
    box = (img.size[0] - crops_width, img.size[1] - crops_heigth, img.size[0], img.size[1])
    subImg = subImgList[(a + b)]
    image.paste(subImg, box)
    l = a + b + 1  # 已经粘贴了多少块子图
    # 贴非边缘的
    a = 0
    while a <= aup:
        b = 0
        while b <= bup:
            i = a * (bup + 1) + b + l
            box = (a * crop_strides, b * crop_strides, a * crop_strides + crops_width,
                   b * crop_strides + crops_heigth)
            if i < len(subImgList):
                subImg = subImgList[i]
                image.paste(subImg, box)
            b += 1
        a += 1
    image_array = np.asarray(image)
    cv2.imwrite("mine3.bmp", image_array)
    return image


def calc_PSNR(generate_img, original_img):
    g_img_array = np.asarray(generate_img).astype(np.float32)
    o_img_array = np.asarray(original_img).astype(np.float32)
    sub_img_array = g_img_array - o_img_array
    img_array_pow = sub_img_array ** 2
    img_array_sum = np.sum(img_array_pow)
    MSE = img_array_sum / (original_img.size[0] * original_img.size[1])
    PSNR = 10 * math.log10((255.0 * 255.0) / (MSE + 0.00000000000001))
    return PSNR

if __name__ == '__main__':
    '''
    #裁剪test_images且保存
    dirs = "./Training_Data_L/test_images/"
    imgNameList = eachFile(dirs)
    subimgArrayList = crop_img(fileNameDir=imgNameList, crop_strides=crops_width)
    save_img_pieces(subimgArrayList=subimgArrayList)
    '''

    dirs = './Training_Data_L/test_img_pieces_D-AMP/iter20/0_01_p/'
    imgNameDict = eachFile2(dirs)
    imgNameDict1 = sorted(imgNameDict.items(), key=lambda imgNameDict: imgNameDict[0], reverse=False)
    #print("imgNameDict1:", imgNameDict1)
    #sorted(imgNameList)
    #print("imgNameDict:", imgNameDict)
    img_pieceArrayList = []
    for k in range(len(imgNameDict1)):
        img_piece = Image.open(imgNameDict1[k][1])
        img_pieceArray = np.asarray(img_piece)
        img_pieceArrayList.append(img_pieceArray)
    testImageDir = './Training_Data_L/test_images/'
    saveDir = './Training_Data_L/re_test_images_D-AMP/iter20/mr_0_01/'
    paste_from_crops_to_fileDir(img1List=img_pieceArrayList, oriFileDir=testImageDir, saveDir=saveDir)




'''
dirs = "/home/rode/下载/ReconNet/Training_Data_L/Test/Set1/"
imgNameList = eachFile(dirs)
img = Image.open(imgNameList[0])
print(img.format, img.size, img.mode)
print("img.size[0]:", img.size[0])
img.show()
img1List = crop_img(imgNameList)
img1List_up = up_dim(img1List)
img1List_ori = down_dim(img1List_up)
print("img1List[0].shape:", img1List[0].shape)
print("img1List_up[0].shape:", img1List_up[0].shape)
print("img1List_ori[0].shape:", img1List_ori[0].shape)
print(img1List[0])
print(img1List_ori[0])
print(img1List_up[0])
if img1List_ori[0][15][20] == img1List_up[0][15][20][0]:
    print("True")
reImg = paste_from_crops(img1List_ori, img=img)
print(reImg.format, reImg.size, reImg.mode)
reImg.show()
'''

'''
dirs = "/home/rode/下载/ReconNet/Training_Data_L/Test/Set1/"
imgNameList = eachFile(dirs)
img = Image.open(imgNameList[0])
print(img.format, img.size, img.mode)
img1List = crop_img(imgNameList)
print("len(img1List):", len(img1List))
reImg = paste_from_crops(img1List, img=img)
reImg.show()
PSNR = calc_PSNR(generate_img=reImg, original_img=img)
print("PSNR:", PSNR)
'''










