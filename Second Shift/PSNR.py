import cv2
import skimage
import numpy as np
from PIL import Image


def cor(img1, img2):
    return np.mean(np.multiply((img1-np.mean(img1)), (img2-np.mean(img2))))/(np.std(img1)*np.std(img2))


def cal(real, fake):
    img1 = cv2.imread(real)
    img2 = cv2.imread(fake)

    result1 = img1
    result2 = img2
    ssim = skimage.measure.compare_ssim(result1, result2, data_range=255, multichannel=True)
    psnr = skimage.measure.compare_psnr(result1, result2, data_range=255)
    cor = np.mean(np.multiply((img1-np.mean(img1)), (img2-np.mean(img2))))/(np.std(img1)*np.std(img2))

    # print('image:{}.png, SSIM:{}, PSNR:{}'
    #       .format(name, ssim, psnr))
    return ssim, psnr, cor


def save_eval(name):
    file = open('results/' + file_name + '/test_latest/evaluate.txt', 'w')

    real = 'results/' + file_name + '/test_latest/images/' + str(name) + '_Ground-truth.png'
    fake = 'results/' + file_name + '/test_latest/images/' + str(name) + '_Reconstruction.png'
    # mask = 'masks/testing_masks/' + str(name) + '_mask.png'
    before_trans = 'results/' + file_name + '/test_latest/images/before/' + str(name) + '.png'
    after_trans = 'results/' + file_name + '/test_latest/images/after/' + str(name) + '.png'
    real = cv2.imread(real)
    fake = cv2.imread(fake)
    # mask = cv2.imread(mask)
    after_trans = cv2.imread(after_trans)
    before_trans = cv2.imread(before_trans)

    #output to file
    file.write('image: {}.png'.format(name))
    file.write('\n')
    file.write('----before----')
    file.write('\n')
    file.write(str(skimage.measure.compare_psnr(real, before_trans, data_range=255)))
    file.write('\n')
    file.write(str(skimage.measure.compare_ssim(real, before_trans, data_range=255, multichannel=True)))
    file.write('\n')
    file.write(str(cor(real, before_trans)))
    file.write('\n')
    file.write('----style-transfer----')
    file.write('\n')
    file.write(str(skimage.measure.compare_psnr(real, after_trans, data_range=255)))
    file.write('\n')
    file.write(str(skimage.measure.compare_ssim(real, after_trans, data_range=255, multichannel=True)))
    file.write('\n')
    file.write(str(cor(real, after_trans)))
    file.write('\n')
    file.write('----shift-net----')
    file.write('\n')
    file.write(str(skimage.measure.compare_psnr(real, fake, data_range=255)))
    file.write('\n')
    file.write(str(skimage.measure.compare_ssim(real, fake, data_range=255, multichannel=True)))
    file.write('\n')
    file.write(str(cor(real, fake)))
    file.write('\n')
    file.write('\n')
    file.close()


#计算参数PSNR和SSIM,COR
if __name__ == "__main__":
    name = 5
    epoch = 50
    file_name = 'img5-mask13-dst+sn-5'
    while epoch <= 2000:
        COR = SSIM = PSNR = 0
        real = 'results/' + file_name + '/test_%d' % epoch + '/images/%s' % name + '_Ground-truth.png'
        fake = 'results/' + file_name + '/test_%d' % epoch + '/images/%s' % name + '_Reconstruction.png'

        ssim, psnr, cor = cal(real, fake)
        print(epoch, ssim, psnr, cor)
        epoch += 50


    # img1 = 's4/gt256.png'
    # img2 = 's4/ca.png'
    #
    # ssim, psnr, cor = cal(img1, img2)
    # print(ssim, psnr, cor)
