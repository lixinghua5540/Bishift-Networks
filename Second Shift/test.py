import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import cv2
import PSNR
from util.visualizer import Visualizer


#用于测试的文件
def test(epoch):
    opt = TestOptions().parse()
    opt.which_epoch = epoch
    opt.nThreads = 1  # test code only supports nThreads = 1  测试时只支持为1
    opt.batchSize = 1  # test code only supports batchSize = 1 测试时只支持为1
    opt.serial_batches = True  # no shuffle 测试时不需要打乱顺序
    opt.no_flip = True  # 测试时也不需要翻转图像
    opt.display_id = -1  # no visdom display 测试时不需要可视化过程     以上的shuffle flip 都是测试时才用的
    opt.loadSize = opt.fineSize  # Do not scale!  #同样测试时不选哟变化图像尺度
    data_loader = CreateDataLoader(opt)  # 依据指定参数创建数据加载器
    dataset = data_loader.load_data()  # 加载数据
    model = create_model(opt)  # 指定参数创建模型

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):

        if i >= opt.how_many:
            # if i >= 1:
            break
        t1 = time.time()
        model.set_input(data)  # 设置网络模型的输入
        model.test()  # 计算
        t2 = time.time()
        print(t2 - t1)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)  # 保存图像

    # img_name = ['5']
    # for name in img_name:
    #     miss = 'results/exp/test_' + str(epoch) + '/images/' + str(name) + '_Missing.png'
    #     mask = 'masks/testing_masks/mask13.png'
    #     miss = cv2.imread(miss)
    #     mask = cv2.imread(mask)
    #     img = cv2.add(miss, mask)
    #     cv2.imwrite('results/exp/test_' + str(epoch) + '/images/' + str(name) + '_Missing.png', img)

    webpage.save()


if __name__ == "__main__":
    # epoch = 50
    # while epoch <= 2000:
    #     test(epoch)
    #     epoch += 50
    test('latest')





