# 测试模型
现在已经有一些模型被训练并且提供了下载下载
| Mask | Paris | CelebaHQ_256 |
| ----     | ----    | ---- |
| center-mask | ok | ok |
| random mask(from **partial conv**)| ok | ok |

有一个download_models.sh文件，在这个sh文件一般都是在linux中执行文件，windows中安装一些插件也可以运行，程序里的工程就是下载训练好的模型。
这里我们可以直接查看文件，里边有预训练模型的下载连接，直接使用下载连接去下载就好（需要翻墙 都已经下载好）
这里共有四个权重模型：
将下载文件的文件名`face_center_mask.pth` 重命名为 `30_net_G.pth`,并且把他们放在指定的目录下 `./log/face_center_mask_20_30`
将下载文件的文件名`face_random_mask.pth` 重命名为 `30_net_G.pth`,并且把他们放在指定的目录下 `./log/face_random_mask_20_30`
将下载文件的文件名`pairs_center_mask.pth` 重命名为 `30_net_G.pth`,并且把他们放在指定的目录下 `./log/pairs_center_mask_20_30`
将下载文件的文件名`pairs_random_mask.pth` 重命名为 `30_net_G.pth`,并且把他们放在指定的目录下 `./log/pairs_random_mask_20_30`
分别针对两个数据集和两种mask类型

将测试的图像分别放入datasets下不同的数据集目录下，

然后运行test.py文件
face_center测试参数：
  --name=face_center_mask_20_30 --dataroot=./datasets/celeba-256/test --offline_loading_mask=0 --mask_type=center

face_random测试参数:
 --name=face_random_mask_20_30 --dataroot=./datasets/celeba-256/test --mask_type=random

paris_center测试参数：
  --name=paris_center_mask_20_30 --dataroot=./datasets/paris/test --mask_type=center --offline_loading_mask=1 

paris_random测试参数:
 --name=paris_random_mask_20_30 --dataroot=./datasets/paris/test --mask_type=random --offline_loading_mask=1 

##name:加载模型的目录以及结果保存的目录        dataroot:测试的图像数据目录##

注意，上述测试使用的masks应该实现放在 `testing_mask_folder` 这个目录下（生成mask 的默认地址）
对于使用中间矩形mask训练的网络，要确保以下参数 `--mask_type='center' --offline_loading_mask=0`.？？
测试的结果将会保存到name的目录下，生成结果图像的同时，也会生成一个html文件，以便于对比查看结果

以上的测试是在线生生成mask测试


也可以再自己离线生成的mask上测试
运行generate_masks.py，生成对应的mask，这里需要设置参数
(1  要修改generate_masks.py中的参数test_folder 即生成mask的图像目录)
（2 在baseoptions中这是mask的生成类型）
生成mask后再运行test.py文件，根据使用的数据集和mask类型带上上面的参数，另外加上参数--offline_loading_mask=1





## 训练模型
- 首先下载自己的修复数据集，构建训练和测试集合，将训练图像放置到/datasets/dataset_name/train目录下。

- 比如在CelebaHQ_256数据集上训练一个模型:
我们挑选了 CelebaHQ_256数据集中的前2000张图像用于测试, 剩下的数据用于训练
python train.py --loadSize=256 --batchSize=1 --name=celeb256 --which_model_netG=unet_shift_triple --niter=30 --datarooot=./datasets/celeba-256/train
```
#niter 为训练的迭代次数
注意: **`loadSize` 在 face数据集中应该设置为`256` , 意味着直接将输入图像resize为 `256x256`.**


通常我们 train/test `使用center` mask.训练一个新的 shift-net`
python train.py --batchsize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --mask_type='center' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1


对于一些数据集比如 `CelebA`, 一些图像尺寸小于 `256*256`,所以你需要在训练时添加 `--loadSize=256`这个参数（这很重要）.
- 为了去可视化训练结果和损失，测开始训练前先在cmd中运行`python -m visdom.server` 并且点击连接 http://localhost:8097.
训练了的checkpoints默认的会存储在 `./log` 的目录下


对于`square` mask的训练不要设置tbatchsize 大于1 ，这样会使性能会减少很多(I don't know why...)
对于 `random mask`(`mask_sub_type不能为rect或者你自己的 random masks), 训练的batchsize如果大于1不会损失性能.

Random mask的训练(离线或者在线的mask) 都是支持的. 我建议你离线加载你的mask.



## 测试你自己训练好的模型
**测试的时候要保持和训练时候的参数保持一致，从而避免出现错误或者出现比较差的结果**

比如，如果你训练的时候网络模型为 `patch_soft_shiftnet`, 在测试阶段，如下的测试命令才是正确的
```bash
python test.py --fuse=1/0 --which_model_netG='patch_soft_unet_shift_triple' --model='patch_soft_shiftnet' --shift_sz=3 --mask_thred=4 
```



## 关于Masks
 **无比保持训练和测试时候设置的参数一致.**
 因为结果的效果与你在训练时使用的mask高度相关，所以训练和测试时的mask需要保持一致

| training | testing |
| ----     | ----    |
| center-mask | center-mask |
| random-square| All |
| random | All|
| your own masks| your own masks|


### 在线生成mask的训练
我们提供了三种类型的在线生成的mask `center-mask, random_square and random_mask`. (中心mask,随机矩形，随机mask)


### 在你自己的masks上进行训练
现在不管是在训练还是测试的时候都支持在线生成mask和离线加载mask (一个是预先生成好mask,一个是即时生成)
默认的是在线生成mask ,如果要训练/测试你自己生成的mask,设置参数 `--offline_loading_mask=1`
**预先准备好的mask应该放在 `--training_mask_folder` and `--testing_mask_folder`目录下.**

### 训练时的Masks
对于每一个batch:
 - Generating online: 一个batch中图像的mask是相同的(以减少计算量)
 - Loading offline: 一个batch中的每个图像的mask都是随机加载的

## 使用 Switchable Norm 而不是 Instance/Batch Norm（不同的标准化/正则化方法）
对于固定的mask的训练, 当batchSize > 1时，使用`Switchable Norm`方法会有更好的稳定性 . **当你在训练的时候，当batchsizes比较大时请使用 switchable norm方法 ,回避 instance norm or batchnorm方法更稳定!**



### 额外的一些参数变量

**以下的三个模型只是为了好玩**

对于`res patch soft shift-net`模型的训练参数为:
```
python train.py --batchSize=1 --which_model_netG=res_patch_soft_unet_shift_triple --model=res_patch_soft_shiftnet --shift_sz=3 --mask_thred=4
```
对于 `res navie shift-net`模型的训练参数为:
```
python train.py --which_model_netG=res_unet_shift_triple --model=res_shiftnet --shift_sz=1 --mask_thred=1
```

对于 `patch soft shift-net`模型的训练参数为:
```
python train.py --which_model_netG='patch_soft_unet_shift_triple' --model='patch_soft_shiftnet' --shift_sz=3 --mask_thred=4
```

不要改变 shift_sz 和 mask_thred的值. 否则出现错误的概率非常高

For `patch soft shift-net` or `res patch soft shift-net`. You may set `fuse=1` to see whether it delivers better results(Mention, you need keep the same setting between training and testing).


Paris StreetView Dataset不是一个公开的数据集,研究者如果需要要联系 pathak22. 你也可以使用 Paris Dataset 去训练你的模型