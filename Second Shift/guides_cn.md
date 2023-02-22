#  Shift-Net结构
## 两个主要的文件
`train.py`, `test.py`.  这两个文件会调用其他文件夹的文件

##data文件夹
这这个文件夹中是加载训练/测试数据的代码，有几种不同的数据aligned_resized dataset，single_dataset和aligned_dataset,分别有各自的实现，
最后还用一个custom_dataset_data_loader类,将这几个数据的加载合并在一起，根据传参加载不同的数据。
## Data
数据处理的文件都放在 data文件夹下.其中 data/aligned_dataset也许是你唯一需要关注的一个文件.
在这个文件中，在调用 `data/image_folder.py文件中的`make_dataset函数，然后我们就会得到所有在`self.dir_A`中的图像
  Mention, as you can see `self.dir_A = os.path.join(opt.dataroot, opt.phase)` in Line 14. Therefore, you need to place all images of groundtruth images inside the folder `opt.dataroot/opt.phase`. Masked images are generated online during training and testing, and this will be illustrated in the following sections. **All preprocessing opertions on data should be  written in `data/aligned_dataset`.** Usually, random crop, resizing, flipping as well as normalization are processed. `__getitem__` returns a dict with two keys, stores a pair images, respectively representing `input` and `groundtruth`. **Batch images are loaded in this way:**, in `train.py`, it calls `CreateDataLoader` function in `data/data_loader`. `CreateDataLoader` firstly create an instance of `CustomDatasetDataLoader`, and intializes it, and finally returns the instance. `train.py` receives the instance and calls `load_data()`. Therefore, let's step into the `load_data()` in `CustomDatasetDataLoader` class in file `custom_dataset_data_loader.py`.
The `initialized()` is called in `CreateDataLoader`, this function adaptively selects the correct `data/aligned_dataset` or `data/single_dataset`, or `data/unaligned_dataset`(Deleted by me, as it is useless in my case). The instance of one of these classes is `dataset`, it will be passed in `torch.utils.data.DataLoader`. You can see:
```python
  self.dataloader = torch.utils.data.DataLoader(
      self.dataset,
      batch_size=opt.batchSize,
      shuffle=not opt.serial_batches,
      num_workers=int(opt.nThreads))
```
This is the code defining the `dataloader`.
You can see
```
def load_data(self):
    return self
```
Thus, it returns the whole class, making it easy for us to get the data in the `train.py`. In `train.py`, we get the `data`,
here, the `data` is dict contains `A` and `B`. Then we call `model.set_input(data)`, successfully pass the dict into `set_input()` in the file `shiftnet_model`.  More details will be illustrated below.


##datasets数据集
这个数据集是后来创建的，用于存放训练和测试的数据集，用数据集的名称和对应的train和test进行区分,在参数中可以指定数据集的目录

#imgs文件夹：
作者放的一些实验结果对比图

#log目录
用来保存训练的模型

#masks:
训练和测试使用的mask文件文件默认的保存目录，这个可以自己在参数中设置目录

#notebooks
这个不用管吧

#results
测试结果的默认保存目录，这个参数中也可以自行设置


## options文件夹
实验参数选项位于“options”文件夹中，
其中对于一些参数在测试和训练的过程中都会使用到，所以就把他们都放在了options/base_options文件里，这些选项包括了 `dataroot`, `which_model_netG`, `name`, `batchSize`, `gpu_ids`等,
包含在train_options.py的参数是训练的时候使用用到的参数
包含在test_options.py文件中的选项是测试的时候使用的参数。
有一些参数比如`niter`, `niter_decay`, `print_freq`, `save_epoch_freq` 等可能都是你会经常用的。
所有的这些参数都有一个默认值，可以直接到文件中修改参数的默认值，也可以在运行是传入参数，


##util文件夹
存放了一些工具类，
html.py,将测试的结果生成一个html文件，这样更便于观察对比结果
png.py, 生成png图像
visualizer.py,可视化工具，用于训练时候的模型，损失变化的可视化
util.py,一些小的工具类
poisson_blending.py:图像泊松融合
NonparametericsShift.py非参数移位算法


## Model 文件夹
- models 中提供了很多模型可供你采用，通常情况下，不需要关注它 just ignore it.

#modules中定义了各个各种模型中使用的模块
denset_net:定义了densenet（密集连接网络模块）模块，经典的cnn模块之一
losses:定义了损失函数
modules:一些小的模块，#L2标准化，#自注意层，#残差网络块等
unet:定义了Unet生成器网络模型
unet:定义了带shift的Unet生成器网络模型
discrimators#鉴别器



#patch_soft_shift
#res_patch_soft_shift
#res_shift_net
#Shiftnet_model
分别定义了四个不同的网络模型


### Shiftnet_model
`shiftnet_model` **是最主要的一个程序，你需要花时间去理解.**
- `base_model` 是 `shiftnet_model`的父类.  `shiftnet_model`没有继承父类中的三个函数.分别是 `save_network`, `load_network` and `update_learning_rate`. 因为这三个函数与您定义的任何模型都是兼容的 ，通常你可以忽略它, you can just ignore this script.
- 在initialize的函数中, 它定义了 G 和 D 网络, 两个优化器, 三个损失函数(Reconstruction loss, GAN loss and guidance loss),和 schedulers调度器. As our model accepts masked images of three channels,
- `set_input` get the `input` receving from `train.py` and aims at filling `mean value` in the mask region on the input images, code refering to `self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)`. Of course, the mask is generated online. By now, you will know, why we just let `B` the same context with `A`. `B` is the groundtruth image, and `A` is the context of `B` with mask region.

### 损失是如何实现的How guidance loss is implemented
The class of `guidance loss` is defined as `models/InnerCos.py`. As it is actually a `L2` constrain with novel target:
the encoder feature of groundtruth(B). This means, for the same input, the target changes as the parameters of network vary
in different iterations. Therefore, in each iteration, we firstly call `set_input`. This function also sets the `resized mask`(mask in shift layer) in `InnerCos`, which is essentially for shift operation. Then we call `set_gt_latent`, 
```python
    def set_gt_latent(self):
        self.netG.forward(Variable(self.input_B, requires_grad=False)) # input ground truth
        gt_latent = self.ng_innerCos_list[0].get_target()  # then get gt_latent(should be variable require_grad=False)
        self.ng_innerCos_list[0].set_target(gt_latent)
```
it acts as a role, providing the latent of encoder feature of groundtruth. Thus the dynamic `target` of `InnerCos` is obtained.
In the second iteration, we pass the `A` into the model as usual.
In `InnerCos.py`, we can see that this class mainly computes the loss of input and target, proving the gradient of guidance
loss.

### 模型在什么地方构建
模型在 `models/networks.py`中进行定义. Unet网络的结构很有趣， `UnetSkipConnectionBlock` 是 Unet网络的基本构成部分. Unet is constructed firstly from the innermost block, then we warp it with a new layer, it returns a new block on which we can continue warpping it with an instance of class `UnetSkipConnectionBlock`. When it reaches the outermost border, we can see:
```python
def forward(self, x):
    if self.outermost:  # if it is the outermost, directly pass the input in.
        return self.model(x)
    else:
        x_latter = self.model(x)
        _, _, h, w = x.size()

        if h != x_latter.size(2) or w != x_latter.size(3):
            x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
        return torch.cat([x_latter, x], 1)  # cat in the C channel
```
这很容易理解
对于shift模型, 我们需要添加 `Guidance loss layer` 和`shift layer`到模型中.
`UnetSkipConnectionShiftTripleBlock` 基于 `UnetSkipConnectionBlock`. It demonstrates distinctiveness in
```python
      # shift triple differs in here. It is `*3` not `*2`.
      upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1)
      down = [downrelu, downconv, downnorm]
      # shift should be placed after uprelu
      # Note: innerCos are placed before shift. So need to add the latent gredient to
      # to former part.
      up = [uprelu, innerCos, shift, upconv, upnorm]
```
As the network is defined in this way, it is not quite elegant to directly get specific layer of the model. Thus, we
pass in `innerCos_list` and `shift_list` as parameters. We build `guidance loss layer` and `shift layer` respectively by `InnerCos` and `InnerShiftTriple` in `UnetGeneratorShiftTriple`. `UnetGeneratorShiftTriple` is called in `define_G`. `define_G` decides which network architecture you will choose for our generative model. It returns `netG` as well as extra
two layers `innerCos_list` and `shift_list`. And finally, we can get these two special layers in `shiftnet_model`. So we
can construct the target of guidance loss layer with `set_gt_latent`. If you are still a bit confused, please refer to 
the above section **`How guidance loss is implemented`**.

### How shift is implemented
`InnerShiftTriple` and `InnerShiftTripleFunction` are the two main scripts. `InnerShiftTriple` get the features in `forward(self, input)`. As `input` here is the data consists of the concatenation of encoder feature with corresponding decoder feature. We split out `former_all` and `latter_all` in `forward` in `InnerShiftTripleFunction`. The known region of `latter_all` will be used to fill mask region of `former_all`. As for more details, it is a little bit complex, so please refer to the code.
