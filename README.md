# Bishift-Networks
Title: *Bishift Networks for Thick Cloud Removal with Multitemporal Remote Sensing Images* [[paper]](https://www.hindawi.com/journals/ijis/2023/9953198/)<br>

Long C, Li X, Jing Y, et al. Bishift Networks for Thick Cloud Removal with Multitemporal Remote Sensing Images. International Journal of Intelligent Systems, vol. 2023, pp. 1-17, Art. no 9953198, 2023.
<br>
<br>
***Introduction***<br>
<br>
Because of the presence of clouds, the available information in optical remote sensing images is greatly reduced. These temporal-based methods are widely used for cloud removal. However, the temporal differences in multitemporal images have consistently been a challenge for these types of methods. Towards this end, a bishift network (BSN) model is proposed to remove thick clouds from optical remote sensing images. As its name implies, BSN is combined of two dependent shifts. Moment matching (MM) and deep style transfer (DST) are the first shift to preliminarily eliminate temporal differences in multitemporal images. In the second shift, an improved shift net is proposed to reconstruct missing information under cloud covers. It introduces multiscale feature connectivity with shift connections and depthwise separable convolution (DSC), which can capture local details and global semantics effectively. Through experiments with Sentinel-2 images, it has been demonstrated that the proposed BSN has great advantages over traditional methods and state-of-the-art methods in cloud removal.
<br>
<br>
![Fig](https://raw.githubusercontent.com/lixinghua5540/Bishift-Networks/master/BSN.jpg)<br>
<br>
<br>
***Usage***<br>
First Shift: The main program is photo ***style.py***, and the main function is ***main1()***. Enter the file name in the main function to run, the training parameters can be modified and added via ***parser.add_argument***.<br>
Second Shift: Use ***train.py*** for training, use ***test.py*** for testing, use ***options*** for parameter adjustment, and use ***models*** for model modification, where ***MyResUnet.py*** is a model built by yourself, which is referenced in the networks function.<br>
