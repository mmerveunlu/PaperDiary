# Vision/Image

* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    * proposed to handle degradation problem (deeper networks have higher training errors)
    * Residual Learning:
        * rather than expect stacked layers to approximate H(x), desired underlying mappings,
	we explicitly let these layers approximate a residual function F(x)=H(x)-x
        * If the identity mappings are optimal, the solvers may simply drive the weights
	of the multiple nonlinear layers towards zero to approach identity mappings.
    * Experiments: ImageNet
        * Comparison between PlainNet (18 and 34 layers) and ResNet (18 and 34 Layers )
        * 34Layer Resnet is better than 18Layer Resnet
        * 18Layer Resnet converges faster than 18Layer PlainNet
        * 34Layer Plain Net is worse than 18 Layer Plain Net
    * Experiments: CIFAR10, MSCOCO,PASCAL
    * 1st place in ILSVRC (2015)

* [Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)
    * The proposed method enables the optimization of the deep networks with a learned gating mechanism for regulating the information flow.
    * Additionally added two nonlinear transforms to the input, called Transform gate and Carry gate, shows how much of the input is transformed and carried to form the output
    * Experiments: MNIST, CIFAR 
        * Optimization: Highway is better than plain network when depth is increased
        * Classification: Highway is easier to train than FitNets  
    
* [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
    * Proposed method DenseNets
    * connects all layers directly each other to ensure maximum information flow
    * Each layer obtains additionnal inputs from all preceding layers and passes its own features maps to all others
    * For x_l is the output of the layer l, x_l = H_l([x_0x_1...x_(l-1)]) where H_l is the non-linear transformation
        * In DenseNets: H_l is the composite of thre functions: Batch Normalization, Relu and 3x3 ConvNet
    * Experiments: CIFAR, SVHN, ImageNet
        * state-of-the-art results in CIFAR10 and CIFAR100
        * DenseNets perform better as L and k increase (L number of layers, k number of feature maps)
    * Conclusion: Deep Supervision (??)
