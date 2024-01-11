# DETR
DETR被认为是端到端的目标检测算法，因为它消除了传统目标检测算法中许多手工设计的组件，实现了完全的端到端训练。传统目标检测算法通常需要多个组件，如锚点生成、区域建议网络等，这些组件不是端到端的，需要手动优化。然而，DETR只使用卷积神经网络和Transformer编码器-解码器，通过注意力机制建模对象，在不需要手工设计组件的情况下，完成目标检测。因此，DETR是端到端的目标检测算法。

**DETR基本思想**

在DECoder中，第一个输入输出一个结果，类似于机器翻译，每一个翻译词预测他是什么。

在解码器中初始化100个向量，每个向量预测出一个坐标框，每一个再去做一个分类的一个概率值（boundingbox回归和cls分类。）

![image-20231017163108060](https://github.com/jiangsu415/DETR/assets/130949548/8bd0e818-434f-474c-bebf-01d0675e3791)


**DETR整体网络架构**
![image-20231017163317449](https://github.com/jiangsu415/DETR/assets/130949548/b0bf9881-6108-4daf-9911-6416177e3e1f)


backbond中首先通过CNN拿到每个Patch所对应的向量并且加上一个位置编码，Enconder获取到这个特征，包括每个Patch的特征和全局的特征。在Decoder中会初始化100个向量，这100个向量要去利用Encoder生成的出来的特征来决定如何进行重构，目的是学100个向量。

Encoder中提供了特征（k和V），Decoder提供了一个Q到Encoder中查比如说这个Q去图像块1中去查k1v1是不是物体，然后去k2v2,Q回去挨个查询是不是物。同样下一个继续去查，Decoder要利用Encoder提供的特征。在DETR中是并行的一个结构红黄蓝绿同时去问是否为我要查询的目标。100框同时出来最后再连接一个全连接层预测Boundbox和cls

![image-20231017164334858](https://github.com/jiangsu415/DETR/assets/130949548/54c9a25b-f559-4996-87cb-468fa036bba7)

Encoder完成的任务

做Self-Attention能把每一个注意力结果提取出来，相比于CNN，Transformer能让每一个物体所在区域是在哪，并且遮挡现象也不会产生干扰。得到各个目标的注意力结果，准备好特征，等解码器来选秀
![image-20231017165431367](https://github.com/jiangsu415/DETR/assets/130949548/47c4eeb9-720f-42de-a82e-04ebdeab4d1b)

网络架构

下面四个颜色可以理解为随机初始化四个向量，每一个向量去查每个位置是不是我所关注的物体，学习每个向量去关注特定的区域。这些向量初始化时直接用0作为一个初始化，向量为一个纯0向量，比如说760维的一个向量，他内部全都是0，并且0+位置编码，这个位置编码，为让其对位置比较敏感。
![image-20231017170824203](https://github.com/jiangsu415/DETR/assets/130949548/b533f2e2-5401-45a8-9021-18223401c436)

![image-20231017171032758](https://github.com/jiangsu415/DETR/assets/130949548/a6eb3e5e-ad5a-46c1-99bb-017f5d034871)

在第一个Self-Attention中，100个向量在蓝色区域内未使用Encoder提供的特征，而是使用了一个自注意力机制，相当于分配好每一个向量要管理的一个区域，在第二步时，我们只需要一个Query向量，第一步主要需要的整合好做好特征所对应的Query向量，在Muti-head Attention中的得到Encoder提供的K和V，还有第一步提供的Query，通过Q去看看K和V之间的关系，基于这个关系好去重构Query，最终通过一个Query连上一个FFN，分别做两次全链接，来得到Boundbox（回归）和cls（分类）。

传统的Mask机制，第一个向量做Attention时，在他眼里后面的向量要Mask掉，后面看不到，预测只能基于前面的向量去预测，第二个向量只能看到第一个。在DETR中设计了两种Mask，因为是并行的所以都是透明的。只有第一次做Self-Attention后面的都是Attention并且不断循环

输出匹配

训练之后输出恒是100个，但是标签只有两个，100个需要找两个做匹配，找最合适的两个做预测框，剩下的98个作为背景

匈牙利匹配
![image-20231017172919082](https://github.com/jiangsu415/DETR/assets/130949548/092a2c64-1153-42b2-a770-50e472233c43)


注意力起到的作用

![image-20231017173350342](https://github.com/jiangsu415/DETR/assets/130949548/c80dca97-9e76-4267-9ada-6a6375cbd8db)
