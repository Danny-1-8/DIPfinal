# DIPfinal

在代码作业中，涉及cv2的所有函数（如medianblur，blockfilter等，但不包括read这种读图片的函数），我都用numpy自己实现了，只是因为跑起来太慢，**所以采用了cv2的版本**；不同版本所消耗的时间我都写在代码注释里了。
mydehaze.py，对应暗通道算法;
histo.py，对应直方图均衡化算法;
third.py，对应单色恢复算法;

**代码引用与参考声明：**
在mydehaze.py中，对于获取暗通道的函数与gfilter函数的定义，我参考了https://github.com/He-Zhang/image_dehaze    （我自己一开始先用numpy实现了一个暗通道函数，但是跑起来太耗时间了，不如参考版本的；对于gfilter函数，何恺明文章的图片里有源码，所以我就没有做过多改动：
）

在third.py中，关于白平衡的定义，我参考了  https://www.cnblogs.com/pear-linzhu/p/12453985.html

算法跑出来的所有去雾结果我都放在百度云盘中了；

benchmark.py是用来测图像的各种指标的。
