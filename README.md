# KerasExtras (Keras==2.2.4) with Tensorflow 2 support

Keras functions that you've always wanted!

* **BatchNormalization that supports float16!**
* Random cropping for augmentation
* Foreground sparse accuracy metric (if backgroud is set to '0')
* Background spare accuracy (if backgroud is set to '0')
* Sparse accuracy ignoring void (last) label
* Mean IOU (support for multi class labels)
* Segmentation generator: yields images, masks and sample weights
* Cyclic learning rate



Note: it works only when the output of the net is flattened. e.g., if the output is a mask of (M, N, #Classes) then you must add a reshape layer to make it (M*N, #Classes) and your labels are one hot encoded (and not sparse)
