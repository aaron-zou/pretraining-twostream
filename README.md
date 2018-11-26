# pretraining-twostream

Cleaner Git repository to hold research work.

## Notes

Per the documentation for torchvision.models, all pretrained models expect "mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are at least 224. The images must be loaded in a range [0, 1] and normalized using(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])".

HMDB51 has a minimum frame count of 17 vs. 28 for UCF-101

## Acknowledgements

Note: all below repositories either have no license or use the MIT License, which I am also using, thereby complying with the provisions of the MIT license.

Using the starter template [pytorch-template](https://github.com/victoresque/pytorch-template) by [victoresque](https://github.com/victoresque) for TensorboardX visualization and useful structuring.

Credit to [twostreamfusion](https://github.com/feichtenhofer/twostreamfusion) by [feichtenhofer](https://github.com/feichtenhofer) for hosting compressed and processed UCF-101 and HMDB51 data.

Heavy credit to [two-stream-action-recognition](https://github.com/jeffreyhuang1/two-stream-action-recognition) by [jeffreyhuang1](https://github.com/jeffreyhuang1) as the base repo extended with HMDB51 support and custom use cases.

Credit to [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab) from [speedinghzl](https://github.com/speedinghzl) as I am using my own fork that adds ImageVid segmentation dataset.

