<p align="center"><img width="90%" src="fig/title.jpg" /></p>

--------------------------------------------------------------------------------

Medical image registration is a typical two-image task which requires specialized feature representation networks for deep-learning-based methods (The existing methods and their limitations have been evaluated in our papers). Therefore, we designed a X-shape feature representation backbone which combines the relationship-aware capacity of Transformer and the traits of two-image tasks which foucus not only on structure information of each image but also on cross correspondence between the image pair. The overall structure of our network is following:

<p align="center"><img width="100%" src="fig/XMorpher.jpg" /></p>

# Paper
Open source for MICCAI2022 paper: [XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention]

# Citation
If you use this code or use our pre-trained weights for your research, please cite our papers:
```
@article{shi2022xmorpher,
  title={XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention},
  author={Shi, Jiacheng and He, Yuting and Kong, Youyong and Coatrieux, Jean-Louis and Shu, Huazhong and Yang, Guanyu and Li, Shuo},
  journal={arXiv preprint arXiv:2206.07349},
  year={2022}
}
```


# Available implementation
- MindSpore/
- Pytorch/


# Major results from our work
![plot](./fig/result.jpg)
