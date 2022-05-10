# Learning Local Displacements for Point Cloud Completion

The implementation of our [paper](https://arxiv.org/abs/2203.16600) accepted in **CVPR** *2022* (*Conference on Computer Vision and Pattern Recognition*, IEEE) 

Authors:
[Yida Wang](https://wangyida.github.io), David Tan, [Nassir Navab](http://campar.in.tum.de/Main/NassirNavab) and [Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari)

BSD 2-Clause License Copyright (c) 2022, Yida Wang All rights reserved.

## Abstrarct
| Completing a car |  |
| :-: | :-- |
![teaser](readme_imgs/CVPR_teaser.png#center) | From the input partial scan to our object completion, we visualize the amount of detail in our reconstruction.

We propose a novel approach aimed at object and semantic scene completion from a partial scan represented as a 3D point cloud.
Our architecture relies on three novel layers that are used successively within an encoder-decoder structure and specifically developed for the task at hand.
The first one carries out feature extraction by matching the point features to a set of pre-trained local descriptors.
Then, to avoid losing individual descriptors as part of standard operations such as max-pooling, we propose an alternative neighbor-pooling operation that relies on adopting the feature vectors with the highest activations. Finally, up-sampling in the decoder modifies our feature extraction in order to increase the output dimension.
While this model is already able to achieve competitive results with the state of the art, we further propose a way to increase the versatility of our approach to process point clouds. To this aim, we introduce a second model that assembles our layers within a transformer architecture.
We evaluate both architectures on object and indoor scene completion tasks, achieving state-of-the-art performance.

## 3D local displacement
### Local displacement operator
| The operation |  |
| :-: | :-- |
![operator](readme_imgs/CVPR_graph_conv.png#center) | (a) *k*-nearest neighbor in reference to an anchor **f**; (b) displacement vectors around the anchor **f** + δ<sub>i</sub> and the corresponding weight σ<sub>i</sub>; and, (c) closest features for all i.

### Architectures
| The *direct* architectrue | The *transformer* architecture |
| :-: | :-: |
![direct](readme_imgs/CVPR_direct_architecture.png#center) | ![transformer](readme_imgs/CVPR_transformer_architecture.png#center)

### Qualitatives
#### Object completion
![objects](readme_imgs/CVPR_shapenet.png#center)

#### Semantic scene completion
![objects](readme_imgs/CVPR_scannet.png#center)

### Setup
with [Conda](https://docs.conda.io/en/latest/)
```bash
conda create --name disp3d pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda activate disp3d
pip install -r dependencies.txt
bash setup.sh
```

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch 8 --n_regions 1 --npoints 2048 4096 --dataset shapenet --savepath exp_shapenet --methods disp3d
```
Training with multiple GPU could be configured using `CUDA_VISIBLE_DEVICES=0,1,2,3 ...`. Optional approach should be indicated by `--methods`, some options are `disp3d` for this work, `folding` for FoldingNet, `atlas` for AtlasNet, `pcn` for PCN, `msn` for MSN, `grnet` for GRNet, `pointr` for PoinTr, `snowflake` for SnowflakeNet, `softpool` for SoftPoolNet, etc.

### Validation
```bash
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --npoints 2048 4096 --model log/exp_shapenet/network.pth --dataset shapenet --methods disp3d
```
The output point cloud will be stored in `./pcds` folder.

### Visualization
Render points with the help of spherical structures in Mitsuba.
```bash
cd render_mitsuba/
./render.sh -f ../pcds
```
To get false positive points on output rendered in red like Figure. 7 in our paper (default color is presenting its categorical labels), the option `with_fp` in colormap function need to get set to be **True** in `val.py`.
```python
from other_tools import colormap
pts_color = colormap.colormap(points, gt=ground_truth, gt_seg=segmentation, with_fp=False, dataset='shapenet'):
```

## Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2022displacement,
  title={Learning Local Displacements for Point Cloud Completion},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
