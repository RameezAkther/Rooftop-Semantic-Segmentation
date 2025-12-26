<h1 align="center">SegFormer-Pytorch</h1>

## [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203.pdf)

This is a warehouse for SegFormer-pytorch-model, can be used to train your image datasets for segmentation tasks.
The code mainly comes from official [source code](https://github.com/NVlabs/SegFormer).

### For ADE20K, COCO segmentation tasks, please refer this [github warehouse](https://github.com/jiaowoguanren0615/Segmentation_Factory)

## Project Structure

```
├── datasets: Load datasets
    ├── cityscapes.py: Customize reading cityscapes dataset
├── models: SegFormer Model
    ├── segformer.py: Construct "SegFormer" model
├── utils:
    ├── augmentations.py: Define transforms data enhancement methods
    ├── distributed_utils.py: Record various indicator information and output and distributed environment
    ├── losses.py: Define loss functions, focal_loss, ce_loss, Dice, etc
    ├── metrics.py: Compute iou, f1, pixel_accuracy.
    ├── optimizer.py: Get a optimizer (AdamW or SGD)
    ├── schedulers.py: Define lr_schedulers (PolyLR, WarmupLR, WarmupPolyLR, WarmupExpLR, WarmupCosineLR, etc)
    ├── utils.py: Define some support functions(fix random seed, get model size, get time, throughput, etc)
    ├── visualize.py: Visualize datasets and predictions
├── engine.py: Function code for a training/validation process
└── train_gpu.py: Training model startup file
```

## Precautions

This code repository is a reproduction of the **_SegFormer-MiT_** model based on the Pytorch framework, including **_MiT-B0~B5_**. It is currently only trained on the CityScape dataset. If you need to train your own dataset, please add a customized Mydataset in the **_datasets_** directory. class, and then modify the **_data_root_**, **_batchsize_**, **_epochs_** and other training parameters in the **_train_gpu.py_** file.

## Train this model

### Parameters Meaning:

```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```

### Note:

If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. **_Do not let batch_size=1 on each GPU_**, otherwise BN layer maybe report an error. If you recive an error like "**_error: unrecognized arguments: --local-rank=1_**" when you use distributed multi-GPUs training, just replace the command "**_torch.distributed.launch_**" to "**_torch.distributed.run_**".

### train model with single-machine single-GPU：

```
python train_gpu.py
```

### train model with single-machine multi-GPU：

```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU:

(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)

```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:

(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)

```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Inference 🔧

Use the included inference script to run a trained checkpoint on a folder of images and save predicted masks.

Example:

```bash
# predict using CPU
python SegFormer/inference.py --checkpoint ./save_weights/best_model.pth --input_dir ./test_images --output_dir ./preds --device cpu

# predict using GPU, save overlay images as well
python SegFormer/inference.py --checkpoint ./save_weights/best_model.pth --input_dir ./test_images --output_dir ./preds --device cuda:0 --save_overlay
```

The script will save per-image prediction masks with suffix `_pred.png` (and `_overlay.png` when `--save_overlay` is used).

## Citation

```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
