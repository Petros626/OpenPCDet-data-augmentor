# Installation updated (Anaconda env)

### Requirements
a. Create a conda environment
```shell
conda create --name openpcdet python=3.8
```

b. Uninstall of old CUDA version

Follow these guides to remove previous versions of CUDA.
https://gist.github.com/kmhofmann/cee7c0053da8cc09d62d74a6a4c1c5e4

https://medium.com/virtual-force-inc/a-step-by-step-guide-to-install-nvidia-drivers-and-cuda-toolkit-855c75efcdb6

With `nvidia-smi` you will see the max. supported version of CUDA. After you installed a proper versions verify with `nvcc --version` your fresh installed CUDA version.

c. Installation of OpenPCDet with Anaconda

```shell
conda activate openpcdet
```
Go to https://pytorch.org/ to copy the command for installing PyTorch with the corresponding CUDA version.

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install the SparseConv library from [`[spconv]`](https://github.com/traveller59/spconv). In the table you see CUDA version and the related installation command.
```shell
pip install spconv-cu118
```

### Install
Clone the OpenPCDet repository from my account and navigate into it.
```shell
git clone https://github.com/Petros626/OpenPCDet-data-augmentor.git
```
```shell
cd OpenPCDet
```
Install this `pcdet` library and its dependent libraries by running the following command
```shell
pip install -r requirements.txt
python setup.py develop
```



# Installation (old)

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv) or [`spconv v2.x`](https://github.com/traveller59/spconv)


### Install `pcdet v0.5`
NOTE: Please re-install `pcdet v0.5` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. Install the dependent libraries as follows:

[comment]: <> (* Install the dependent python libraries: )

[comment]: <> (```)

[comment]: <> (pip install -r requirements.txt )

[comment]: <> (```)

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 
    * You could also install latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
