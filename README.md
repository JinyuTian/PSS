# PSS
The code of our our CVPR paper: Probabilistic Selective Encryption of Convolutional Neural Networks for Hierarchical Services 

# Probabilistic Selective Encryption of Convolution Neural Networks for Hierarchical Services


## Preliminaries

* Pytorch
 
## Datasets

* CIFAR-10

## Pretrained model

* VGG19, DnCNN
Pretrained DnCNN are provided in the file: `./PretrainedModel/`. 
Pretrained VGG19 are provided on https://drive.google.com/file/d/1qDWiwpgpUwn9ESXtu4BRbGgZsd-4b7RJ/view?usp=sharing

### Protect VGG19 (results in Fig. 3):
	ProtectVGG19.py
	Pretrained importance of paramters of VGG19 are provided in the file: `./Importance/`.
	python ProtectVGG19.py
	
### Hierarchical services of DnCNN:
	python HierarchicalServicesDnCNN.py





