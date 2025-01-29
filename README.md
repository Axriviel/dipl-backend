# KeraSage - backend
Official repository for KeraSage backend.

Required libraries are in requirements.txt file.

Dockerfile and docker-compose include basic settings for deployment via docker.

Due to github file size limitations it is necessary to manualy place "cifar10_normalized.npz" file in /datasets/default, which is necessary for the default dataset selection to work. Example of how to create such file is available in /utils/cifar-example.py