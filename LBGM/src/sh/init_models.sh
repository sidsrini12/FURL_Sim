mnist(){
    python3 init_models.py --dataset mnist --models fcn svm resnet18
}

fmnist(){
    python3 init_models.py --dataset fmnist --models fcn svm resnet18
}

cifar(){
    python3 init_models.py --dataset cifar --models fcn svm resnet18
}


$1
