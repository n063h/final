echo "testing cifar"
python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=nosemi label_ratio=0.01 notes=test_train_nosemi_cifar10
python train.py max_epochs=1  name=test_train dataset=cifar dataset.num_classes=100 model=resnet arch=nosemi label_ratio=0.01 notes=test_train_nosemi_cifar100

python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=pi label_ratio=0.01 notes=test_train_pi_cifar10
python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=fixmatch label_ratio=0.01 notes=test_train_fixmatch_cifar10
python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=pi arch.name=pi2 label_ratio=0.01 notes=test_train_pi2_cifar10
python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=pi arch.name=pi3 label_ratio=0.01 notes=test_train_pi3_cifar10
python train.py max_epochs=1  name=test_train dataset=cifar model=resnet arch=fixmatch arch.name=fixmatch2 label_ratio=0.01 notes=test_train_fixmatch2_cifar10


echo "testing single head nir name=test_train"
python train.py max_epochs=1  name=test_train dataset=nir dataset.axis=0 model=resnet1d arch=nosemi label_ratio=0.01 notes=test_train_nosemi_nir10
python train.py max_epochs=1  name=test_train dataset=nir dataset.axis=0 model=resnet1d model.name=2resnet1d arch=nosemi label_ratio=0.01 notes=test_train_nosemi_nir10_2resnet1d

python train.py max_epochs=1  name=test_train dataset=nir dataset.axis=0 dataset.num_classes=50 model=resnet1d arch=nosemi label_ratio=0.01 notes=test_train_nosemi_nir50
python train.py max_epochs=1  name=test_train dataset=nir dataset.axis=0 model=resnet1d arch=pi label_ratio=0.01 notes=test_train_pi_nir10
python train.py max_epochs=1  name=test_train dataset=nir dataset.axis=0 model=resnet1d arch=fixmatch label_ratio=0.01 notes=test_train_fixmatch_nir10


echo "testing three heads nir name=test_train"
python train.py max_epochs=1  name=test_train dataset=nir model=resnet1d arch=tri label_ratio=0.01 semi=False notes=test_train_nosemi_3nir10
python train.py max_epochs=1  name=test_train dataset=nir dataset.num_classes=50 model=resnet1d arch=nosemi label_ratio=0.01 semi=False notes=test_train_nosemi_3nir50

python train.py max_epochs=1  name=test_train dataset=nir model=resnet1d arch=tri arch.name=tri1 label_ratio=0.01 notes=test_train_tri1_3nir10
python train.py max_epochs=1  name=test_train dataset=nir model=resnet1d arch=tri arch.name=tri2 label_ratio=0.01 notes=test_train_tri2_3nir10
python train.py max_epochs=1  name=test_train dataset=nir model=resnet1d arch=tri arch.name=tri3 label_ratio=0.01 notes=test_train_tri2_3nir10