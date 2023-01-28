python test.py dataset=3cls10  model=3resnet1d arch=tri
python test.py dataset=cls10ww  model=resnet1d arch=pi
python test.py dataset=cls10ws  model=resnet1d arch=fixmatch
python test.py dataset=cls10  model=resnet1d arch=nosemi

python test.py dataset=3cls10  model=3resnet1d arch=tri label_ratio=0.1
python test.py dataset=cls10ww  model=resnet1d arch=pi label_ratio=0.1
python test.py dataset=cls10ws  model=resnet1d arch=fixmatch label_ratio=0.1
python test.py dataset=cls10  model=resnet1d arch=nosemi label_ratio=0.1

python test.py dataset=cifar10  model=resnet arch=nosemi
python test.py dataset=cifar10ww  model=resnet arch=pi
python test.py dataset=cifar10ws  model=resnet arch=fixmatch




shutdown