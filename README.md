# chalearn_antispoof-competiton
This project is created for Chalearn Multi-Modality anti-spoof competition

Instructions on running the code

A.) Preprocessing the data (Run the following on the command line)
1.) python data_preprocess.py 
2.) python data_preprocess.py train 
3.) python data_preprocess.py train --no-enmfake
4.) python data_preprocess.py train --aug 
5.) cd data
6.) python im2rec.py train_depth_all_112_29266.lst ../phase1 
7.) python im2rec.py train_depth_noenmfake_112_15460.lst ../phase1 
8.) python im2rec.py train_depth_aug_112_38208.lst ../phase1 
9.) python im2rec.py val_depth_all_112_9608.lst ../phase1

B.) Training the data
1.) python train_depth_resnet.py
2.) python train_depth_resnet2.py
3.) python train_depth_densenet.py
4.) python train_depth_densenet_2.py
5.) python train_depth_shufflenet_v2.py
6.) python train_depth_vmspoofface.py
7.) python train_depth_vmspoofnet.py
8.) python train_depth_vmspoofnet_step2.py
9.) python train_depth_vmspoofnet_step3.py
10.) python train_depth_vmspoofnet_v2.py
11.) python train_depth_vmspoofnet_v2_step2.py

C.) Final Testing phase 
1.) python commit.py ../phase1/val_public_list.txt --load-epoch 73
2.) python commit_phase2.py ../phase2/test_public_list.txt --load-epoch 73

