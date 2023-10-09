# training
python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4_1024 --name pathlogical_classic_cyclegan_he2cd4 --model classic_cycle_gan --gpu_id 1 --batch_size 16 --epoch_count 95 --epoch 95

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4 --model cycle_gan --gpu_id 1 --batch_size 16

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_byol_cyclegan_he2cd4_vit3 --model byol_cycle_gan --gpu_id 7 --batch_size 16 --netG resnet_6blocks_vit3

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD56 --name pathlogical_mask_56 --model cycle_gan --gpu_id 2 --batch_size 16 --epoch 105

# Training new model mask guided 
python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD56 --name pathlogical_mask_stain_others --model mask_cycle_gan --gpu_id 2 --batch_size 32

python new_train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD56 --name byol_mask_stain_others --model byol_cycle_gan --gpu_id 1 --batch_size 32

# testing
python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE_vessel --name pathlogical_classic_cyclegan_he2cd4 --model test --no_dropout

python test.py --dataroot /home1/qiuliwang/Data/BCI_dataset/test --name BCI_model --model test --no_dropout

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/final_test_cd4 --name pathlogical_classic_cyclegan_he2cd4 --model test --no_dropout --gpu_id 1,2 --epoch 200
python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/final_test_he --name pathlogical_mask_cyclegan_he2cd4 --model test --no_dropout --gpu_id 1,2 --epoch 200
python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/final_test_he --name pathlogical_mask_byol_cyclegan_he2cd4 --model test --no_dropout --gpu_id 1,2 --epoch 200

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE_vessel --name pathlogical_mask_byol_cyclegan_he2cd4 --model test --no_dropout



### multiscale
python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4_multiscale --model cycle_gan --gpu_id 1 --batch_size 16 --epoch_count 95 --epoch 95

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_classic_cyclegan_he2cd4_multiscale --model classic_cycle_gan --gpu_id 0 --batch_size 16 --epoch_count 115 --epoch 115

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/CD4_vessel --name pathlogical_classic_cyclegan_he2cd4_multiscale --model test --no_dropout --gpu_id 1,2 --epoch 100

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/CD4_vessel --name pathlogical_mask_cyclegan_he2cd4_multiscale --model test --no_dropout --gpu_id 1,2 --epoch 100

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE_vessel --name pathlogical_classic_cyclegan_he2cd4 --model test --no_dropout --gpu_id 1,2 --epoch 100


python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4_multiscale_transtest --model cycle_gan --gpu_id 1 --batch_size 16 

        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')


### Unet

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4_multiscale_unet --model cycle_gan --gpu_id 7 --batch_size 16 --netG unet_128

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE_vessel --name pathlogical_mask_cyclegan_he2cd4_multiscale_unet --model test --no_dropout --gpu_id 1,2 --epoch 100 --netG unet_128

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4/testA --name pathlogical_mask_cyclegan_he2cd4_multiscale_unet --model test --no_dropout --gpu_id 1,2 --epoch 100 --netG unet_128


python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4_multiscale_res6 --model cycle_gan --gpu_id 7 --batch_size 16 --netG resnet_6blocks


python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_cyclegan_he2cd4_multiscale --model cycle_gan --gpu_id 1,0 --batch_size 16 --netG unet_128


### classify
python new_train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_classify_cyclegan_he2cd4_multiscale --model multiscale_cycle_gan --gpu_id 0 --batch_size 16 --dataset_mode unaligned_classify


### transformer
python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name transformer_mask --model cycle_gan --gpu_id 0 --batch_size 16 --netG resnet_6blocks_vit

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name transformer_mask_V2 --model cycle_gan --gpu_id 1 --batch_size 16 --netG resnet_6blocks_vit2

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name transformer_mask_V3_windowsize7 --model cycle_gan --gpu_id 4,5 --batch_size 16 --netG resnet_6blocks_vit3

python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name transformer_mask_V4_windowsize4 --model cycle_gan --gpu_id 4 --batch_size 16 --netG resnet_6blocks_vit4

 --epoch_count 135 --epoch 135 --continue_train


python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4/testA --name transformer_mask --model test --no_dropout --gpu_id 1,2 --epoch 150 --netG resnet_6blocks_vit

python test.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE_vessel --name pathlogical_mask_cyclegan_he2cd4_multiscale --model test --no_dropout --gpu_id 4 --epoch 200 --netG resnet_6blocks_vit4


--epoch_count  --continue_train
pathlogical_mask_cyclegan_he2cd4_multiscale

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name pathlogical_classic_cyclegan_he2cd4 --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16 

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name pathlogical_mask_cyclegan_he2cd4_multiscale --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16 

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name pathlogical_mask_cyclegan_he2cd4_multiscale_unet --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16  --netG unet_128

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name transformer_mask_V3 --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16  --netG resnet_6blocks_vit3

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name transformer_mask_V3 --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16  --netG resnet_6blocks_vit3

python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name transformer_mask --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16  --netG resnet_6blocks_vit


python test.py --dataroot /home1/qiuliwang/Data/Glioma_annotation/test_13472_2048/inputs --name pathlogical_mask_byol_cyclegan_he2cd4_vit3 --model test --no_dropout --gpu_id 4 --epoch 200 --num_test 20 --batch_size 16  --netG resnet_6blocks_vit3


# loss weight self.criterionByolCD4 is 1
python train.py --dataroot /home1/qiuliwang/Data/Generative_Data/HE2CD4 --name pathlogical_mask_byol_cyclegan_he2cd4_vit3_weight --model byol_cycle_gan --gpu_id 0,1 --batch_size 16 --netG resnet_6blocks_vit3
