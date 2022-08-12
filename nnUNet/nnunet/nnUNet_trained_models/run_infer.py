import os

main_dir = os.path.join("/ssd_Samsung870_2T",'nnUNet/nnunet')
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

os.system('nnUNet_predict -i /ssd_Samsung870_2T/nnUNet/nnunet/nnUNet_trained_models/val/images/ -o /ssd_Samsung870_2T/nnUNet/nnunet/nnUNet_trained_models/output/images/ -t 777 -tr nnUNetTrainerV2_insaneDA -m 3d_fullres')