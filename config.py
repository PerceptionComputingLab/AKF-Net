class MicroadenomaConfig(object):
    '''dataset'''
    # dataset_path = "./dataset/sample_data"
    dataset_path = "/home/lyf/gitcode/PMiA-Seg/dataset/microadenoma_cor"
    frame_num = 7
    patch_size =  [5, 128, 128]
    crop_size =  [128, 128]
    crop_ratio = [0.4, 0.62, 0.4, 0.6]
    step_size = [4, 90, 90]
    n_class = 3
    train_ratio = 0.55
    roi_crop_pos=[127, 302]
    '''model'''
    resnet_out_channels=[32, 64, 128, 256]  
    dropout=0.5
    resnet_depth = 18
    '''training'''
    lr = 0.001
    lr_step_size = 150
    lr_ratio = 0.5
    n_epoch = 800 # 800
    num_batches_per_epoch = 50
    num_validation_batches_per_epoch = 50
    batch_size = 2
    save_step = 10
    val_step = 1
    '''testing'''

    checkpoint_path = [""] # /home/AEPformer/runs/best_checkpoint_0.pt

    def __init__(self):
        pass
    
    def get_str_config(self):
        '''
        get string of the configurations for displaying and storing
        :return: a string of the configuration
        '''
        str_config = "configurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                temp = "{:30} {}\n".format(a, getattr(self, a))
                str_config += temp
        return str_config