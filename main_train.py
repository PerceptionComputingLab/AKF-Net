import torch
import time
import os
import numpy as np
import torch.utils.data as dataloader
from config import MicroadenomaConfig
from utils.logger import Logger
from model.model import AEPformer
from tqdm import tqdm
from utils.data_utils import train_validate_split, pre_processingV2, NormalizationV2, get_metrics_dice, MicroadenomaDataset
from monai.losses import DiceFocalLoss, DiceCELoss
from utils.utils import DiceLogHDLoss, FocalLogHDLoss

import utils.augment as aug

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(model, dataloader):
    num = 0
    with torch.no_grad():
        dice_new_list = []
        for batch in tqdm(dataloader):
            image_val = NormalizationV2(batch['data'])
            label_val = batch['seg']
            image_val = image_val.to(device)
            '''==================predicted=================='''
            model.eval()
            output = model(image_val)
            _, predicton = torch.max(output.data, 1)
            whole_pred = predicton.data.cpu().numpy().astype(np.int16)
            whole_target = label_val[:, 0, ...].data.numpy().astype(np.int16)
            '''==================calculate metric=================='''
            dsc_itm = get_metrics_dice(whole_pred, whole_target)
            dice_new_list.append(dsc_itm)
            num += 1
        dice_array = np.array(dice_new_list)
        dice_mean = np.mean(dice_array, axis=0)
    return dice_mean


if __name__ == "__main__":
    config = MicroadenomaConfig()
    mylogger = Logger("runs", write=True, save_freq=4)
    mylogger.log(config.get_str_config())
    mylogger.save_pkl(config)

    train_data_path_list, val_data_path_list = train_validate_split(config.dataset_path, train_ratio=config.train_ratio, seed=1)
    sometimes = lambda augmentor: aug.Sometimes(0.3, augmentor) # Used to apply augmentor with specified probability
    augment_train = aug.Sequential([
        sometimes(aug.HorizontalFlip()),
        sometimes(aug.VerticalFlip()),
        sometimes(aug.RandomRotate(30))
    ])
    dataset_train = MicroadenomaDataset(data_path_list=train_data_path_list, 
                                        frame_num=config.frame_num, 
                                        crop_size=config.crop_size, 
                                        crop_ratio=config.crop_ratio,
                                        transfor=augment_train)
    dataset_val = MicroadenomaDataset(data_path_list=val_data_path_list, 
                                        frame_num=config.frame_num, 
                                        crop_size=config.crop_size,
                                        crop_ratio=config.crop_ratio)
    train_loader = dataloader.DataLoader(dataset_train, batch_size=config.batch_size)
    val_loader = dataloader.DataLoader(dataset_val, batch_size=1)

    model = AEPformer(frame_num=config.frame_num,
                    img_shape=config.patch_size,
                    output_channel=config.n_class,
                    resnet_depth=config.resnet_depth,
                    resnet_out_channels=config.resnet_out_channels,
                    dropout=config.dropout).to(device)

    criterion = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5, betas=(0.97, 0.999))
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_ratio)

    best_dice_val = 0
    best_val_num = -1
    mylogger.log('LR\t\t\tepoch\tLoss \tDSC_BG\t\tDSC_MP\t\tDSC_MA')
    for epoch in range(config.n_epoch):
        num_batches_per_epoch = 0
        model.train()
        loss_epoch = 0
        beta_epoch = 0
        epoch_start_time = time.time()
        '''===================== Training ====================='''
        for batch in tqdm(train_loader):
            image_data, label_data = pre_processingV2(batch)
            image_data = image_data.to(device)
            label_data = label_data.to(device)
            '''============== forward =============='''
            optimizer.zero_grad()
            pred = model(image_data)
            loss = criterion(pred, label_data)
            '''======== backward and optimize ======'''
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            num_batches_per_epoch += 1
        epoch_train_end = time.time()
        loss_epoch /= num_batches_per_epoch
        '''===================== Validation ====================='''
        if epoch % config.val_step == 0:
            metric_val = evaluation(model, val_loader)
            mylogger.log('%0.6f\t%6d\t\t%0.5f\t%0.5f\t\t%0.5f\t\t%0.5f' % (
                        optimizer.state_dict()["param_groups"][0]['lr'],
                        epoch,
                        loss_epoch,
                        metric_val[0], metric_val[1], metric_val[2]))
            structure = ["DSC_bg", "DSC_WP", "DSC_MA"]
            board_info= {structure[i]: metric_val[i] for i in range(len(metric_val))}
            mylogger.write_to_board(f"Validation/score", board_info, epoch)
            mylogger.write_to_board(f"Loss", {"loss": loss_epoch}, epoch)
        '''===================== save model ====================='''
        if epoch % config.save_step == 0:
            mylogger.save_model(model.state_dict(), f"checkpoint_{epoch}.pt", forced=True)
        if np.mean(metric_val[1:])>best_dice_val:
            best_checkpoint_path = os.path.join(mylogger.dir, f"best_checkpoint_{best_val_num}.pt")
            if os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            best_val_num = epoch
            mylogger.save_model(model.state_dict(), f"best_checkpoint_{best_val_num}.pt", forced=True)
            best_dice_val = np.mean(metric_val[1:])
        '''==== save the latest checkpoint ===='''
        mylogger.save_model(model.state_dict(), "latest_checkpoint.pt", forced=True)
        '''================== learning rate decay ================='''
        scheduler_lr.step()
        epoch_end_time = time.time()