
import os
import torch
import numpy as np
import SimpleITK as sitk
from config import MicroadenomaConfig
from utils.logger import Logger
from model.model import PMiASeg
from utils.data_utils import get_metrics, train_validate_split, NormalizationV1
from matplotlib import pyplot as plt
import pickle as pkl

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_data_mean(dice_per_class_list):
    dice_per_class=[]
    for i in range(len(dice_per_class_list[0])):
        dice_per_class.append(np.mean(dice_per_class_list[:, i]))

    dice_mean = np.mean(dice_per_class[1:])
    return dice_per_class, dice_mean

def soft_vote(predicted_probas):
    sv_predicted_prob = torch.mean(predicted_probas, dim=0)
    sv_predictions = torch.argmax(sv_predicted_prob, dim=0)
    return sv_predictions

def hard_vote(predicted_probas):
    hv_predictions = torch.argmax(predicted_probas, dim=1)
    hv_predictions = torch.mode(hv_predictions, dim=0).values
    return hv_predictions

if __name__ == "__main__":
    mylogger = Logger("runs", write=True, save_freq=4)
    os.mkdir(mylogger.plt_dir)
    config = MicroadenomaConfig()
    mylogger.log(config.get_str_config())
    '''======================== load model ========================'''
    model_list = []
    for ckpt_path in config.checkpoint_path:
        model_hyper_param = mylogger.load_from_pkl(os.path.dirname(ckpt_path))
        model = PMiASeg(frame_num=model_hyper_param["frame_num"],
                        img_shape=model_hyper_param["patch_size"],
                        output_channel=model_hyper_param["n_class"],
                        resnet_depth=model_hyper_param["resnet_depth"],
                        resnet_out_channels=model_hyper_param["resnet_out_channels"],
                        dropout=model_hyper_param["dropout"]).to(device)
        mylogger.log("start load checkpoint: {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        model_list.append(model)
    mylogger.log("load checkpoints succeed!")
    '''===================== load testing data ====================='''
    
    train_data_path_list, val_data_path_list = train_validate_split(config.dataset_path, train_ratio=config.train_ratio, seed=1)
    test_data_list = val_data_path_list

    dice_per_class_list = []
    rvd_per_class_list = []
    ji_per_class_list = []
    asd_per_class_list = []
    hd95_per_class_list = []
    pre_per_class_list = []
    rec_per_class_list = []
    for file_itm, file_path in enumerate(test_data_list):
        patient_num = file_path["image"].split('/')[-2]
        image_path = file_path["image"]
        label_path = file_path["label"]
        orig_image_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (frm, slice, H, W)
        orig_label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))  # (slice, H, W)
        '''========================= crop image =========================='''
        slice_num = orig_image_arr.shape[1]
        if slice_num == 6 or slice_num == 5:
            s_start, s_end = 0, 4
        elif slice_num == 7:
            s_start, s_end = 1, 5
        elif slice_num == 8:
            s_start, s_end = 1, 5
        elif slice_num ==9:
            s_start, s_end = 2, 6
        else:
            s_start, s_end = None, None
        assert s_start is not None 
        assert s_end is not None
        x_start = int(orig_image_arr.shape[2] * config.crop_ratio[0]) - 1
        x_end = int(orig_image_arr.shape[2] * config.crop_ratio[1]) + 1
        y_start = int(orig_image_arr.shape[3] * config.crop_ratio[2]) - 1
        y_end = int(orig_image_arr.shape[3] * config.crop_ratio[3]) + 1
        assert config.crop_size[0] >= (x_end - x_start + 1)
        assert config.crop_size[1] >= (y_end - y_start + 1)
        x_prior = (config.crop_size[0] - (x_end - x_start + 1)) // 2
        x_post = config.crop_size[0] - (x_end - x_start + 1) - x_prior
        x_start -= x_prior
        x_end += x_post
        y_prior = (config.crop_size[1] - (y_end - y_start + 1)) // 2
        y_post = config.crop_size[1] - (y_end - y_start + 1) - y_prior
        y_start -= y_prior
        y_end += y_post
        assert config.crop_size[0] == (x_end - x_start + 1)
        assert config.crop_size[1] == (y_end - y_start + 1)
        crop_image_arr = orig_image_arr[:, s_start:s_end + 1, x_start:x_end + 1, y_start:y_end + 1]
        '''===================== uniform frame num to frame_num ====================='''
        frm_gap = config.frame_num - crop_image_arr.shape[0]
        flag = bool(np.random.randint(0, 2))
        while frm_gap:
            if flag:
                crop_image_arr = np.insert(crop_image_arr, 0, crop_image_arr[0], axis=0) if frm_gap > 0 else np.delete(crop_image_arr, 0, axis=0)
            else:
                crop_image_arr = np.insert(crop_image_arr, crop_image_arr.shape[0], crop_image_arr[-1], axis=0) if frm_gap > 0 else np.delete(crop_image_arr, -1, axis=0)
            flag = not flag
            frm_gap -= 1 if frm_gap > 0 else -1
        '''============================== normalization =============================='''
        img_arr = crop_image_arr[np.newaxis, :].astype(np.float64)
        img_arr = NormalizationV1(img_arr)
        img_arr = torch.from_numpy(img_arr).float().to(device)
        label_arr = orig_label_arr.astype(np.int16)
        '''============================== model predict =============================='''
        output_prob_list = []
        with torch.no_grad():
            for model in model_list:
                model.eval()
                output = model(img_arr)
                output = torch.softmax(output.squeeze(dim=0).data.cpu(), dim=0)
                output_prob_list.append(output)
        output_prob_list = torch.stack(output_prob_list)
        '''================================== vote =================================='''
        # output_vot = hard_vote(output_prob_list)
        output_vot = soft_vote(output_prob_list)

        predict = np.zeros(orig_image_arr.shape[1:])
        predict[s_start:s_end + 1, x_start:x_end + 1, y_start:y_end + 1] = output_vot.numpy()
        predict = np.array(predict, dtype=np.int16)     # (slice, H, W)
        target = np.copy(label_arr)                     # (slice, H, W)
        
        "================================== save =================================="
        with open(os.path.join(mylogger.plt_dir, "{}.pkl".format(patient_num )), 'wb') as f:
            pkl.dump(predict, f)
            
        # # 展示结果
        # for idx in range(img_arr.shape[2]):
        #     plt.subplot(1, 3, 1)
        #     plt.title("image")
        #     plt.imshow(img_arr[0, -1, idx, :, :].data.cpu())
        #     plt.xticks([])
        #     plt.yticks([])

        #     plt.subplot(1, 3, 2)
        #     plt.title("label")
        #     plt.imshow(target[idx, x_start:x_end + 1, y_start:y_end + 1], vmin=0, vmax=2)
        #     plt.xticks([])
        #     plt.yticks([])

        #     plt.subplot(1, 3, 3)
        #     plt.title("pred")
        #     plt.imshow(predict[idx, x_start:x_end + 1, y_start:y_end + 1], vmin=0, vmax=2)
        #     plt.xticks([])
        #     plt.yticks([])

        #     plt.suptitle(f"{patient_num} slice {str(idx)}")
        #     plt.savefig(os.path.join(mylogger.plt_dir, "{}.png".format(patient_num + "_slice_" + str(idx))))

        '''============================== calculate metrics =============================='''
        assert predict.dtype == target.dtype
        metric_dict = get_metrics(predict, target)

        dsc_test = metric_dict["dsc"][1:]
        dice_per_class_list.append(np.array(dsc_test))

        rvd_test = metric_dict["rvd"]
        rvd_per_class_list.append(np.array(rvd_test))

        ji_test = metric_dict["ji"]
        ji_per_class_list.append(np.array(ji_test))

        asd_test = metric_dict["asd"]
        asd_per_class_list.append(np.array(asd_test))

        hd95_test = metric_dict["hd95"]
        hd95_per_class_list.append(np.array(hd95_test))

        pre_test = metric_dict["pre"]
        pre_per_class_list.append(np.array(pre_test))

        rec_test = metric_dict["rec"]
        rec_per_class_list.append(np.array(rec_test))

        mylogger.log(f"{file_itm}th/{len(test_data_list)} {patient_num} volume data:" + 
                        " DSC: {dsc1:.5f}, {dsc2:.5f} || RVD: {rvd1:.5f}, {rvd2:.5f} || Jaccard: {ji1:.5f}, {ji2:.5f} || ASD: {asd1:.5f}, {asd2:.5f} || HD95: {hd1:.5f}, {hd2:.5f} || PRE: {pre1:.5f}, {pre2:.5f} || REC: {rec1:.5f}, {rec2:.5f}"
                        .format(dsc1=dsc_test[0], dsc2=dsc_test[1], 
                                rvd1=rvd_test[0], rvd2=rvd_test[1],
                                ji1=ji_test[0], ji2=ji_test[1],
                                asd1=asd_test[0], asd2=asd_test[1],
                                hd1=hd95_test[0], hd2=hd95_test[1],
                                pre1=pre_test[0], pre2=pre_test[1],
                                rec1=rec_test[0], rec2=rec_test[1]))

    dice_per_class_list = np.array(dice_per_class_list)
    dice_per_class, dice_mean = get_all_data_mean(dice_per_class_list)

    rvd_per_class_list = np.array(rvd_per_class_list)
    rvd_per_class, rvd_mean = get_all_data_mean(rvd_per_class_list)

    ji_per_class_list = np.array(ji_per_class_list)
    ji_per_class, ji_mean = get_all_data_mean(ji_per_class_list)

    asd_per_class_list = np.array(asd_per_class_list)
    asd_per_calss, asd_mean = get_all_data_mean(asd_per_class_list)

    hd95_per_class_list = np.array(hd95_per_class_list)
    hd95_per_class, hd95_mean = get_all_data_mean(hd95_per_class_list)

    pre_per_class_list = np.array(pre_per_class_list)
    pre_pre_class, pre_mean = get_all_data_mean(pre_per_class_list)

    rec_per_class_list = np.array(rec_per_class_list)
    rec_pre_class, rec_mean = get_all_data_mean(rec_per_class_list)

    mylogger.log("All test data: \tDSC: {dsc1:.5f}, {dsc2:.5f} || RVD: {rvd1:.5f}, {rvd2:.5f} || Jaccard: {ji1:.5f}, {ji2:.5f} || ASD: {asd1:.5f}, {asd2:.5f} || HD95: {hd1:.5f}, {hd2:.5f} || PRE: {pre1:.5f}, {pre2:.5f} || REC: {rec1:.5f}, {rec2:.5f}"
                    .format(dsc1=dice_per_class[0], dsc2=dice_per_class[1],
                            rvd1=rvd_per_class[0], rvd2=rvd_per_class[1],
                            ji1=ji_per_class[0], ji2=ji_per_class[1],
                            asd1=asd_per_calss[0], asd2=asd_per_calss[1],
                            hd1=hd95_per_class[0], hd2=hd95_per_class[1],
                            pre1=pre_pre_class[0], pre2=pre_pre_class[1],
                            rec1=rec_pre_class[0], rec2=rec_pre_class[1]))
