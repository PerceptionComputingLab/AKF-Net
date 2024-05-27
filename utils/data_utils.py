from cProfile import label
import os
import copy
import torch
import numpy as np
import SimpleITK as sitk
import torch.utils.data as data
from utils.utils import dice, rAVD, jaccard, ASSD, hausdorff_distance, precision, recall
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform
from batchgenerators.transforms.abstract_transforms import Compose
import torch.utils.data as data



def get_train_transform():
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    # tr_transforms.append(
    #     SpatialTransform_2(patch_size=None, do_rotation=True,
    #         angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         border_mode_data='constant', border_cval_data=0,
    #         border_mode_seg='constant', border_cval_seg=0,
    #         order_seg=1, order_data=3)
    # )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.9, 1.1), per_channel=True, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


class MicroadenomaPatchData(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, n_class=4, frame_num=7, crop_pos=None):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle, infinite)
        self.patch_size = patch_size
        self.indices = list(range(len(data)))
        self.n_class = n_class
        self.frame_num = frame_num
        self.crop_pos = crop_pos

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        data = np.zeros((self.batch_size, self.frame_num, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        for i, itm_path in enumerate(patients_for_batch):
            image_data, label_data = self.load_volumes_label(itm_path)
            image_pad_data = pad_nd_image(image_data, [self.frame_num] + self.patch_size)
            label_pad_data = pad_nd_image(label_data, self.patch_size)

            label_pad_data = label_pad_data[np.newaxis, :]
            # label_pad_data = np.tile(label_pad_data, reps=[self.frame_num, 1, 1, 1])

            image_crop_data, label_crop_data = crop(image_pad_data[None], label_pad_data[None], self.patch_size, crop_type="random")

            data[i] = image_crop_data[0]
            seg[i] = label_crop_data[0]

        return {"data": data, "seg": seg}

    def load_volumes_label(self, src_path):
        image_path = src_path["image"]
        label_path = src_path["label"]
        image_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (frm, slice, H, W)
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))  # (slice, H, W)
        # crop ROI
        image_arr = image_arr[:, :, self.crop_pos[0]:self.crop_pos[1] + 1, self.crop_pos[0]:self.crop_pos[1] + 1]
        label_arr = label_arr[:, self.crop_pos[0]:self.crop_pos[1] + 1, self.crop_pos[0]:self.crop_pos[1] + 1]
        # uniform frame num to self.frame_num
        frm_gap = self.frame_num - image_arr.shape[0]
        flag = bool(np.random.randint(0, 2))
        while frm_gap:
            if flag:
                image_arr = np.insert(image_arr, 0, image_arr[0], axis=0) if frm_gap > 0 else np.delete(image_arr, 0, axis=0)
            else:
                image_arr = np.insert(image_arr, image_arr.shape[0], image_arr[-1], axis=0) if frm_gap > 0 else np.delete(image_arr, -1, axis=0)
            flag = not flag
            frm_gap -= 1 if frm_gap > 0 else -1
        return image_arr, label_arr


class MicroadenomaVolumeData(data.Dataset):
    def __init__(self, data_path, mode="train", frame_num=7, crop_pos=None):
        train_list, validate_list = train_validate_split(data_path, train_ratio=0.8, seed=1)
        if mode == 'train':
            self.data_list = train_list
        else:
            # self.data_list = validate_list
            self.data_list = train_list
        self.frame_num = frame_num
        self.crop_pos = crop_pos

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        single_dir_path = self.data_list[index]
        image_data, label_data = self.load_volumes_label(single_dir_path)
        label_data = label_data[np.newaxis, :]
        single_data = {"data": image_data, "seg": label_data}
        return pre_processingV1(single_data)
    
    def load_volumes_label(self, src_path):
        image_path = src_path["image"]
        label_path = src_path["label"]
        image_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (frm, slice, H, W)
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))  # (slice, H, W)
        # crop ROI
        image_arr = image_arr[:, :, self.crop_pos[0]:self.crop_pos[1] + 1, self.crop_pos[0]:self.crop_pos[1] + 1]
        label_arr = label_arr[:, self.crop_pos[0]:self.crop_pos[1] + 1, self.crop_pos[0]:self.crop_pos[1] + 1]
        # uniform frame num to self.frame_num
        frm_gap = self.frame_num - image_arr.shape[0]
        flag = bool(np.random.randint(0, 2))
        while frm_gap:
            if flag:
                image_arr = np.insert(image_arr, 0, image_arr[0], axis=0) if frm_gap > 0 else np.delete(image_arr, 0, axis=0)
            else:
                image_arr = np.insert(image_arr, image_arr.shape[0], image_arr[-1], axis=0) if frm_gap > 0 else np.delete(image_arr, -1, axis=0)
            flag = not flag
            frm_gap -= 1 if frm_gap > 0 else -1
        return image_arr, label_arr


class MicroadenomaDataset(data.Dataset):
    def __init__(self, data_path_list, frame_num=7, crop_size=[128, 128], crop_ratio=[0.4, 0.62, 0.4, 0.6], transfor=None):
        self.data_path_list = data_path_list
        self.frame_num = frame_num
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.transfor = transfor
    
    def __getitem__(self, index):
        itm_path = self.data_path_list[index]
        image_data, label_data = self.load_volumes_label(itm_path)
        label_data = label_data[np.newaxis, :]
        out = {"data": image_data.astype(np.float64), "seg": label_data.astype(np.int16)}
        if self.transfor is not None:
            out = self.transfor(out)
        return out

    def __len__(self):
        return len(self.data_path_list)

    def load_volumes_label(self, src_path):
        image_path = src_path["image"]
        label_path = src_path["label"]
        image_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (frm, slice, H, W)
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))  # (slice, H, W)
        # crop ROI
        slice_num = label_arr.shape[0]
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
        x_start = int(label_arr.shape[1] * self.crop_ratio[0]) - 1
        x_end = int(label_arr.shape[1] * self.crop_ratio[1]) + 1
        y_start = int(label_arr.shape[2] * self.crop_ratio[2]) - 1
        y_end = int(label_arr.shape[2] * self.crop_ratio[3]) + 1
        assert self.crop_size[0] >= (x_end - x_start + 1)
        assert self.crop_size[1] >= (y_end - y_start + 1)
        x_prior = (self.crop_size[0] - (x_end - x_start + 1)) // 2
        x_post = self.crop_size[0] - (x_end - x_start + 1) - x_prior
        x_start -= x_prior
        x_end += x_post
        y_prior = (self.crop_size[1] - (y_end - y_start + 1)) // 2
        y_post = self.crop_size[1] - (y_end - y_start + 1) - y_prior
        y_start -= y_prior
        y_end += y_post
        assert self.crop_size[0] == (x_end - x_start + 1)
        assert self.crop_size[1] == (y_end - y_start + 1)
        image_arr = image_arr[:, s_start:s_end + 1, x_start:x_end + 1, y_start:y_end + 1]
        label_arr = label_arr[s_start:s_end + 1, x_start:x_end + 1, y_start:y_end + 1]
        # uniform frame num to self.frame_num
        frm_gap = self.frame_num - image_arr.shape[0]
        flag = bool(np.random.randint(0, 2))
        while frm_gap:
            if flag:
                image_arr = np.insert(image_arr, 0, image_arr[0], axis=0) if frm_gap > 0 else np.delete(image_arr, 0, axis=0)
            else:
                image_arr = np.insert(image_arr, image_arr.shape[0], image_arr[-1], axis=0) if frm_gap > 0 else np.delete(image_arr, -1, axis=0)
            flag = not flag
            frm_gap -= 1 if frm_gap > 0 else -1
        return image_arr, label_arr


def train_validate_split(dir_path, train_ratio, seed=4):
    train_txt = os.path.join(dir_path, 'train.txt')
    test_txt = os.path.join(dir_path, 'test.txt')
    image_dir_path = os.path.join(dir_path, 'image')
    label_dir_path = os.path.join(dir_path, 'label')
    train_patient_num = []
    test_patient_num = []
    if os.path.exists(train_txt) and os.path.exists(test_txt):
        file = open(train_txt, 'r')
        content = file.read()
        content = content.split('\n')
        file.close()
        train_patient_num = [x for x in content if x != '']

        file = open(test_txt, 'r')
        content = file.read()
        content = content.split('\n')
        file.close()
        test_patient_num = [x for x in content if x != '']
    else:
        label_file_list = os.listdir(label_dir_path)
        patient_list = [label_file.split('.')[0] for label_file in label_file_list]
        ignore_patient = ['00064', '00145']
        for itm in ignore_patient:
            if itm in patient_list:
                patient_list.remove(itm)

        np.random.seed(seed)
        np.random.shuffle(patient_list)
        split_idx = int(len(patient_list) * train_ratio)
        for idx, patient_num in enumerate(patient_list):
            train_patient_num.append(patient_num) if idx < split_idx else test_patient_num.append(patient_num)
        
        file = open(train_txt, "w")
        for item in train_patient_num:
            file.write(item + "\n")
        file.close()

        file = open(test_txt, 'w')
        for item in test_patient_num:
            file.write(item + "\n")
        file.close()

    train_list = []
    for idx, patient_num in enumerate(train_patient_num):
        path_dict = {"image": os.path.join(image_dir_path, patient_num, "dyn_cor.nii.gz"),
                        "label": os.path.join(label_dir_path, patient_num + ".nii.gz")}
        train_list.append(path_dict)
   
    test_list = []
    for idx, patient_num in enumerate(test_patient_num):
        path_dict = {"image": os.path.join(image_dir_path, patient_num, "dyn_cor.nii.gz"),
                     "label": os.path.join(label_dir_path, patient_num + ".nii.gz")}
        test_list.append(path_dict)
    
    return train_list, test_list


def NormalizationV1(volume):
    batch, c, _,_,_, = volume.shape
    bg_mask = volume == 0
    mean_arr = np.zeros(c, dtype="float32")
    std_arr = np.zeros(c, dtype="float32")
    norm_volume = copy.deepcopy(volume.transpose(0, 2, 3, 4, 1))
    for j in range(batch):
        for i in range(c):
            data = volume[j, i, ...]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std
        norm_volume[j] = (volume[j].transpose(1, 2, 3, 0) - mean_arr) / std_arr

    norm_volume = norm_volume.transpose(0,4,1,2,3)
    norm_volume[bg_mask] = 0

    return norm_volume


def NormalizationV2(volume):
    batch, c, _,_,_, = volume.shape
    bg_mask = volume == 0
    mean_arr = torch.zeros(c, dtype=torch.float32)
    std_arr = torch.zeros(c, dtype=torch.float32)
    norm_volume = copy.deepcopy(volume.permute(0, 2, 3, 4, 1))
    for j in range(batch):
        for i in range(c):
            data = volume[j, i, ...]
            selected_data = data[data > 0]
            mean = torch.mean(selected_data)
            std = torch.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std
        norm_volume[j] = (volume[j].permute(1, 2, 3, 0) - mean_arr) / std_arr

    norm_volume = norm_volume.permute(0,4,1,2,3)
    norm_volume[bg_mask] = 0

    return norm_volume.float()


def pre_processingV1(data_dict):
    '''
    transfer numpy data to that of tensor
    :param data:
    :return:
    '''
    img_array = data_dict["data"]
    label_array = data_dict["seg"]
    img_array = img_array.astype(np.float64)
    label_array = label_array.astype(np.int16)
    img_normed = NormalizationV1(img_array)

    return (torch.from_numpy(img_normed).float(),
            torch.from_numpy(label_array).long())


def pre_processingV2(data_dict):
    '''
    transfer numpy data to that of tensor
    :param data:
    :return:
    '''
    img_array = data_dict["data"].to(dtype=torch.float64)
    label_array = data_dict["seg"].to(dtype=torch.int16)
    img_normed = NormalizationV2(img_array)
    return (img_normed.float(), label_array.long())


def get_metrics_dice(whole_pred, whole_target):
    """
    :param whole_pred: ndarry, batch_size*W*H*D
    :param whole_target: ndarry, batch_size*W*H*D
    """
    # whole pituitary ans background
    WP_pred = whole_pred > 0
    WP_target = whole_target > 0
    dsc_bg = dice(WP_pred, WP_target, 0)
    dsc_wp = dice(WP_pred, WP_target, 1)
    # microadenoma
    whole_target[whole_target==1] = 0
    whole_pred[whole_pred==1] = 0
    MA_target = whole_target > 0
    MA_predict = whole_pred > 0
    dsc_ma = dice(MA_predict, MA_target, 1)
    return [dsc_bg, dsc_wp, dsc_ma]

def get_metrics(whole_pred, whole_target):
    """
    :param whole_pred: ndarry, batch_size*W*H*D
    :param whole_target: ndarry, batch_size*W*H*D
    """
    # whole pituitary ans background
    WP_pred = whole_pred > 0
    WP_target = whole_target > 0
    dsc_bg = dice(WP_pred, WP_target, 0)
    dsc_wp = dice(WP_pred, WP_target, 1)
    rvd_wp = rAVD(WP_pred, WP_target)
    ji_wp = jaccard(WP_pred, WP_target, 1)
    asd_wp = ASSD(WP_pred, WP_target, (1.0, 1.0, 1.0))
    hd95_wp = hausdorff_distance(WP_pred, WP_target, (1.0, 1.0, 1.0))
    pre_wp = precision(WP_pred, WP_target)
    rec_wp = recall(WP_pred, WP_target)
    # microadenoma
    whole_target[whole_target==1] = 0
    whole_pred[whole_pred==1] = 0
    MA_target = whole_target > 0
    MA_predict = whole_pred > 0
    dsc_ma = dice(MA_predict, MA_target, 1)
    rvd_ma = rAVD(MA_predict, MA_target)
    ji_ma = jaccard(MA_predict, MA_target, 1)
    asd_ma = ASSD(MA_predict, MA_target, (1.0, 1.0, 1.0))
    hd95_ma = hausdorff_distance(MA_predict, MA_target, (1.0, 1.0, 1.0))
    pre_ma = precision(MA_predict, MA_target)
    rec_ma = recall(MA_predict, MA_target)

    return {"dsc":[dsc_bg, dsc_wp, dsc_ma], 
            "rvd":[rvd_wp, rvd_ma], 
            "ji":[ji_wp, ji_ma], 
            "asd":[asd_wp, asd_ma],
            "hd95":[hd95_wp, hd95_ma],
            "pre":[pre_wp, pre_ma],
            "rec":[rec_wp, rec_ma]}

"""
if __name__ == '__main__':
    data_path = r"G:\microadenoma_dataset\microadenoma_dataset_abnormal\microadenoma_cor"
    n_epochs = 200
    num_batches_per_epoch = 250
    train_data_path_list, test_data_path_list = train_validate_split(data_path, train_ratio=0.8, seed=4)

    dataloader_train = MicroadenomaData(train_data_path_list, batch_size=3, patch_size=[5, 128, 128], num_threads_in_multithreaded=4, n_class=3, frame_num=7, crop_pos=[127, 302])
    tr_transforms = get_train_transform()

    train_loader = SingleThreadedAugmenter(dataloader_train, transform=tr_transforms)
    for epoch in range(n_epochs):
        for _ in range(num_batches_per_epoch):
            batch = next(train_loader)
            image_data = batch["data"]
            label_data = batch["seg"]
"""