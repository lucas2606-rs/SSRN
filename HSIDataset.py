import numpy as np
import scipy.io as sio
import random
from sklearn.model_selection import train_test_split

class HSIDataset:
    def __init__(self, dataset_name, patchsz):
        info = DataInfo.info[dataset_name]
        data = sio.loadmat(info['data_path'])[info['data_key']]
        data = data.astype(np.float32)

        label = sio.loadmat(info['label_path'])[info['label_key']]
        label = label.astype(np.int32)

        self.train_val_test = info['train_val_test']

        self.y = label
        self.x = self.standardize(data)
        self.x = self.addMirror(self.x, patchsz)
        self.x_patch, self.y_patch = self.createPatches(self.x, self.y, patchsz)
        self.x_train, self.y_train, self.x_val, \
        self.y_val, self.x_test, self.y_test = self.splitTrainValTest(self.x_patch, self.y_patch, self.train_val_test)

    def standardize(self, x):
        mean = np.mean(x)
        std = np.std(x)
        new_x = (x - mean) / std
        return new_x

    def padWithZeros(self, x, patchsz):
        dx = patchsz // 2
        h, w, c = x.shape
        new_x = np.zeros((h + 2 * dx, w + 2 * dx, c), dtype=np.float32)
        new_x[dx:-dx, dx:-dx] = x
        return new_x

    def addMirror(self, x, patchsz):
        dx = patchsz // 2
        x = self.padWithZeros(x, patchsz)
        for i in range(dx):
            x[i, :, :] = x[2 * dx - i, :, :]
            x[:, i, :] = x[:, 2 * dx - i, :]
            x[-i - 1, :, :] = x[-(2 * dx - i) - 1, :, :]
            x[:, -i - 1, :] = x[:, -(2 * dx - i) - 1, :]
        return x

    def createPatches(self, data, label, patchsz):
        dx = patchsz // 2
        nonzero = np.nonzero(label)
        sample_ind = list(zip(*nonzero))
        num_sample = len(sample_ind)
        patched_data = np.zeros((num_sample, patchsz, patchsz, data.shape[2], 1), dtype=np.float32)
        patched_label = np.zeros(num_sample, dtype=np.int32)
        for i, (x, y) in enumerate(sample_ind):
            patched_data[i] = np.expand_dims(data[x:x + 2 * dx + 1, y:y + 2 * dx + 1], axis=-1)
            patched_label[i] = label[x, y] - 1
        return patched_data, patched_label

    def splitTrainValTest(self, x_patch, y_patch, train_val_test, random_state=11413):
        train_ratio = train_val_test[0]
        val_ratio = train_val_test[1]
        test_ratio = train_val_test[2]
        x_train, x_val_test, y_train, y_val_test = train_test_split(x_patch, y_patch,
                                                                    train_size=train_ratio,
                                                                    random_state=random_state,
                                                                    stratify=y_patch)
        test_size = test_ratio / (val_ratio + test_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y_val_test)
        return x_train, y_train, x_val, y_val, x_test, y_test


class DataInfo:
    info = {
        'IN':{
            'data_path': './data/Indian/Indian_pines_corrected.mat',
            'label_path': './data/Indian/Indian_pines_gt.mat',
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt',
            'target_names': [
                'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                'Stone-Steel-Towers'],
            'train_val_test':[0.2, 0.1, 0.7]},
        'KSC':{
            'data_path': './data/KSC/KSC.mat',
            'label_path': './data/KSC/KSC_gt.mat',
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'target_names': [
                'Scrub', 'Willow swamp', 'Cabbage palm hammock',
                'Cabbage palm/oak hammock', 'Slash pine', 'Oak/broadleaf hammock',
                'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
                'Cattail marsh', ' Salt marsh', ' Mud flats', 'Water'],
            'train_val_test':[0.2, 0.1, 0.7]},
        'UP':{
            'data_path': './data/PaviaU/PaviaU.mat',
            'label_path': './data/Pavia/PaviaU_gt.mat',
            'data_key': 'paviaU',
            'label_key': 'paviaU_gt',
            'target_names':[
                'Asphalt', 'Meadows', 'Gravel', 'Trees',
                'Metal sheets', 'Bare soil', 'Bitumen',
                'Bricks', 'Shadows'],
            'train_val_test':[0.1, 0.1, 0.8]}}


