from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch
import pickle as pkl
from torch.utils.data import Dataset

class dalDataset_target(Dataset):
    """
    ASOCT-2 class
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(dalDataset_target, self).__init__()
        self.root = root
        self.foreign = '/home/qiuzhen/data/foreign_val_data'
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list_ca2 = train_dict['train_list']
        val_list_ca2 = train_dict['val_list']
        with open('./data/foreign_all_data_re1_train_val.pkl', 'rb') as f1:
            train_dict1 = pkl.load(f1)
        train_list = train_dict1['train_list']
        val_list = train_dict1['val_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                file_list = os.listdir(os.path.join(self.foreign, train_list[i][0]))
                file_list.sort()
                self.train_data.append(os.path.join(self.foreign, train_list[i][0], file_list[0]))
                self.train_labels.append((train_list[i][1], train_list[i][2]))
                self.train_data.append(os.path.join(self.foreign, train_list[i][0], file_list[64]))
                self.train_labels.append((train_list[i][1], train_list[i][2]))
            for i in range(len(train_list_ca2)):
                for j in range(2):
                    self.train_data.append(os.path.join(self.root, train_list_ca2[i][0] + '_' + '%.03d' % (9*j) + '.jpg'))
                    self.train_labels.append((train_list_ca2[i][1], train_list_ca2[i][1]))
        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                 file_list = os.listdir(os.path.join(self.foreign, val_list[i][0]))
                 file_list.sort()
                 self.test_data.append(os.path.join(self.foreign, val_list[i][0], file_list[0]))
                 self.test_labels.append((val_list[i][1], val_list[i][2]))
                 self.test_data.append(os.path.join(self.foreign, val_list[i][0], file_list[64]))
                 self.test_labels.append((val_list[i][1], val_list[i][2]))
            for i in range(len(val_list_ca2)):
                for j in range(2):
                    self.test_data.append(os.path.join(self.root, val_list_ca2[i][0] + '_' + '%.03d' % (9*j) + '.jpg'))
                    self.test_labels.append((val_list_ca2[i][1], val_list_ca2[i][1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('L')
        img = np.array(img)
        img_L = img[:, :int(img.shape[1] / 2)]
        img_R = img[:, int(img.shape[1] / 2):]
        img_L = Image.fromarray(img_L).convert('RGB')
        img_R = Image.fromarray(img_R).convert('RGB')

        if self.transform is not None:
            img_L = self.transform(img_L)
            img_R = self.transform(img_R)

        img = torch.stack([img_L, img_R], dim=0)
        target = torch.LongTensor([target[0], target[1]])
        ind = torch.LongTensor([2*index, 2*index+1])
        return img, target, ind

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class officeDataset(Dataset):
    """
    ASOCT-2 class
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(officeDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list'] + train_dict['test_list']
        val_list = train_dict['train_list'] + train_dict['test_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                 self.test_data.append(os.path.join(self.root, val_list[i][0]))
                 self.test_labels.append(val_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class visDataset(Dataset):
    """
    ASOCT-2 class
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(visDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']
        val_list = train_dict['test_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                 self.test_data.append(os.path.join(self.root, val_list[i][0]))
                 self.test_labels.append(val_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class visDataset_target(Dataset):
    def __init__(self, root, label_file, train=True, transform=None):
        super(visDataset_target, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(train_list)):
                 self.test_data.append(os.path.join(self.root, train_list[i][0]))
                 self.test_labels.append(train_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class visDataset_target_select(Dataset):
    def __init__(self, root, label_file, pseudo_label, select_index, transform=None):
        super(visDataset_target_select, self).__init__()
        self.root = root
        self.transform = transform
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']

        self.train_data = []
        self.train_labels = []
        for i in range(len(train_list)):
            if i in select_index:
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(pseudo_label[i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name, target = self.train_data[index], self.train_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
