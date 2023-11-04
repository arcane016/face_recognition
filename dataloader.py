import os
from os.path import join as pjoin

import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image
from tqdm import tqdm
import json


def generate_randinit(start, stop, ignore=[]):
    flag = True
    while flag:
        num = random.randint(start, stop-1)
        if num not in ignore:
            flag = False
            return num

class FaceRecogDataLoader(Dataset):
    def __init__(self, root_path, split, transforms=None, random=True, data_config_path=None):
        self.transforms = transforms
        self.prefix_path = root_path
        self.split = split
        self.random = random
        self.pos_slots = 3 if self.split == "train" else 2
        self.neg_slots = 9
        self.dataset_path = pjoin(self.prefix_path, self.split)
        assert os.path.exists(self.dataset_path)
        self.data_config_path = data_config_path
        assert self.data_config_path==None or os.path.exists(self.data_config_path)
        self.dataset_size = self.getlen()
        self.dataset_list = self.getlist()
        self.dataset_size = len(self.dataset_list["label"])

    def getlen(self):
        x = len(os.listdir(self.dataset_path))
        p = self.pos_slots
        n = self.neg_slots
        return x*p*n + int(p*(p-1)/2)*x
    
    def getlist(self):
        if not self.random and self.data_config_path != None:
            with open(self.data_config_path, "r") as file:
                data_list = json.load(file)
            assert isinstance(data_list, dict)
        else:
            data_list = {
                "folder": [],
                "label": [],
                "left_idx": [],
                "right_idx": []
            }

            p = self.pos_slots
            n = self.neg_slots

            for folder in os.listdir(self.dataset_path):

                l_folder = int(p*(p-1)/2)*[folder] + p*n*[folder]
                l_label = int(p*(p-1)/2)*[0] + p*n*[1]
                
                l_left = []
                l_right = []
                for i in range(p):
                    l_left += (p-1-i)*[i]
                    l_right += [n for n in range(i+1, p)]
                for i in range(p):
                    l_left += n*[i]
                    l_right += n*[-1]
                
                assert len(l_left) == len(l_folder)


                data_list["folder"] += l_folder
                data_list["label"] += l_label
                data_list["left_idx"] += l_left
                data_list["right_idx"] += l_right
            
        assert len(data_list["label"]) == self.dataset_size, f" len(data_list) = {len(data_list['label'])}, dataset_size = {self.dataset_size}"
        return data_list


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, idx):
        folder = self.dataset_list["folder"][idx]
        label = self.dataset_list["label"][idx]
        left_idx = self.dataset_list["left_idx"][idx]

        if label == 0:
            folder2 = folder
            right_idx = self.dataset_list["right_idx"][idx]
            assert right_idx != -1 and left_idx != right_idx
            assert isinstance(right_idx, int)
        elif label == 1:
            if self.random or ((not self.random) and self.dataset_list["right_idx"][idx] == -1):
                folder_list = os.listdir(self.dataset_path) 
                folder2 = folder_list[generate_randinit(0, len(folder_list), [folder_list.index(folder)])]
                # folder2 = self.dataset_list["folder"][generate_randinit(0, self.dataset_size, [idx])]
                nfiles2 = len(os.listdir(pjoin(self.dataset_path, folder2)))
                right_idx = generate_randinit(0, nfiles2, [])
                if not self.random:
                    self.dataset_list["right_idx"][idx] = (folder2, right_idx)
                    if self.data_config_path == None and idx+5 >= self.dataset_size:
                        with open(pjoin("./data_configs", f"{self.split}_config2.json"), 'w') as outfile:
                            json.dump(self.dataset_list, outfile)
            else:
                (folder2, right_idx) = self.dataset_list["right_idx"][idx]
            
            assert folder != folder2
        
        left_image = read_image(pjoin(self.dataset_path, folder, str(left_idx)+".jpg"))
        right_image = read_image(pjoin(self.dataset_path, folder2, str(right_idx)+".jpg"))

        if self.transforms != None:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)
        
        return {"data":[left_image.float(), right_image.float()], "label": label}


if __name__ == "__main__":

    ROOT_PATH = "/home2/kushagra0301/projects_self/face_recognition/datasets/lfw_dataset"

    train_data = FaceRecogDataLoader(ROOT_PATH, "train", random=False, data_config_path=None)#"./data_configs/train_config.json")
    print("train:",train_data.dataset_size)

    val_data = FaceRecogDataLoader(ROOT_PATH, "val", random=False, data_config_path=None)#"./data_configs/val_config.json")
    print("val:",val_data.dataset_size)


    for sample in tqdm(train_data):
        assert sample["label"] == 1 or sample["label"] == 0
    
    for sample in tqdm(val_data):
        assert sample["label"] == 1 or sample["label"] == 0
    
    # print()

    train_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=3)
    
    # print("Done")

    for batch_sample in tqdm(train_dataloader):
        # print(batch_sample["label"].shape)
        # print(batch_sample["data"][0].shape, batch_sample["data"][1].shape)
        pass