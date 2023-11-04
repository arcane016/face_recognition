# model = torchvision.models.vgg16()
# print("VGG16:\n",model)

# print()

# model = torchvision.models.resnet50()
# print("RESNET50:\n",model, end="\n\n")

# print()

# model = torchvision.models.mobilenet_v2()
# print("MOBILENETv2:\n", model)

# class FeatureExtractor(nn.Module):
#     def __init__(self, model="resnet50_l2"):
#         super().__init__()
#         self.model = self.get_model(model)
#         print(self.model)
    
#     def get_model(self, model_name):
#         if model_name == "resnet50_l2":
#             resnet50_model = torchvision.models.resnet50()
#             model = nn.Sequential(
#                 resnet50_model.conv1,
#                 resnet50_model.bn1,
#                 resnet50_model.relu,
#                 resnet50_model.maxpool,
#                 resnet50_model.layer1,
#                 resnet50_model.layer2
#             )

#             return model


# hello = FeatureExtractor()
# print("\n\n")
# print(hello)



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
                l_label = int(p*(p-1)/2)*[1] + p*n*[0]
                
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

        if label == 1:
            folder2 = folder
            right_idx = self.dataset_list["right_idx"][idx]
            assert right_idx != -1 
            assert isinstance(right_idx, int)
        elif label == 0:
            if self.random or ((not self.random) and self.dataset_list["right_idx"][idx] == -1):
                folder_list = os.listdir(self.dataset_list) 
                negi_folder = folder_list[generate_randinit(0, len(folder_list), [folder_list.index(folder)])]
                # folder2 = self.dataset_list["folder"][generate_randinit(0, self.dataset_size, [idx])]
                nfiles2 = len(os.listdir(pjoin(self.dataset_path, folder2)))
                right_idx = generate_randinit(0, nfiles2, [])
                if not self.random:
                    self.dataset_list["right_idx"][idx] = (folder2, right_idx)
                    if self.data_config_path == None and idx+5 >= self.dataset_size:
                        with open(pjoin("./data_configs", "config.json"), 'w') as outfile:
                            json.dump(self.dataset_list, outfile)
            else:
                (folder2, right_idx) = self.dataset_list["right_idx"][idx]
        
        left_image = read_image(pjoin(self.dataset_path, folder, str(left_idx)+".jpg"))
        right_image = read_image(pjoin(self.dataset_path, folder2, str(right_idx)+".jpg"))

        if self.transforms != None:
            left_image = self.transforms(left_image)
            right_image = self.transforms(right_image)
        
        return {"data":[left_image.float(), right_image.float()], "label": label}
    

class FaceRecogDataLoader2(Dataset):
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
        return x* int(p*(p-1)/2)* n
    
    def getlist(self):
        if not self.random and self.data_config_path != None:
            with open(self.data_config_path, "r") as file:
                data_list = json.load(file)
            assert isinstance(data_list, dict)
        else:
            data_list = {
                "anch": [],
                "posi": [],
                "nege": []
            }

            p = self.pos_slots
            n = self.neg_slots

            for folder in os.listdir(self.dataset_path):
                nFiles = min(p, len(os.listdir(pjoin(self.dataset_path, folder))))
                for i in range(nFiles-1):
                    for j in range(i+1, nFiles):
                        data_list["anch"] += n*[(folder, None)]
                        data_list["posi"] += n*[(folder, None)]
                        data_list["nege"] += n*[(None, None)]

        assert len(data_list["anch"]) == self.dataset_size, f" len(data_list) = {len(data_list['anch'])}, dataset_size = {self.dataset_size}"
        return data_list


    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, idx):
        anch_folder, anch_idx = self.dataset_list["anch"][idx]
        posi_folder, posi_idx = self.dataset_list["posi"][idx]
        negi_folder, negi_idx = self.dataset_list["negi"][idx]

        assert anch_folder == posi_folder, "Folder mismatch in folder of anchor and positive"

        if negi_folder == None:
            folder_list = os.listdir(self.dataset_path) 
            if self.random:
                negi_folder = folder_list[generate_randinit(0, len(folder_list), [folder_list.index(anch_folder)])]
            else:
                index = folder_list.index(anch_folder)
                index += (idx%self.neg_slots*2)
                index = index%len(folder_list)
                negi_folder = folder_list[index]
            
            assert negi_folder != anch_folder

        
        if anch_idx == None:
            anch_idx = generate_randinit(0, len(pjoin(self.dataset_path, anch_folder)))
        
        if posi_idx == None:
            posi_idx = generate_randinit(0, len(pjoin(self.dataset_path, anch_folder)), [anch_idx])
        
        if negi_idx == None:
            negi_idx = generate_randinit(0, len(pjoin(self.dataset_path, negi_folder)))

        
        # if self.random =



        # folder = self.dataset_list["folder"][idx]
        # label = self.dataset_list["label"][idx]
        # left_idx = self.dataset_list["left_idx"][idx]

        # if label == 1:
        #     folder2 = folder
        #     right_idx = self.dataset_list["right_idx"][idx]
        #     assert right_idx != -1 
        #     assert isinstance(right_idx, int)
        # elif label == 0:
        #     if self.random or ((not self.random) and self.dataset_list["right_idx"][idx] == -1):
        #         folder2 = self.dataset_list["folder"][generate_randinit(0, self.dataset_size, [idx])]
        #         nfiles2 = len(os.listdir(pjoin(self.dataset_path, folder2)))
        #         right_idx = generate_randinit(0, nfiles2, [])
        #         if not self.random:
        #             self.dataset_list["right_idx"][idx] = (folder2, right_idx)
        #             if self.data_config_path == None and idx+5 >= self.dataset_size:
        #                 with open(pjoin("./data_configs", "config.json"), 'w') as outfile:
        #                     json.dump(self.dataset_list, outfile)
        #     else:
        #         (folder2, right_idx) = self.dataset_list["right_idx"][idx]
        
        # left_image = read_image(pjoin(self.dataset_path, folder, str(left_idx)+".jpg"))
        # right_image = read_image(pjoin(self.dataset_path, folder2, str(right_idx)+".jpg"))

        # if self.transforms != None:
        #     left_image = self.transforms(left_image)
        #     right_image = self.transforms(right_image)
        
        # return {"data":[left_image.float(), right_image.float()], "label": label}


if __name__ == "__main__":

    ROOT_PATH = "/home2/kushagra0301/projects_self/face_recognition/datasets/lfw_dataset"

    train_data = FaceRecogDataLoader(ROOT_PATH, "train")
    print("train:",train_data.dataset_size)

    val_data = FaceRecogDataLoader(ROOT_PATH, "val", random=False, data_config_path="./data_configs/val_config.json")
    print("val:",val_data.dataset_size)


    for sample in tqdm(train_data):
        assert sample["label"] == 1 or sample["label"] == 0
    
    for sample in tqdm(val_data):
        assert sample["label"] == 1 or sample["label"] == 0
    
    # print()

    train_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=3)
    
    # print("Done")

    # for batch_sample in tqdm(train_dataloader):
    #     # print(batch_sample["label"].shape)
    #     # print(batch_sample["data"][0].shape, batch_sample["data"][1].shape)
    #     pass