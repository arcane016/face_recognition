import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import os
from os.path import join as pjoin

from dataloader import FaceRecogDataLoader
from models import FaceRecogNet
from utils import train_loop, val_loop, CustomDistanceLoss

import wandb

MODEL_CONFIG = {
    "extractor":{
        "backbone": "resnet50", #"resnet50",
    },
    "classifier":{
        "use_cls_head":True,
        "aggregation":"add", # "concat", "add"
        "attention":None, # "self", "cross", None
        "f_size":512 # 512, 256
    }
}

TRAINING_CONFIG = {
    "loss_fn":["bce_loss", "distance_loss"],
    "optimizer":"adam",
    "epochs":20,
    "batch_size":128,
    "dataset":"lfw_dataset",
    "metrics":["mae", ("acc", 0.5), "auc"]
}

DATASET_PATHS= {
    "lfw_dataset":"/home2/kushagra0301/projects_self/face_recognition/datasets/lfw_dataset"
}

CONFIGS = {
    "model":MODEL_CONFIG,
    "training": TRAINING_CONFIG
}


RUN_INDEX = 2



if MODEL_CONFIG["classifier"]["use_cls_head"] == True:
    if MODEL_CONFIG["classifier"]["attention"] != None:
        folder_name = MODEL_CONFIG["extractor"]["backbone"]+"_"+MODEL_CONFIG["classifier"]["aggregation"]+"_"+MODEL_CONFIG["classifier"]["attention"]
    else:
        folder_name = MODEL_CONFIG["extractor"]["backbone"]+"_"+MODEL_CONFIG["classifier"]["aggregation"]+"_"+"none"
else:
    folder_name = MODEL_CONFIG["extractor"]["backbone"]+"_"+"no_cls_head"

MODEL_CKPT_PATH = pjoin(f"./save/run{RUN_INDEX}", folder_name)
if not os.path.exists(MODEL_CKPT_PATH):
    os.makedirs(MODEL_CKPT_PATH)
else:
    # raise(f"{MODEL_CKPT_PATH} exists")
    pass

with open(pjoin(MODEL_CKPT_PATH, "log.txt"), "w") as file:
    file.write("")

def write_in_file(text, file_path=pjoin(MODEL_CKPT_PATH, "log.txt")):
    with open(file_path, "a") as file:
        file.write(text)

EPOCHS = TRAINING_CONFIG["epochs"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
DATASET_PATH = DATASET_PATHS[TRAINING_CONFIG["dataset"]]

USE_WANDB = True

if USE_WANDB:
    wandb.login()
    wandb.init(
        project="Face Recognition",
        notes = f"run{RUN_INDEX}/{folder_name}"
    )

transforms = transforms.Resize((128,128))

device = ("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = FaceRecogDataLoader(DATASET_PATH, "train", transforms, random=True, data_config_path=None)
train_dataloader =  DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, drop_last=True)

val_dataset = FaceRecogDataLoader(DATASET_PATH, "val", transforms, random=False, data_config_path="./data_configs/val_config2.json")
val_dataloader =  DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, drop_last=True)

# print("1\n\n")

model = FaceRecogNet(
    backbone=MODEL_CONFIG["extractor"]["backbone"],
    use_cls_head=MODEL_CONFIG["classifier"]["use_cls_head"],
    aggregation=MODEL_CONFIG["classifier"]["aggregation"],
    attention=MODEL_CONFIG["classifier"]["attention"],
    f_size=MODEL_CONFIG["classifier"]["f_size"]
).to(device)
# print("3\n\n")

loss_fns = {}
for loss_type in TRAINING_CONFIG["loss_fn"]:
    if loss_type == "bce_loss":
        loss_fns[loss_type] = nn.BCELoss()
    elif loss_type == "distance_loss":
        loss_fns[loss_type] = CustomDistanceLoss()
    else:
        raise("Loss fn error")

if TRAINING_CONFIG["optimizer"] == "adam":
    optim = torch.optim.Adam(model.parameters())
else:
    raise("Optim error")

# print("2\n\n")

for epoch in range(EPOCHS):
    print(f"EPOCH: [{epoch+1}/{EPOCHS}]", end="\n")
    write_in_file(f"EPOCH: [{epoch+1}/{EPOCHS}]\n")

    train_loss, train_mae, train_acc, train_auc = train_loop(train_dataloader, model, loss_fns, optim, device)
    write_in_file(f"\tTRAINING  : Loss = {train_loss:>7f}, MAE = {train_mae:>4f}, ACC = {train_acc:>4f}, AUC = {train_auc:>4f}\n")

    val_loss, val_mae, val_acc, val_auc = val_loop(val_dataloader, model, loss_fns, device)
    write_in_file(f"\tVALIDATION: Loss = {val_loss:>7f}, MAE = {val_mae:>4f}, ACC = {val_acc:>4f}, AUC = {val_auc:>4f}\n\n")

    if USE_WANDB:
        wandb.log({"train_loss":train_loss, "train_mae":train_mae, "train_acc":train_acc, "train_auc":train_auc, "val_loss":val_loss, "val_mae":val_mae, "val_acc":val_acc, "val_auc":val_auc})

    torch.save(model.state_dict(), pjoin(MODEL_CKPT_PATH, f"ckpt_{epoch+1}.pth"))
    
    print("\n", end="")