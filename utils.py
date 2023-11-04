import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_auprc, binary_accuracy
from tqdm import tqdm

def eval_accuracy(pred, target, thres=0.5):
    acc = binary_accuracy(pred.squeeze(), target.squeeze(), threshold=thres)
    return acc

def eval_mae(pred, target):
    mae = nn.functional.l1_loss(pred, target)
    return mae

def eval_auc(pred, target):
    auc = binary_auprc(pred.squeeze(), target.squeeze())
    return auc

def train_loop(dataloader, model, loss_fns, optim, device):
    model.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    totat_acc = 0
    total_mae = 0
    total_auc = 0

    pbar = tqdm(dataloader, leave=False)
    pbar.set_description(f"t_loss = 0, t_mae = 0, t_acc = 0, t_auc = 0")
    for batch in pbar:

        X = batch["data"]
        y = batch["label"].unsqueeze(-1).to(device)

        (pred, query_emb, target_emb) = model(X[0].to(device), X[1].to(device))
        if pred == None:
            pred = nn.PairwiseDistance(keepdim=True)(query_emb, target_emb)
            pred = nn.Sigmoid()(pred)
        
        loss = 0
        for key in loss_fns.keys():
            if key == "bce_loss":
                l = loss_fns[key](pred, y.type(pred.dtype))
            elif key == "distance_loss":
                l = loss_fns[key](query_emb, target_emb, y)
            loss += l

        total_loss += loss

        mae = eval_mae(pred, y)
        total_mae += mae

        acc = eval_accuracy(pred, y)
        totat_acc += acc

        auc = eval_auc(pred, y)
        total_auc += auc

        pbar.set_description(f"t_loss = {loss:>4f}, t_mae = {mae:>4f}, t_acc = {acc:>4f}, t_auc = {auc:>4f}")

        loss.backward()
        optim.step()
        optim.zero_grad()

    train_loss = total_loss/num_batches
    train_mae = total_mae/num_batches
    train_acc = totat_acc/num_batches
    train_auc = total_auc/num_batches

    print(f"\tTRAINING  : Loss = {train_loss:>7f}, MAE = {train_mae:>4f}, ACC = {train_acc:>4f}, AUC = {train_auc:>4f}")

    return train_loss, train_mae, train_acc, train_auc

def val_loop(dataloader, model, loss_fns, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    val_loss = 0
    val_mae = 0
    val_acc = 0
    val_auc = 0

    with torch.no_grad():
        # for i, batch in enumerate(dataloader):
        pbar = tqdm(dataloader, leave=False)
        pbar.set_description(f"v_loss = 0, v_mae = 0, v_acc = 0, v_auc = 0")
        for batch in pbar:
            X = batch["data"]
            y = batch["label"].unsqueeze(-1).to(device)

            (pred, query_emb, target_emb) = model(X[0].to(device), X[1].to(device))
            if pred == None:
                pred = nn.PairwiseDistance(keepdim=True)(query_emb, target_emb)
                pred = nn.Sigmoid()(pred)

            loss = 0
            for key in loss_fns.keys():
                if key == "bce_loss":
                    l = loss_fns[key](pred, y.type(pred.dtype))
                elif key == "distance_loss":
                    l = loss_fns[key](query_emb, target_emb, y)
                loss += l

            acc = eval_accuracy(pred, y)
            mae = eval_mae(pred, y)
            auc = eval_auc(pred, y)

            pbar.set_description(f"v_loss = {loss:>4f}, v_mae = {mae:>4f}, v_acc = {acc:>4f}, v_auc = {auc:>4f}")

            val_loss += loss.item()
            val_mae += mae
            val_acc += acc
            val_auc += auc

    
    val_loss /= num_batches
    val_mae /= num_batches
    val_acc /= num_batches
    val_auc /= num_batches

    print(f"\tVALIDATION: Loss = {val_loss:>7f}, MAE = {val_mae:>4f}, ACC = {val_acc:>4f}, AUC = {val_auc:4f}")

    return val_loss, val_mae, val_acc, val_auc


class CustomDistanceLoss(nn.Module):
    def __init__(self, rMargin=1, p=2, eps=0.1):
        super().__init__()
        self.rmargin = rMargin
        self.p_distance = nn.PairwiseDistance(p=2)
        self.eps=0.1
    
    def forward(self, emb1, emb2, target):
        ldistance = self.p_distance(emb1, emb2)
        rdistance = self.rmargin+self.eps - ldistance
        loss = target*ldistance + (1-target)*(torch.maximum(torch.zeros_like(rdistance), rdistance))
        loss = loss.mean()
        return loss
