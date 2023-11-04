import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1) # For Hin == Hout and Win == Wout
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_res = nn.Conv2d(in_channels, out_channels, 3, stride=2)
        self.bn_res = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)

        res = self.conv_res(inp)
        res = self.bn_res(res)

        x = x+res

        return x

class FeatureExtractor(nn.Module):
    def __init__(self, model="resnet50"):#, out_layer="layer2"):
        super().__init__()
        self.model = self.get_model(model)
    
    def get_model(self, model_name):#, out_layer):
        if model_name == "resnet50":
            resnet50_model = torchvision.models.resnet50()
            children_list = []
            for n,c in resnet50_model.named_children():
                children_list.append((n,c))
                if n == "layer2":
                    break
            
            model = nn.Sequential(OrderedDict(children_list))
            return model
        
        if model_name == "efficientnetv2s":
            effnetv2_model = torchvision.models.efficientnet_v2_s()
            model = effnetv2_model.features[:-1]

            return model
        
        if model_name == "custom_s":
            chs = [3, 64, 128, 256]
            model = nn.Sequential(OrderedDict([
                (f"block{i+1}", CustomBlock(chs[i], chs[i+1])) for i in range(len(chs)-1)
            ]))

            return model

    
    def forward(self, input):
        out = self.model(input)
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.squeeze()
        return out

class Classifier(nn.Module):
    def __init__(self, aggregation_type, attention_type, f_size=512):
        super().__init__()
        self.aggregation = aggregation_type
        self.attention = attention_type
        self.self_att_l = None
        self.self_att_r = None
        self.cross_att_l = None
        self.cross_att_r = None
        self.classifier = None

        if self.attention != None:

            self.self_att_l = nn.MultiheadAttention(batch_first=True, num_heads=1, embed_dim=f_size, kdim=f_size, vdim=f_size)
            self.self_att_r = nn.MultiheadAttention(batch_first=True, num_heads=1, embed_dim=f_size, kdim=f_size, vdim=f_size)
            self.cross_att_l = nn.MultiheadAttention(batch_first=True, num_heads=1, embed_dim=f_size, kdim=f_size, vdim=f_size)
            self.cross_att_r = nn.MultiheadAttention(batch_first=True, num_heads=1, embed_dim=f_size, kdim=f_size, vdim=f_size)

        if self.aggregation == "add":
            if f_size == 512:
                self.classifier = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8,1),
                    nn.Sigmoid()
                )

            elif f_size == 256:
                self.classifier = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

                # self.classifier = nn.Sequential(
                #     nn.Linear(256, 16),
                #     nn.ReLU(),
                #     nn.Linear(16, 1),
                #     nn.Sigmoid()
                # )

        elif self.aggregation == "concat":
            if f_size == 512:
                self.classifier = nn.Sequential(
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

            elif f_size == 256:
                self.classifier = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8,1),
                    nn.Sigmoid()
                )

                # self.classifier = nn.Sequential(
                #     nn.Linear(512, 32),
                #     nn.ReLU(),
                #     nn.Linear(32, 1),
                #     nn.Sigmoid()
                # )
        elif self.aggregation == "diff":
            if f_size == 512:
                self.classifier = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8,1),
                    nn.Sigmoid()
                )

            elif f_size == 256:
                self.classifier = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

                # self.classifier = nn.Sequential(
                #     nn.Linear(256, 16),
                #     nn.ReLU(),
                #     nn.Linear(16, 1),
                #     nn.Sigmoid()
                # )


    def forward(self, left, right):
        
        if self.aggregation == "add":

            if self.attention == "self":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
            elif self.attention == "cross":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
                left, _ = self.cross_att_l(right, left, left)
                right, _ = self.cross_att_r(left, right, right)

            emb = left+right
            return self.classifier(emb)

        
        elif self.aggregation == "concat":

            if self.attention == "self":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
            elif self.attention == "cross":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
                left1, _ = self.cross_att_l(right, left, left)
                right1, _ = self.cross_att_r(left, right, right)
                left = left1
                right = right1

            emb = torch.cat((left, right), 1)
            return self.classifier(emb)
        
        elif self.aggregation == "diff":

            if self.attention == "self":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
            elif self.attention == "cross":
                left, _ = self.self_att_l(left, left, left)
                right, _ = self.self_att_r(right, right, right)
                left, _ = self.cross_att_l(right, left, left)
                right, _ = self.cross_att_r(left, right, right)

            emb = torch.abs(left-right)*10
            return self.classifier(emb)


class FaceRecogNet(nn.Module):
    def __init__(self, backbone, use_cls_head, aggregation, attention, f_size):
        super().__init__()

        self.extractor = FeatureExtractor(model=backbone)#, out_layer=out_layer)
        self.use_cls_head = use_cls_head
        if self.use_cls_head:
            self.classifier = Classifier(aggregation_type=aggregation, attention_type=attention, f_size=f_size)
    
    def forward(self, query_image, target_image):

        query_emb = self.extractor(query_image)
        target_emb = self.extractor(target_image)

        if not self.use_cls_head:
            # out = nn.PairWiseDistance()(query_emb, target_emb)
            return (None, query_emb, target_emb)
        
        out = self.classifier(query_emb, target_emb)

        return (out, query_emb, target_emb)



