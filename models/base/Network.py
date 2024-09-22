import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import resnet18
from models.resnet20_cifar import *
from models.resnet12 import resnet12, resnet12_wide
import numpy as np

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            #self.encoder = resnet20()
            #self.num_features = 64
            #self.map_size = 8
            self.encoder = resnet12_wide(False)
            self.num_features = 640
            self.map_size = 8

        if self.args.dataset in ['mini_imagenet']:
            #self.encoder = resnet18(False, args)  # pretrained=False
            #self.num_features = 512
            #self.map_size = 7
            self.encoder = resnet12_wide(False)
            self.num_features = 640
            self.map_size = 5
        
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
            self.map_size = 7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc_base = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        
        if args.map_metric_option != 'none':
            self.fc_map = nn.Parameter(torch.empty((self.args.num_classes, self.num_features, self.map_size, self.map_size), dtype=torch.float32))
            self.register_parameter(name='fc_map', param=self.fc_map)
            nn.init.kaiming_uniform_(self.fc_map, a=math.sqrt(5))

            self.fc_map_base = nn.Parameter(torch.empty((self.args.num_classes, self.num_features, self.map_size, self.map_size), dtype=torch.float32))
            self.register_parameter(name='fc_map_base', param=self.fc_map_base)
            nn.init.kaiming_uniform_(self.fc_map_base, a=math.sqrt(5))
            
            if self.num_features == 64:
                self.fc_map_temperature = nn.Parameter(torch.tensor(1.0))
            else:
                self.fc_map_temperature = nn.Parameter(torch.tensor(16.0))
            self.register_parameter(name='fc_map_temperature', param=self.fc_map_temperature)
            

        if args.temperature < 0:
            self.temperature = nn.Parameter(torch.tensor(abs(args.temperature), dtype=torch.float32))
            self.register_parameter(name='fc_temperature', param=self.temperature)
            self.args.temperature = self.temperature

        if args.dropout_rate != 0.0:
            self.dropout_fn = nn.Dropout(args.dropout_rate)
        else:
            self.dropout_fn = None

        self.end_points = {}




    def map_metric_forward(self, proto=None, feat=None, is_base=False, aaa=False):
        # encode must have been called before !!!
        args = self.args
        if proto is None:
            proto = self.fc_map if is_base==False else self.fc_map_base # [num_classes, c, hp, wp]

        num_classes, c, hp, wp = proto.shape
        
        if feat is None:
            feat = self.end_points['final_map'] # [b, c, h, w]

        b, _, h, w = feat.shape


        assert args.map_metric_option == 'cka'
        proto = proto.view(num_classes, c, hp*wp)
        feat = feat.view(b, c, h*w)
        
        if self.args.map_pow != 1.0:
            neg_mask = 1 - torch.sign(torch.sign(feat) + 1)
            neg_feat = feat * neg_mask
            feat = F.relu(feat)
            feat = feat ** args.map_pow
            feat = feat + neg_feat

        logits = self.cka_logits(feat, proto)
        self.cka_logit_preds = logits

        logits = logits * self.fc_map_temperature
        return logits




    def forward_metric(self, x, is_base=False, epoch=None):
        x = self.encode(x)
     
        if self.args.bkb_feat_pow != 1.0:
            neg_mask = 1 - torch.sign(torch.sign(x) + 1)
            neg_x = x * neg_mask
            x = F.relu(x)
            x = x ** self.args.bkb_feat_pow
            x = x + neg_x
       
        self.end_points['final_feature'] = x

        if 'cos' in self.mode:
            weight = self.fc.weight if is_base==False else self.fc_base.weight
            
            if self.args.dataset == 'cub200' and self.args.map_metric_option != 'none':
                x = x - x.mean(dim=1, keepdim=True)
                weight = weight - weight.mean(dim=1, keepdim=True)           


            if self.dropout_fn is None:
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(weight, p=2, dim=-1))
            else:
                x = F.linear(self.dropout_fn(F.normalize(x, p=2, dim=-1)), F.normalize(weight, p=2, dim=-1))
            
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            weight = self.fc.weight if is_base==False else self.fc_base.weight

            if self.training:
                x = x - x.mean(dim=1, keepdim=True)
                weight = weight - weight.mean(dim=1, keepdim=True)

                x = x.unsqueeze(1) # [b, 1, c]
                weight = weight.unsqueeze(0) # [1, num_classes, c]
                dist = torch.linalg.norm(x - weight, ord=2, dim=-1) ** 2 # [b, num_classes]
                x = - dist

            else:
                x = x - x.mean(dim=1, keepdim=True)
                weight = weight - weight.mean(dim=1, keepdim=True)

                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(weight, p=2, dim=-1))

            x = x * self.args.temperature

                       
        return x

    def encode(self, x):
        x = self.encoder(x)

        self.end_points['final_map'] = x # [b, c, h, w]      
        
        x = F.adaptive_avg_pool2d(x, 1)       
        x = x.squeeze(-1).squeeze(-1)

        return x


    def forward(self, input, is_base=False, epoch=None):
        if self.mode != 'encoder':
            input = self.forward_metric(input, is_base, epoch)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)

            self.end_points['final_feature'] = input

            return input
        else:
            raise ValueError('Unknown mode')


    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()
            
            self.end_points['final_feature'] = data
        
        if self.args.not_data_init_novel:
            new_fc = nn.Parameter(
                        torch.rand(len(class_list), self.num_features, device="cuda"),
                        requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))

            if self.args.map_metric_option != 'none':
                new_fc_map = nn.Parameter(
                        torch.empty((len(class_list), self.num_features, self.fc_map.shape[2], self.fc_map.shape[3]), dtype=torch.float32, device="cuda"), 
                        requires_grad=True)
                nn.init.kaiming_uniform_(new_fc_map, a=math.sqrt(5))
                self.new_fc_map = new_fc_map

        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            if self.args.in_domain_feat_cls_weight != 0.0:
                assert args.map_metric_option != 'none'
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        if self.args.map_metric_option != 'none':
            new_fc_map = []

        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)

            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            
            if self.args.map_metric_option != 'none':
                backbone_map = self.end_points['final_map'][data_index]
                map_proto = backbone_map.mean(0)
                self.fc_map.data[class_index] = map_proto
                new_fc_map.append(map_proto)

        new_fc=torch.stack(new_fc,dim=0)
        if self.args.map_metric_option != 'none':
            self.new_fc_map = torch.stack(new_fc_map, dim=0)

        return new_fc

    def get_logits(self,x,fc):
        if self.args.map_metric_option != 'none':
            weight = fc
            if 'dot' in self.args.new_mode:
                x = x - x.mean(dim=1, keepdim=True)
                weight = weight - weight.mean(dim=1, keepdim=True)
                x = x.unsqueeze(1) # [b, 1, c]
                weight = weight.unsqueeze(0) # [1, num_classes, c]
                dist = torch.linalg.norm(x - weight, ord=2, dim=-1) ** 2 # [b, num_classes]
                x = - dist
                return x
            elif 'cos' in self.args.new_mode:
                x = x - x.mean(dim=1, keepdim=True)
                weight = weight - weight.mean(dim=1, keepdim=True)
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(weight, p=2, dim=-1))
                return x * self.args.temperature
        else:
            if 'dot' in self.args.new_mode:
                return F.linear(x,fc)
            elif 'cos' in self.args.new_mode:
                return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        


    def update_fc_ft(self,new_fc,data,label,session):
        is_base = True if self.args.not_data_init else False
        if self.args.map_metric_option != 'none':
            new_fc=new_fc.clone().detach()
            new_fc.requires_grad=True

            new_fc_map = self.new_fc_map.clone().detach()
            new_fc_map.requires_grad = True

            optimized_parameters = [{'params': [new_fc, new_fc_map]}]

        else:
            new_fc=new_fc.clone().detach()
            new_fc.requires_grad=True
            optimized_parameters = [{'params': new_fc}]

        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                if is_base:
                    old_fc = self.fc_base.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                else:
                    old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()

                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)

                if self.args.map_metric_option != 'none':
                    if is_base:
                        old_fc_map = self.fc_map_base[:self.args.base_class + self.args.way * (session - 1), :].detach()
                    else:
                        old_fc_map = self.fc_map[:self.args.base_class + self.args.way * (session - 1), :].detach()

                    fc_map = torch.cat([old_fc_map, new_fc_map], dim=0)

                    map_logits = self.map_metric_forward(proto=fc_map, feat=self.end_points['final_map'].detach(), is_base=is_base)
                    map_metric_loss = F.cross_entropy(map_logits, label)
                    loss = self.args.backbone_feat_cls_weight * loss + self.args.map_metric_cls_w * map_metric_loss


                    if self.args.ft_primitive_recon_weight != 0.0:
                        base_proto = old_fc_map[:self.args.base_class].detach() # [base_class, c, hp, wp]
                        bc, c, hp, wp = base_proto.shape
                        base_proto = base_proto.reshape(bc, c, hp*wp) # denote s = hp*wp
                        bc, c, s = base_proto.shape
                        base_proto = base_proto.permute(0, 2, 1).reshape(bc*s, c) # [bc*s, c]

                        novel_proto = new_fc_map # [novel_class, c, hp, wp]
                        nc = novel_proto.shape[0]
                        novel_proto = novel_proto.reshape(nc, c, s)
                        novel_proto = novel_proto.permute(0, 2, 1).reshape(nc*s, c) # [novel_class*s, c]
            
                        sims = - torch.cdist(novel_proto, base_proto, p=2) ** 2

                        # soft reuse
                        atten = torch.softmax(sims * self.args.ft_prim_recon_tau, dim=-1)

                        reused_novel_proto = torch.matmul(atten, base_proto) # [nc*s, c]
                        
                        reused_novel_proto = reused_novel_proto.reshape(nc, hp, wp, c)
                        reused_novel_proto = reused_novel_proto.permute(0, 3, 1, 2)
                        reused_novel_logits = self.map_metric_forward(proto=reused_novel_proto, feat=self.end_points['final_map'].detach(), is_base=is_base)
                        novel_map_logits = map_logits[:, -new_fc_map.shape[0]:]

                        reused_logits = torch.cat([map_logits[:, :-new_fc_map.shape[0]], reused_novel_logits], dim=1)
                        prim_recon_loss = F.cross_entropy(reused_logits, label)
                        
                        loss = self.args.ft_primitive_recon_weight * prim_recon_loss + loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        if is_base:
            self.fc_base.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
        else:
            self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

        if self.args.map_metric_option != 'none':       
            if is_base:
                self.fc_map_base.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc_map.data)
            else:
                self.fc_map.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc_map.data)



    def cka_logits(self, feat, proto):
        # equivalent to linear_CKA, batch computation
        # feat: [b, c, h*w]
        # proto: [num_classes, c, hp*wp]
        def centering(feat):
            assert len(feat.shape) == 3
            return feat - torch.mean(feat, dim=1, keepdims=True)
        
        def cka(va, vb):
            return torch.norm(torch.matmul(va.t(), vb)) ** 2 / (torch.norm(torch.matmul(va.t(), va)) * torch.norm(torch.matmul(vb.t(), vb)))
        
        proto = centering(proto); feat = centering(feat)

        ### equivalent implementation ###
        proto = proto.unsqueeze(0) # [1, num_classes, c, hp*wp]
        feat = feat.unsqueeze(1) # [b, 1, c, h*w]

        cross_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), proto), dim=[2,3]) ** 2 # [b, num_classes]
        feat_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), feat), dim=[2,3]) # [b, 1]
        proto_norm = torch.norm(torch.matmul(proto.permute(0, 1, 3, 2), proto), dim=[2,3]) # [1, num_classes]

        logits = cross_norm / (feat_norm * proto_norm) # [b, num_classes]

        return logits

