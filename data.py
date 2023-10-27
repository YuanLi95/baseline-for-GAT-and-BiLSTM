import math

import torch
import numpy as np
import  pickle
import os
import json

class DataIterator(object):
    def __init__(self, x_data,x_mask_data,x_edge_data, args):
        self.x_data,self.x_mask_data,self.x_edge_data,=x_data,x_mask_data,x_edge_data,
        #date跟fearture的分开
        self.x_date,self.x_feature,self.x_tags=self.x_data[:,:,0],self.x_data[:,:,1:-2],x_data[:,:,-2:]
        # print(self.x_date.shape,self.x_feature.shape,self.x_tags.shape)
        self.args = args
        self.batch_count = math.ceil(len(x_data)/args.batch_size)

    def get_batch(self, index):
        x_date = []
        x_feature = []
        x_mask_data=[]
        x_edge_data = []
        x_tags = []


        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.x_data))):

            x_date.append(self.x_date[i])
            x_feature.append(self.x_feature[i].float() )

            # print(self.x_mask_data[i].shape)
            x_mask_data.append(self.x_mask_data[i])
            # print(self.x_edge_data[i].shape)
            x_edge_data.append(self.x_edge_data[i])
            x_tags.append(self.x_tags[i].float() )

        x_date = torch.stack(x_date).to(self.args.device)
        x_feature = torch.FloatTensor(torch.stack(x_feature)).to(self.args.device)
        x_mask_data = torch.stack(x_mask_data).to(self.args.device)
        x_edge_data = torch.stack(x_edge_data).to(self.args.device)
        x_tags = torch.stack(x_tags).to(self.args.device)


        return  x_date,x_feature,x_mask_data,x_edge_data,x_tags