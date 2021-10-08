import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str, timing
# from models import print_log
import models
import random
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict

class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

  
  def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
            print("filter codebook done")
        elif len(weight_torch.size()) == 2:
            weight_torch = weight_torch.view(weight_torch.size()[0], weight_torch.size()[1], 1, 1)
            codebook = self.get_filter_codebook(weight_torch, compress_rate, length)
            print("filter codebook for fc done")
        else:
            pass
        return codebook

    @timing
    def get_filter_similar_old(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            print('weight_vec.size', weight_vec.size())
            # distance using pytorch function
            similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            for x1, x2 in enumerate(filter_large_index):
                for y1, y2 in enumerate(filter_large_index):
                    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
                    pdist = torch.nn.PairwiseDistance(p=2)
                    # print('weight_vec[x2].size', weight_vec[x2].size())
                    similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
                    # print('weight_vec[x2].size after', weight_vec[x2].size())
            # more similar with other filter indicates large in the sum of row
            similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('filter_large_index', filter_large_index)
            print('filter_small_index', filter_small_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            # for cos similarity
            # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook
     
    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
                    
                    
     def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        if 'vgg' in args.arch:
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            cfg_CP_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_Thinet_conv = [32, 32, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
            if args.VGG_pruned_style == "CP_5x":
                cfg_now = cfg_CP_5x
            elif args.VGG_pruned_style == "Thinet_conv":
                cfg_now = cfg_Thinet_conv

            cfg_index = 0
            previous_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                if len(item[1].size()) == 4:
                    if not previous_cfg:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] = 1
                        self.distance_rate[index] = 1 - cfg_now[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_now[cfg_index])
                        cfg_index += 1
        elif "resnet" in args.arch:
            for index, item in enumerate(self.model.parameters()):
                self.compress_rate[index] = 1
                self.distance_rate[index] = 1
            for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
                self.compress_rate[key] = rate_norm_per_layer
                self.distance_rate[key] = rate_dist_per_layer
            # different setting for  different architecture
            if args.arch == 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [21, 36, 51]
            elif args.arch == 'resnet34':
                last_index = 108
                skip_list = [27, 54, 93]
            elif args.arch == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
            elif args.arch == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
            elif args.arch == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
            self.mask_index = [x for x in range(0, last_index, 3)]
            # skip downsample layer
            if args.skip_downsample == 1:
                for x in skip_list:
                    self.compress_rate[x] = 1
                    self.mask_index.remove(x)
                    print(self.mask_index)
            else:
                pass
            
            
    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index])
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")





   
   
