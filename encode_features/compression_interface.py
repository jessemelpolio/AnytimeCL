import torch
from sklearn.decomposition import PCA
import numpy as np


def pca_compress(clip_intermediate_feature, args, atten_weights = None):
    pca = PCA(n_components=args.components)
    
    if args.per_instance:
        clip_intermediate_feature = clip_intermediate_feature.unsqueeze(0)
        if args.CLS_weight:
            if atten_weights == None:
                raise ValueError("Attention weight is needed to perform CLS weighting")
            cls_atten = torch.exp(atten_weights[1:,0])
            atten_sum = sum(cls_atten)
            cls_coef = cls_atten/atten_sum
            cls_coef = torch.cat((torch.tensor([1]).to(args.device),cls_coef), dim = 0)
            weighted_feature = (cls_coef.view(1,len(cls_coef),1) * clip_intermediate_feature)

            feature_flat = weighted_feature.view(-1, weighted_feature.shape[2]).to('cpu')
            pca_feature = torch.tensor(pca.fit_transform(feature_flat)).to(args.device).float()
            principle_components = torch.tensor(pca.components_).to(args.device)
            means = torch.tensor(pca.mean_).to(args.device)
            compressed = (feature_flat.to(args.device) - means) @ principle_components.T
        else:
            feature_flat = clip_intermediate_feature.view(-1, clip_intermediate_feature.shape[2]).to('cpu')   
            compressed = torch.tensor(pca.fit_transform(feature_flat)).to(args.device)
            principle_components = torch.tensor(pca.components_).to(args.device)
            means = torch.tensor(pca.mean_).to(args.device)
        
    else:
        feature_flat = clip_intermediate_feature.view(-1, clip_intermediate_feature.shape[2]).to('cpu')     
        compressed = torch.tensor(pca.fit_transform(feature_flat)).to(args.device)
        principle_components = torch.tensor(pca.components_).to(args.device)
        means = torch.tensor(pca.mean_).to(args.device)
    
    
    max_p = None
    min_p = None
    max_c = None
    min_c = None
    max_m = None
    min_m = None

    if args.int_quantize:
        max_p = torch.max(principle_components,1)[0].unsqueeze(0).T
        min_p = torch.min(principle_components,1)[0].unsqueeze(0).T

        max_c = torch.max(compressed,1)[0].unsqueeze(0).T
        min_c = torch.min(compressed,1)[0].unsqueeze(0).T

        max_m = torch.max(means)
        min_m = torch.min(means)

        
        principle_components = ((principle_components - min_p)/(max_p-min_p)*args.int_range).type(torch.uint8)
        compressed = ((compressed-min_c)/(max_c-min_c) * args.int_range).type(torch.uint8)
        means = ((means-min_m)/(max_m-min_m) * args.int_range).type(torch.uint8)

    return compressed, principle_components, means, max_p, min_p, max_c, min_c, max_m, min_m



def restore_pca_compressed(compressed_feature, principle_components, means, args, max_min_vals = None, original_shape = None):
    if args.int_quantize:
        max_p = max_min_vals[0]
        min_p = max_min_vals[1]
        max_c = max_min_vals[2]
        min_c = max_min_vals[3]
        max_m = max_min_vals[4]
        min_m = max_min_vals[5]
        principle_components = principle_components/args.int_range*(max_p-min_p) + min_p
        compressed_feature = compressed_feature/args.int_range * (max_c-min_c) +min_c
        means = means/args.int_range * (max_m-min_m) +min_m
    reconstructed = (compressed_feature @ principle_components + means).astype(np.float32)
    return reconstructed


def decompress_from_npz(args, raw_image):
    pack_load = np.load(raw_image)
    compressed = pack_load["compressed"]
    principle = pack_load["principle"]
    means = pack_load["means"]
    max_min_val = None
    if args.int_quantize:
        max_p = pack_load["max_p"]
        min_p = pack_load["min_p"]
        max_c = pack_load["max_c"]
        min_c = pack_load["min_c"]
        max_m = pack_load["max_m"]
        min_m = pack_load["min_m"]
        max_min_val = [max_p, min_p, max_c, min_c,max_m,min_m]
    
    raw_image = restore_pca_compressed(compressed, principle, means, args, max_min_val)
    raw_image = torch.from_numpy(raw_image)
    return raw_image






    




