import torch
import numpy as np
from terminaltables import AsciiTable
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn
from src.model import CNNModel
def parse_module_defs2(model):
    all_conv = []
    bn_id = []
    ignore_idx = set()
    #add bn module to prune_id and every first conv layer of block like layer *.0.conv1
    for i , layer in enumerate(list(model.named_modules())):
        # choose the first conv in every basicblock
        if '.bn1' in layer[0]:
            bn_id.append(i)
        if 'conv' in layer[0]:
            all_conv.append(i)
        # if 'downsample' in layer[0]:
        #     ignore_idx.add(i-1)
        # if layer[0] in ['layer{}.0.conv1'.format(x) for x in range(2,5)]:
        #     ignore_idx.add(i+1)
    prune_id = [idx for idx in bn_id if idx not in ignore_idx]
    return prune_id,ignore_idx,all_conv



    # [layer for layer in list(model.named_modules()) if 'conv' in layer[0]]

def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, model, s, prune_idx):
        if sr_flag:
            bn_weights = gather_bn_weights(list(model.named_modules()), prune_idx)
            bn_numpy = bn_weights.numpy()
            if np.mean(bn_numpy) < 0.01:
                s = s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = list(model.named_modules())[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1



def obtain_quantiles(bn_weights, num_quantile=5):
    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total // num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile + 1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):
    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))

        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]

        elif len(route_in_idxs) == 2:
            # return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
            if module_defs[route_in_idxs[0]]['type'] == 'upsample':
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional':
                mask1 = CBLidx2mask[route_in_idxs[0]]
            if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                mask2 = CBLidx2mask[route_in_idxs[1]]
            else:
                mask2 = CBLidx2mask[route_in_idxs[1] - 1]
            return np.concatenate([mask1, mask2])

        elif len(route_in_idxs) == 4:
            # spp结构中最后一个route
            mask = CBLidx2mask[route_in_idxs[-1]]
            return np.concatenate([mask, mask, mask, mask])

        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':  # tiny
        return CBLidx2mask[idx - 2]


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        for next_idx in next_idx_list:
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):
    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)


def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)



        elif model_def['type'] == 'route':
            # spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

    return pruned_model


def get_mask(model, prune_idx, shortcut_idx):
    sort_prune_idx = [idx for idx in prune_idx if idx not in shortcut_idx]
    bn_weights = gather_bn_weights(model.module_list, sort_prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]
    highest_thre = []
    for idx in sort_prune_idx:
        # .item()可以得到张量里的元素值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)
    filters_mask = []
    idx_new = dict()
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in prune_idx:
        bn_module = model.module_list[idx][1]
        if idx not in shortcut_idx:
            mask = obtain_bn_mask(bn_module, torch.tensor(highest_thre)).cpu()
            idx_new[idx] = mask
        else:
            mask = idx_new[shortcut_idx[idx]]
            idx_new[idx] = mask

        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask


def get_mask2(model, prune_idx, percent):
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    filters_mask = []
    for idx in prune_idx:
        bn_module = model.module_list[idx][1]
        mask = obtain_bn_mask(bn_module, thre).cpu()
        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask


def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())
def sort_bn(model, prune_idx):
    # size_list = [m[idx][1].weight.data.shape[0] for idx, m in enumerate(list(model.named_modules())) if idx in prune_idx]
    size_list = [m[1].weight.data.shape[0] for idx, m in enumerate(model.named_modules()) if idx in prune_idx]
    # bn_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m[1] for idx, m in enumerate(list(model.named_modules())) if idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))

    index = 0
    for module, size in zip(bn_prune_layers, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size
    sorted_bn = torch.sort(bn_weights)[0]

    return sorted_bn

def obtain_bn_threshold(sorted_bn, percentage):
    thre_index = int(len(sorted_bn) * percentage)
    thre = sorted_bn[thre_index]
    return thre


def obtain_filters_mask(model, prune_idx, thre):
    pruned = 0
    bn_count = 0
    total = 0
    num_filters = []
    pruned_filters = []
    filters_mask = []
    pruned_maskers = []

    for idx, module in enumerate(list(model.named_modules())):
        if isinstance(module[1], nn.BatchNorm2d):
            # print(idx)
            if idx in prune_idx:
                mask = obtain_bn_mask(module[1], thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module[1].weight.data.abs().max()
                    mask = obtain_bn_mask(module[1], max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')

                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
            else:

                mask = np.ones(module[1].weight.data.shape)
                remain = mask.shape[0]


            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return pruned_filters[1:], pruned_maskers

def init_weights_from_loose_model(compact_model, loose_model, Conv_idx, CBLidx2mask):

    for i,idx in enumerate(Conv_idx):
        if i >0:
            out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
            in_channel_idx = np.argwhere(CBLidx2mask[Conv_idx[i-1]])[:, 0].tolist()

            compact_bn, loose_bn         = list(compact_model.modules())[idx+1], list(loose_model.modules())[idx+1]
            compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
            compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
            compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

            #input mask is

            compact_conv, loose_conv = list(compact_model.modules())[idx], list(loose_model.modules())[idx]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
            print(idx)