from src.tester import ModelInference
from prune.utils import *


def normal_prune(model_path, cfg_file):
    # model_path = '/media/hkuit164/WD20EJRX/CNN_classification/weight/test/default/default_resnet18_2cls_best.pth'
    name = "resnet18"
    classes = ['drown','stand']
    num_classes = len(classes)
    thresh = 60
    model = ModelInference(num_classes, name, model_path,cfg = []).CNN_model
    print([list(layer[1].weight.shape[0:2]) for layer in list(model.named_modules()) if 'conv' in layer[0]])

    prune_idx,ignore_id ,all_conv = parse_module_defs2(model)
    sorted_bn = sort_bn(model, prune_idx)
    threshold = obtain_bn_threshold(sorted_bn, thresh/100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)
    Conv_id = [conv - 1 for conv in prune_idx]
    CBLidx2mask = {idx: mask.astype('float32') for idx, mask in zip(Conv_id, pruned_maskers)}

    new_model = ModelInference(num_classes, name, None,cfg = []).CNN_model

    for i, idx in enumerate(all_conv):
        if i > 0:
            if idx in Conv_id:
                out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
                #last conv index
                last_conv_index = all_conv[all_conv.index(idx) - 1]
                if last_conv_index in Conv_id:
                    in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
                else:
                    in_channel_idx = list(range(list(new_model.named_modules())[all_conv[all_conv.index(idx) - 1]][1].out_channels))
            else:
                # we should make conv2's input equal to the output channel of conv1.
                out_channel_idx = list(range(list(new_model.named_modules())[idx][1].out_channels))
                index = all_conv[all_conv.index(idx) - 1]
                in_channel_idx = np.argwhere(CBLidx2mask[index])[:, 0].tolist()

            compact_bn, loose_bn = list(new_model.named_modules())[idx + 1][1], list(model.named_modules())[idx + 1][1]
            compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
            compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
            compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

            compact_conv, loose_conv = list(new_model.named_modules())[idx][1], list(model.named_modules())[idx][1]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    cfg_file = open(cfg_file,'a+')
    cfg = [list(layer[1].weight.shape[1::-1]) for layer in list(new_model.named_modules()) if 'conv' in layer[0]]
    print(cfg,file=cfg_file)
    print([list(layer[1].weight.shape[1::-1]) for layer in list(new_model.named_modules()) if 'conv' in layer[0]])
    torch.save(new_model.state_dict(),'./new_model.pth')

if __name__ == '__main__':
    normal_prune(model_path='/media/hkuit164/WD20EJRX/CNN_classification/weight/test/default/default_resnet18_2cls_best.pth',cfg_file='./cfg.txt')