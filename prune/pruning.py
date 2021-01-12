from prune import prune_config as config
from prune.utils import *
from src.tester import ModelInference


def normal_prune(model_path, save_cfg_path, save_model_path):
    model_name = config.model_name
    classes = config.classes
    num_classes = len(classes)
    thresh = config.thresh
    model = ModelInference(num_classes, model_name, model_path, cfg='default').CNN_model
    prune_idx, ignore_id, all_conv_layer = parse_module_defs(model)
    sorted_bn = sort_bn(model, prune_idx)
    threshold = obtain_bn_threshold(sorted_bn, thresh / 100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)
    Conv_id = [conv - 1 for conv in prune_idx]
    CBLidx2mask = {idx: mask.astype('float32') for idx, mask in zip(Conv_id, pruned_maskers)}

    prune_model = ModelInference(num_classes, model_name, None, cfg='default').CNN_model

    init_weights_from_loose_model(all_conv_layer, Conv_id, prune_model, model, CBLidx2mask)

    cfg_file = open(save_cfg_path, 'a+')
    cfg = [[list(layer[1].weight.shape[1::-1]) for layer in list(prune_model.named_modules()) if 'conv' in layer[0]]][
          1:]
    print(cfg, file=cfg_file)
    torch.save(prune_model.state_dict(), save_model_path)


def test_prune_model(test_model_path, img_path):
    model = ModelInference(config.num_classes, config.model_name, test_model_path, cfg='prune').CNN_model
    img = cv2.imread(img_path)
    res = predict(model, img)
    print(res)


if __name__ == '__main__':
    normal_prune(model_path=config.model_path, save_cfg_path=config.pruned_cfg_file,
                 save_model_path=config.pruned_model_file)
    test_prune_model(config.test_model_path, config.img_path)
