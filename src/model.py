import itertools as it
import torch.optim as optim

import efficientnet_pytorch


def get_model(config):
    return efficientnet_pytorch.EfficientNet.from_pretrained(
        config.network.name,
        num_classes=config.network.num_classes,
    )


def add_weight_decay(model, weight_decay=1e-4, skip_list=("bn",)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_opt(args, model, criterion):
    # Scale learning rate based on global batch size
    if args.opt.opt == "AdamW":
        opt = optim.AdamW(
            model.parameters(),
            lr=args.opt.lr,
            weight_decay=args.opt.weight_decay
        )
    elif args.opt.opt == "Adam":
        opt = optim.Adam(
            model.parameters(),
            lr=args.opt.lr,
            weight_decay=args.opt.weight_decay
        )
    else:
        raise

    return opt
