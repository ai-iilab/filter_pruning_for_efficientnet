import torch


def hard_prune_network(network, args):
    if args.network == 'vgg':
        network = hard_prune_vgg(network, args)
    elif args.network == 'resnet':
        network = hard_prune_resnet(network, args)
        # network = hard_prune_resnet_2(network, args)
    return network


def hard_prune_vgg(network, args):
    if network is None:
        return

    network = hard_prune_vgg_step(network, args.prune_layers, args.prune_channels, args.independent_prune_flag)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_vgg_step(network, prune_layers, prune_channels, independent_prune_flag):
    count = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'

    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            if dim == 1:
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1

            if 'conv%d' % conv_count in prune_layers:
                channel_index = get_channel_index(network.features[i].weight.data, prune_channels[count], residue)
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1
        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    if 'conv13' in prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network


def hard_prune_resnet(network, args):
    if network is None:
        return

    # channel_index = get_channel_index(network.conv_1_3x3.weight.data, int(round(network.conv_1_3x3.out_channels * args.prune_rate[0])))
    # network.conv_1_3x3 = get_new_conv(network.conv_1_3x3, 0, channel_index, args.independent_prune_flag)
    # network.bn_1 = get_new_norm(network.bn_1, channel_index)

    for block in network.stage_1:
        block, _ = hard_prune_block(block, [], args.prune_rate[0], args.independent_prune_flag)
    for block in network.stage_2:
        block, _ = hard_prune_block(block, [], args.prune_rate[1], args.independent_prune_flag)
    for block in network.stage_3:
        block, _ = hard_prune_block(block, [], args.prune_rate[2], args.independent_prune_flag)

    # network.classifier = get_new_linear(network.classifier, channel_index)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_block(block, channel_index, prune_rate, independent_prune_flag):
    # block.conv_a, greedy_residue = get_new_conv(block.conv_a, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_a.weight.data, int(round(block.conv_a.out_channels * prune_rate)), residue=None)
    block.conv_a = get_new_conv(block.conv_a, 0, channel_index, independent_prune_flag)
    block.bn_a = get_new_norm(block.bn_a, channel_index)

    block.conv_b, greedy_residue = get_new_conv(block.conv_b, 1, channel_index, independent_prune_flag)
    # channel_index = get_channel_index(block.conv_b.weight.data, int(round(block.conv_b.out_channels * prune_rate)), greedy_residue)
    # block.conv_b = get_new_conv(block.conv_b, 0, channel_index, independent_prune_flag)
    # block.bn_b = get_new_norm(block.bn_b, channel_index)

    return block, []


def hard_prune_resnet_2(network, args):
    if network is None:
        return

    channel_index = get_channel_index(network.conv_1_3x3.weight.data, int(round(network.conv_1_3x3.out_channels * args.prune_rate[0])))
    network.conv_1_3x3 = get_new_conv(network.conv_1_3x3, 0, channel_index, args.independent_prune_flag)
    network.bn_1 = get_new_norm(network.bn_1, channel_index)

    for block in network.stage_1:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[0], args.independent_prune_flag)
    for block in network.stage_2:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[1], args.independent_prune_flag)
    for block in network.stage_3:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[2], args.independent_prune_flag)

    network.classifier = get_new_linear(network.classifier, channel_index)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_block_2(block, channel_index, prune_rate, independent_prune_flag):
    block.conv_a, greedy_residue = get_new_conv(block.conv_a, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_a.weight.data, int(round(block.conv_a.out_channels * prune_rate)), greedy_residue)
    block.conv_a = get_new_conv(block.conv_a, 0, channel_index, independent_prune_flag)
    block.bn_a = get_new_norm(block.bn_a, channel_index)

    block.conv_b, greedy_residue = get_new_conv(block.conv_b, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_b.weight.data, int(round(block.conv_b.out_channels * prune_rate)), greedy_residue)
    block.conv_b = get_new_conv(block.conv_b, 0, channel_index, independent_prune_flag)
    block.bn_b = get_new_norm(block.bn_b, channel_index)

    return block, channel_index


def get_channel_index(kernel, num_elimination, residue=None):
    # get cadidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    _, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


