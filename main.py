import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
# from torch.utils.data import *
import torch.utils.data
import torchvision
import numpy as np
import torch.autograd
from torch import nn
from torch_flops import TorchFLOPsByFX

best_prec1 = 0
zeroinput = 0
oneinput = 1


class Attack(nn.Module):
    def __init__(self, args, model, criterion, batch_num):
        super(Attack, self).__init__()
        self.model = model
        self.criterion = criterion
        self.batch_num = batch_num

        # [bs, new_length*3, H, W]
        self.frame_indices = [(k * (args.motion_len + args.new_length) + j) * 3 + i for k in range(0, args.num_segments)
                              for j in
                              range(args.motion_len, args.motion_len + args.new_length) for i in range(0, 3)]
        self.frame_indices = torch.tensor(self.frame_indices).cuda()

    # input [bs, 3 * num_segment * new_length, 224 ,224]
    def forward(self, addition_input, target):
        temp = 0
        with torch.no_grad():
            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=self.frame_indices)

            frameval = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda()  # [farther ... closer]
            frameval = frameval.repeat(args.batch_size, 1)
            stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
            motioninput = input.clone() * (1 - args.motion_val)
            input_roll = addition_input.clone()
            for step in range(1, args.motion_len + 1):
                input_roll = torch.roll(input_roll, 3, 1)
                motioninput += torch.index_select(input_roll, dim=1, index=self.frame_indices) * stdval[:,
                                                                                                 args.motion_len - step].view(
                    args.batch_size, 1, 1, 1)  # dim=1 frames*3的维度 移动3位（3通道）
            motionpertu = motioninput - input.clone()

        input.requires_grad = True
        output = self.model(input)
        loss = self.criterion(output, target).cuda()
        loss.backward()
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))
        # batch_time.update(time.time() - end)
        end = time.time()
        '''if i % args.print_freq == 0:
            logger.info(
                ('Test: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))'''

        attgrad = input.grad.data * 1e6
        self.model.zero_grad()
        pertu = motionpertu

        if args.attack_type == "2":
            temp = torch.mul(torch.sign(attgrad), motionpertu)
            temp = torch.clamp(input=temp, min=0)
            # test1 hmdb51 mean ~ -0.00034   std ~ 77.041
            # test2 hmdb51 mean ~ -0.00036   std ~ 78.004
            if "hmdb51" in args.dataset:
                grad_mean = 0
                grad_std = 77.5
            else:
                return
            attgrad = (attgrad - grad_mean) / grad_std
            attgrad = torch.abs(torch.clamp(attgrad, min=-1, max=1))
            pertu = torch.where(temp > 0, motionpertu, motionpertu * (1 - attgrad))

        elif args.attack_type == "3":
            self.model.zero_grad()
            adv_input = (input + motionpertu * args.step_start).clone().detach()
            adv_input.requires_grad = True
            output = self.model(adv_input)
            loss = self.criterion(output, target).cuda()
            loss.backward()
            attgrad = adv_input.grad.data * 1e6
            attgrad.requires_grad = False
            self.model.zero_grad()

            for step in range(args.step_times):
                temp = torch.mul(torch.sign(attgrad), motionpertu)
                temp = torch.clamp(input=temp, min=0)
                pertu = torch.where(temp > 0, motionpertu * args.step_weight, temp)

                adv_input = (adv_input + pertu).detach()
                adv_input = torch.clamp(adv_input, min=zeroinput, max=oneinput)
                self.model.zero_grad()
                adv_input.requires_grad = True
                output = self.model(adv_input)
                loss = self.criterion(output, target).cuda()
                loss.backward()
                attgrad = adv_input.grad.data * 1e6
                adv_input.grad.data.zero_()
                self.model.zero_grad()

            pertu = adv_input - input

        elif "4" in args.attack_type:
            frameval = torch.ones([args.batch_size, args.motion_len]).cuda()
            best_loss = 0
            best_adv = input.clone().detach()
            best_adv.requires_grad = False
            for e in range(args.frame_times):
                self.model.zero_grad()
                frameval = frameval.clone().detach()
                frameval.requires_grad = True
                if args.attack_type == "4":
                    stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                    motioninput = input.clone() * (1 - args.motion_val)
                    input_roll = addition_input.clone()

                    for step in range(1, args.motion_len + 1):
                        input_roll = torch.roll(input_roll, 3, 1)
                        motioninput = motioninput + torch.index_select(input_roll, dim=1,
                                                                       index=self.frame_indices) * stdval[:,
                                                                                                   args.motion_len - step].view(
                            args.batch_size, 1, 1, 1)  # dim=1 frames*3的维度 移动3位（3通道）
                    motionpertu = motioninput - input.clone()
                    adv_input = input + motionpertu
                    output = self.model(adv_input)
                    loss = self.criterion(output, target).cuda()
                    loss.backward()

                    frameval = frameval + torch.sign(frameval.grad.data) * args.frame_weight
                    frameval = torch.clamp(frameval, min=0)

                    self.model.zero_grad()
                    # if self.batch_num % 50 == 0:
                    #     print(loss)

                elif "3" in args.attack_type:
                    self.model.zero_grad()
                    adv_input = (input + motionpertu * args.step_start).clone().detach()
                    adv_input.requires_grad = True
                    output = self.model(adv_input)
                    loss = self.criterion(output, target).cuda()
                    loss.backward()
                    attgrad = adv_input.grad.data * 1e6
                    attgrad.requires_grad = False
                    self.model.zero_grad()
                    # 先得到起步对应梯度

                    framevalgrad = torch.zeros(frameval.size())

                    for it in range(args.step_times):
                        frameval = frameval.clone().detach()
                        frameval.requires_grad = True
                        stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                        motioninput = input.clone() * (1 - args.motion_val)
                        input_roll = addition_input.clone().detach()
                        for step in range(1, args.motion_len + 1):
                            input_roll = torch.roll(input_roll, 3, 1)
                            motioninput = motioninput + torch.index_select(input_roll, dim=1,
                                                                           index=self.frame_indices) * stdval[:,
                                                                                                       args.motion_len - step].view(
                                args.batch_size, 1, 1, 1)
                        motionpertu = motioninput - input.clone().detach()

                        temp = torch.mul(torch.sign(attgrad), motionpertu)
                        temp = torch.clamp(input=temp, min=0)
                        pertu = torch.where(temp > 0, motionpertu * args.step_weight, 0)

                        adv_input = adv_input.clone().detach() + pertu
                        adv_input = torch.clamp(adv_input, min=zeroinput, max=oneinput)
                        self.model.zero_grad()
                        output = self.model(adv_input)
                        adv_input.retain_grad()

                        loss = self.criterion(output, target).cuda()
                        loss.backward()
                        # if self.batch_num % 50 == 0:
                        #     print(loss)

                        attgrad = adv_input.grad.data * 1e6
                        framevalgrad = torch.sign(frameval.grad.data) * args.frame_weight
                        frameval.grad.zero_()
                        adv_input.grad.zero_()
                        self.model.zero_grad()
                    # if self.batch_num % 50 == 0:
                    #     print(e, end="========\n")
                    if loss > best_loss:
                        best_adv = adv_input
                        best_loss = loss
                        # if self.batch_num % 50 == 0:
                        #     print(loss, end=" best-----------------\n")
                    frameval = frameval + framevalgrad / args.step_times
                    frameval = torch.clamp(frameval, min=0)
                    self.model.zero_grad()
                    # if self.batch_num % 50 == 0:
                    #    print(loss)
                    #    print("===========")
            pertu = best_adv - input

        return input, pertu


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == "hmdb51":
        args.tune_from = "F:\\workspace\\TDN_checkpoint\\best-hmdb51\\kin-101.tar"
    elif args.dataset == "ucf101":
        args.tune_from = "F:\\workspace\\TDN_checkpoint\\best-ucf101\\kin-101.tar"
    else:
        return

    if args.step_weight * args.step_times - 2 * (1 - args.step_start) > 1e-5:
        print("error step weight")
        return
    print('dataset: ' + str(args.dataset))
    print('attack type: ' + str(args.attack_type))
    if not args.attack_type == "0":
        print('motion len: ' + str(args.motion_len))
        print('motion val: ' + str(args.motion_val))
    else:
        args.motion_len = 0
    print("=============================")
    if "4" in args.attack_type:
        args.print_freq = 10
        print('frame times: ' + str(args.frame_times))
        print('frame weight: ' + str(args.frame_weight))
    print("=============================")
    if "3" in args.attack_type:
        print('step times: ' + str(args.step_times))
        print('step weight: ' + str(args.step_weight))
        print('step start: ' + str(args.step_start))
    print("=============================")

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality,
                                                                                                      args.root_dataset)
    full_arch_name = args.arch
    if not args.attack_type == "0":
        args.store_name += "att" + args.attack_type + "-v" + str(args.motion_val) + "="
        if "4" in args.attack_type:
            args.store_name += str(args.frame_times) + "-" + str(args.frame_weight).replace(".", "")[:2] + "="
        if "3" in args.attack_type:
            args.store_name += str(args.step_start).replace(".", "")[:2] + "-" + str(
                args.step_start + args.step_times * args.step_weight).replace(".", "")[:2] + "="
    args.store_name = '_'.join(
        [args.store_name, str(args.attack_type), '_TDN_', args.dataset, args.modality, full_arch_name,
         args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_model, args.store_name, "log"),
                          distributed_rank=0,
                          name=f'TDN')
    logger.info('storing name: ' + args.store_name)

    model = TSN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    for group in policies:
        logger.info(
            ('[TDN-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
                                                  Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                  ToTorchFormatTensor(
                                                      div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                  normalize, ]),
        dense_sample=args.dense_sample)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')
    # print("\n")
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, shuffle=True, drop_last=True)

    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize, ]),
        dense_sample=args.dense_sample,
        new_length=args.new_length, addition_length=args.motion_len)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=None, shuffle=False, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(policies, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = model.cuda()

    global zeroinput, oneinput
    zeroinput = torch.zeros(3, 224, 224).cuda()
    oneinput = torch.ones(3, 224, 224).cuda()
    for z, o, m, s in zip(zeroinput, oneinput, model.input_mean, model.input_std):
        z.sub_(m).div_(s)
        o.sub_(m).div_(s)
    zeroinput = zeroinput.repeat(args.batch_size, 40, 1, 1)
    oneinput = oneinput.repeat(args.batch_size, 40, 1, 1)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    print(args.tune_from)
    if args.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        if (args.dataset == 'hmdb51' or args.dataset == 'ucf101') and (
                'v1' in args.tune_from or 'v2' in args.tune_from):
            sd = {k.replace('module.base_model.', 'base_model.'): v for k, v in sd.items()}
            sd = {k.replace('module.new_fc.', 'new_fc.'): v for k, v in sd.items()}
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                logger.info('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                logger.info('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info('#### Notice: keys loaded but not in models: {}'.format(keys1 - keys2))
        logger.info('#### Notice: keys required but not in pre-models: {}'.format(keys2 - keys1))
        if args.dataset not in args.tune_from:  # new dataset
            logger.info('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    with open(os.path.join(args.root_model, args.store_name, "log", 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_model, args.store_name, "log"))

    if args.evaluate:
        logger.info(("===========evaluate==========="))

        prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)

        is_best = prec1 > best_prec1
        best_prec1 = prec1
        logger.info(("Best Prec@1: '{}'".format(best_prec1)))
        save_epoch = args.start_epoch + 1
        save_checkpoint(
            {
                'epoch': args.start_epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'prec1': prec1,
                'best_prec1': best_prec1,
            }, save_epoch, is_best)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch=epoch,
                                                   logger=logger, scheduler=scheduler)

        tf_writer.add_scalar('loss/train', train_loss, epoch)
        tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
        tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, prec5, val_loss = validate(val_loader, model, criterion, logger, epoch=epoch)

            tf_writer.add_scalar('loss/test', val_loss, epoch)
            tf_writer.add_scalar('acc/test_top1', prec1, epoch)
            tf_writer.add_scalar('acc/test_top5', prec5, epoch)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            logger.info(("Best Prec@1: '{}'".format(best_prec1)))
            tf_writer.flush()
            save_epoch = epoch + 1
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'prec1': prec1,
                    'best_prec1': best_prec1,
                }, save_epoch, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger=None, scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.partialBN(False)
    else:
        model.partialBN(True)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO

    logger.info(('Training Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, logger=None, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_att = AverageMeter()
    top5_att = AverageMeter()
    losses_att = AverageMeter()

    model.eval()

    end = time.time()
    from itertools import islice
    # 470 694
    # for i, (addition_input, target) in enumerate(val_loader):
    for i, (addition_input, target) in enumerate(islice(val_loader, 470, 471), start=470):
    # for i, (addition_input, target) in enumerate(islice(val_loader, 694, 695), start=694):
        if args.attack_type == "0":
            with torch.no_grad():
                target = target.cuda()
                addition_input = addition_input.cuda()
                input = torch.index_select(addition_input, dim=1, index=indices)

                output = model(input)
                loss = criterion(output, target).cuda()
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info(
                        ('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

        else:

            attack = Attack(args, model, criterion, batch_num=i)

            input, pertu = attack(addition_input, target)

            # flops_counter = TorchFLOPsByFX(attack)
            # flops_counter.propagate(addition_input, target)
            # flops_1 = flops_counter.print_total_flops(show=False)
            # print(f"torch_flops: {flops_1} FLOPs")

            target = target.cuda()
            addition_input = addition_input.cuda()

            with torch.no_grad():
                output = model(input + pertu)
                loss = criterion(output, target).cuda()
                prec1_att, prec5_att = accuracy(output.data, target, topk=(1, 5))
                top1_att.update(prec1_att.item(), input.size(0))
                top5_att.update(prec5_att.item(), input.size(0))
                losses_att.update(loss.item(), input.size(0))

                if i % args.print_freq == 0:
                    logger.info(
                        ('Attack Test: [{0}/{1}]\t'
                         'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                         'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                            i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

                saveImg = os.path.join(args.root_model, args.store_name, str(epoch))
                if not os.path.exists(saveImg):
                    os.makedirs(saveImg)
                img_input = back_normalize(input.clone().detach().view(-1, 3, 224, 224), model.input_mean,
                                           model.input_std).cuda()
                img_adv = back_normalize((input.clone().detach() + pertu.clone().detach()).view(-1, 3, 224, 224),
                                         model.input_mean, model.input_std).cuda()
                img_pertu = torch.abs(img_adv - img_input)
                # img_moinput = back_normalize(motioninput.clone().detach().view(-1, 3, 224, 224), model.input_mean, model.input_std).cuda()

                img_input = img_input.to(torch.device('cpu'))
                img_adv = img_adv.to(torch.device('cpu'))
                img_pertu = img_pertu.to(torch.device('cpu'))
                # img_moinput = img_moinput.to(torch.device('cpu'))

                torchvision.utils.save_image(img_input, saveImg + "/input" + str(i) + ".jpg")
                torchvision.utils.save_image(img_adv, saveImg + "/adv" + str(i) + ".jpg")
                torchvision.utils.save_image(img_pertu, saveImg + "/pertu" + str(i) + ".jpg")
                # torchvision.utils.save_image(img_moinput, saveImg + "/moinput" + str(i) + ".jpg")

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))
    if not args.attack_type == "0":
        logger.info(('Attacking Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                     .format(top1=top1_att, top5=top5_att, loss=losses_att)))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


def hookgrad(x):
    print(x)


if __name__ == '__main__':
    main()

# torch_flops: 8798034378302 FLOPs
# 8798.034 GFLOPs

# torch_flops: 144299956262 FLOPs
# 144.299 GFLOPs


'''
python -u main.py hmdb51 --evaluate  --attack_type 34 --frame_times 20 --frame_weight 0.2 --step_start 0.3 --step_times 5 --step_weight 0.28 > test.txt 2>&1
'''

'''
python main.py ucf101 --evaluate --attack_type 3 --step_start 0.3 --step_times 5 --step_weight 0.28 > ./ucf-att/ucf-att-3=m5=03-17.txt 2>&1

'''
