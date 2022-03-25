# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import sys
import time
from typing import Dict, List
from typing import (Optional, Union)

import matplotlib
import torch
from medutils.medutils import count_parameters
from mlflow import log_metric, log_param, start_run, end_run, log_params
import mlflow
sys.path.append("../..")

import argparse
# import streamlit as st
matplotlib.use('Agg')
from sharednet.modules.set_args import get_args
from sharednet.modules.tool import record_1st, record_2nd
from sharednet.modules.nets import get_net
from sharednet.modules.loss import get_loss
from sharednet.modules.path import Mypath, MypathDataDir
from sharednet.modules.evaluator import get_evaluator
from sharednet.modules.dataset import DataAll

from argparse import Namespace

LogType = Optional[Union[int, float, str]]  # a global type to store immutable variables saved to log files

def get_out_chn(task_name):
    if task_name=="lobe_all":
        out_chn = 6
    elif task_name=="AV_all":
        out_chn = 3
    elif 'all' in task_name and '-' in task_name:  # multi-class dataset' segmentation
        raise Exception(f"The model_names is {task_name} but we have not set the output channel number for multi-class "
                        f"dataset' segmentation. Please reset the model_names.")

    else:
        out_chn = 2
    return out_chn


def mt_netnames(net_names: str) -> List[str]:
    """Get net names from arguments.

    Define the Model, use dash to separate multi net names, do not use ',' to separate it, because ',' can lead to
    unknown error during parse arguments

    Args:
        myargs:

    Returns:
        A list of net names

    """
    #
    net_names = net_names.split('-')
    net_names = [i.lstrip() for i in net_names]  # remove backspace before each net name
    print('net names: ', net_names)

    return net_names


def task_of_model(model_name):
    for task in ['lobe', 'vessel', 'AV', 'liver', 'pancreas']:
        if task in model_name:
            return task


def all_loaders(model_name):
    data = DataAll(model_name)
    # if model_name == 'lobe_ll':
    #     data = DataLobeLL()
    # elif model_name == 'lobe_lu':
    #     data = DataLobeLU()
    # elif model_name == 'lobe_ru':
    #     data = DataLobeRU()
    # elif model_name == 'lobe_rm':
    #     data = DataLobeRM()
    # elif model_name == 'lobe_rl':
    #     data = DataLobeRL()
    # elif model_name == 'vessel':
    #     data = DataVessel()
    # elif model_name == 'AV_Artery':
    #     data = DataAVArtery()
    # elif model_name == 'AV_Vein':
    #     data = DataAVVein()
    # elif model_name == 'liver':
    #     data = DataLiver()
    # elif model_name == 'pancreas':
    #     data = DataPancreas()
    # else:
    #     raise Exception(f"Wrong task name {model_name}")

    tr_dl, vd_dl, ts_dl = data.load(cond_flag=args.cond_flag,
                                    same_mask_value=args.same_mask_value,
                                    pps=args.pps,
                                    batch_size=args.batch_size)
    return data, tr_dl, vd_dl, ts_dl

def loop_dl(dl):
    # keys = ("image", "mask", "cond")
    dl_endless = iter(dl)
    while True:
        try:
            out = next(dl_endless)
        except:
            dl_endless = iter(dl)
            out = next(dl_endless)
        yield out

    # while True:
    #     for data in dl:
    #         x_pps = data[keys[0]]
    #         y_pps = data[keys[1]]
    #         cond_pps = data[keys[2]]
    #         for x, y, cond in zip(x_pps, y_pps, cond_pps):
    #             # print("start put by single thread, not fluent thread")
    #             x = x[None, ...]
    #             y = y[None, ...]
    #             cond = cond[None, ...]
    #             data_single = (x, y, cond)
    #             yield data_single

class Task:
    def __init__(self, model_name, net, out_chn, opt, loss_fun):
        self.model_name = model_name
        task = task_of_model(self.model_name)
        self.data_dir = MypathDataDir(task).data_dir
        self.net = net
        self.mypath = Mypath(args.id, check_id_dir=False)
        self.out_chn = out_chn
        self.opt = opt
        self.loss_fun = loss_fun
        self.device = torch.device("cuda")
        self.scaler = torch.cuda.amp.GradScaler()
        data, self.tr_dl, self.vd_dl, self.ts_dl = all_loaders(self.model_name)
        self.tr_dl_endless = loop_dl(self.tr_dl)  # loop training dataset


        self.eval_vd = get_evaluator(net, self.vd_dl, self.mypath, data.psz_xy, data.psz_z, args.batch_size, 'valid',
                                        out_chn)
        self.eval_ts = get_evaluator(net, self.ts_dl, self.mypath, data.psz_xy, data.psz_z, args.batch_size, 'test',
                                       out_chn)
        self.accumulate_loss = 0

    def step(self, step_id):
        # print(f"start a step for {self.model_name}")
        t1 = time.time()
        image, mask, cond = (next(self.tr_dl_endless).get(key) for key in ('image', 'mask', 'cond'))
        t2 = time.time()
        image, mask, cond = image.to(self.device), mask.to(self.device), cond.to(self.device)
        t3 = time.time()

        self.opt.zero_grad()
        if 1:
            with torch.cuda.amp.autocast():
                pred = self.net(image,cond)
                loss = self.loss_fun(pred, mask)
            t4 = time.time()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            pred = self.net(image, cond)
            loss = self.loss_fun(pred, mask)
            t4 = time.time()
            loss.backward()
            self.opt.step()
        t5 = time.time()

        self.accumulate_loss += loss.item()

        if step_id % 200 == 0:
            period = 1 if step_id==0 else 200  # the first accumulate_loss is the first loss
            log_metric(self.model_name + '_TrainBatchLossIn200Steps', self.accumulate_loss/period, step_id)
            self.accumulate_loss = 0
        print(f" {self.model_name} loss: {loss:.3f}, "
              f"load batch cost: {t2-t1:.1f}, "
              f"forward costs: {t4-t3:.1f}, "
              f"backward costs: {t5-t4:.1f}; ", end='' )

    def do_validation_if_need(self, step_id, steps, valid_period=2000):

        if step_id % valid_period == 0 or step_id == steps - 1:
            print(f"start a valid for {self.model_name}")
            self.eval_vd.run()
        if step_id == steps - 1:
            print(f"start a test for {self.model_name}")
            self.eval_ts.run()


def task_dt(model_names, net, out_chn, opt, loss_fun):
    ta_dict: Dict[str, Task] = {}
    for model_name in model_names:
        ta = Task(model_name, net, out_chn, opt, loss_fun)
        ta_dict[model_name] = ta
    return ta_dict


def run(args: Namespace):
    """The main body of the training process.

    Args:
        args: argparse instance

    """
    out_chn = get_out_chn(args.model_names)
    log_param('out_chn', out_chn)

    net = get_net(args.cond_flag, args.cond_method, args.cond_pos, out_chn, args.base)
    net_parameters = count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))
    log_param('net_parameters_M', net_parameters)
    net = net.to(torch.device("cuda"))

    loss_fun = get_loss(loss=args.loss)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # weight decay is L2 weight norm

    model_names: List[str] = mt_netnames(args.model_names)
    ta_dict = task_dt(model_names, net, out_chn, opt, loss_fun)

    for step_id in range(args.steps):
        print(f'\nstep number: {step_id}, ', end='' )
        for model_name, ta in ta_dict.items():
            ta.step(step_id)
            ta.do_validation_if_need(step_id, args.steps)

    print('Finish all training/validation/testing + metrics!')


if __name__ == "__main__":
    args = get_args()
    log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

    id, log_dict = record_1st(args)  # write super parameters from set_args.py to record file.

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        log_params(log_dict)
        args.id = id  # do not need to pass id seperately to the latter function
        run(args)
        record_2nd(log_dict=log_dict, args=args)  # write more parameters & metrics to record file.

