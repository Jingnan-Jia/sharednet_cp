# -*- coding: utf-8 -*-
# @Time    : 7/5/21 7:24 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import os
from typing import Union
from pathlib import Path


def get_data_dir(task):
    if 'lobe' in task:
        directory = "lobe/GLUCOLD_fine_niigz"
    elif 'av' in task:
        raise Exception(f'data_dir Not implemented Error for task {task}')
    elif 'vessel' in task:
        raise Exception(f'data_dir Not implemented Error for task {task}')
    elif 'liver' in task:
        directory = "liver"
    elif 'pancreas' in task:
        directory = "pancreas"
    else:
        raise Exception(f"task {task} is not correct!")
    return "/data/jjia/mt/data/" + directory


class MypathBase:
    """Fixed pathes"""
    def __init__(self):
        self.result_dir = Path('results')
        self.record_fpath = self.result_dir.joinpath('records.csv')
        self.log_dir = self.result_dir.joinpath('logs')
        self.ex_dir = self.result_dir.joinpath('experiments')


class Mypath(MypathBase):
    """
    Common path values are initialized.
    """
    # data_dir = Path("/data/jjia/mt/data/lobe/GLUCOLD_fine_niigz")
    # result_dir = Path('results')
    # record_fpath = result_dir.joinpath('records.csv')
    # log_dir = result_dir.joinpath('logs')
    # ex_dir = result_dir.joinpath('experiments')

    def __init__(self, id: Union[int, str], check_id_dir: bool = False):
        super().__init__()
        if isinstance(id, (int, float)):
            self.id = str(int(id))
        else:
            self.id = id
        self.id_dir = self.ex_dir.joinpath(str(id))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if self.id_dir.is_dir():  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for directory in [self.result_dir, self.log_dir, self.ex_dir, self.id_dir]:
            if not directory.is_dir():
                directory.mkdir(parents=True)
                print('successfully create directory:', directory.absolute())

        self.loss_fpath = self.id_dir.joinpath('loss.csv')

        self.model_fpath = self.id_dir.joinpath('model.pt')
        self.model_wt_structure_fpath = self.id_dir.joinpath('model_wt_structure.pt')
    def metrics_fpath(self, mode):
        metrics_fpath = self.id_dir.joinpath(mode + '_metrics.csv')
        return metrics_fpath

class MypathDataDir(MypathBase):
    """Dataset directory is implemented here"""
    def __init__(self, task):
        super().__init__()
        self.data_dir = get_data_dir(task)
