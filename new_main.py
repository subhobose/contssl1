#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.utils.tensorboard import SummaryWriter
import submitit
import hydra.utils as hydra_utils
import hydra
from pathlib import Path
import random
import logging
import os
import copy
import warnings
import torch

warnings.filterwarnings("ignore")
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def __init__(self, origargs):
        """TODO: Docstring for __call__.

        :args: TODO
        :returns: TODO

        """
        import importlib
        main_worker = importlib.import_module(
            origargs.model.main_worker).main_worker
        from new_worker_lincls import main_worker as main_lincls_worker
        import numpy as np
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        args = copy.deepcopy(origargs)
        np.set_printoptions(precision=3)
        if args.environment.seed == 0:
            args.environment.seed = None
        assert args.environment.world_size <= 1, print("Only single node training implemented.")

        if args.logging.log_tb:
            os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name),
                        exist_ok=True)
            writer = SummaryWriter(
                os.path.join(args.logging.tb_dir, args.logging.name))
            writer.add_text('exp_dir', os.getcwd())

        ngpus_per_node = torch.cuda.device_count()
        print(ngpus_per_node)
        lincls_key = list(args.lincls.keys())[0]
        # Simply call main_worker function
        lincls_args = copy.deepcopy(args)
        main_worker(args.environment.gpu, ngpus_per_node, args)
        #print(args.lincls[lincls_key].eval_params.resume_epoch)
        #if args.lincls[lincls_key].eval_params.resume_epoch >= 0:
           # main_lincls_worker(args.environment.gpu, ngpus_per_node, args)

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            Worker(), *args, **kwargs)  # submits to requeuing


def jobs_running():
    return [jobname for jobname in
            os.popen('squeue -o %j').read().split("\n")]


@hydra.main(config_path='./configs/simsiam/', config_name='config')
def main(args):
    update_pythonpath_relative_hydra()
    args.logging.ckpt_dir = hydra_utils.to_absolute_path(args.logging.ckpt_dir)
    args.logging.tb_dir = hydra_utils.to_absolute_path(args.logging.tb_dir)
    args.data.train_filelist = hydra_utils.to_absolute_path(
        args.data.train_filelist)
    args.data.val_filelist = hydra_utils.to_absolute_path(
        args.data.val_filelist)
    for i in list(args.lincls.keys()):
        args.lincls[i].data.train_filelist = hydra_utils.to_absolute_path(
            args.lincls[i].data.train_filelist)
        args.lincls[i].data.val_filelist = hydra_utils.to_absolute_path(
            args.lincls[i].data.val_filelist)

    # If job is running, ignore
    jobnames = jobs_running()
    if args.logging.name.replace('.',
                                 '_').replace('-', '_') in jobnames and args.environment.slurm:
        print('Skipping {} because already in queue'.format(args.logging.name))
        return

    # If model is trained, ignore
    ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                              'checkpoint_{:04d}.pth')
    if os.path.exists(ckpt_fname.format(args.optim.epochs)):
        all_exist = True
        model_name = ckpt_fname.format(args.optim.epochs)
        for dataset_i in list(args.lincls.keys()):
            lincls_fname = model_name + \
                args.lincls[dataset_i].eval_params.suffix + '.lincls'
            if not os.path.exists(lincls_fname):
                all_exist = False
        if all_exist:
            print('Skipping {}'.format(args.logging.name))
            return

    
    job = Worker(args)

if __name__ == '__main__':
    main()
