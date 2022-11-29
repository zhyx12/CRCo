#
# ----------------------------------------------
import os
import shutil
import torch
import random
from . import get_root_logger, get_root_writer
from ..loaders import parse_args_for_multiple_datasets
from ..models import parse_args_for_models
from . import deal_with_val_interval
#
import time
from ..hooks import LrLogger, TrainTimeLogger, SaveCheckpoint, SchedulerStep
from mmcv import Config
from ..runner import build_trainer, build_validator
from mmcv.runner import get_dist_info
from .collect_env import collect_env
from .basic_utils import init_random_seed, set_random_seed, build_custom_hooks

Predefined_Control_Keys = ['max_iters', 'log_interval', 'val_interval', 'save_interval', 'max_save_num',
                           'seed', 'cudnn_deterministic', 'pretrained_model', 'checkpoint', 'test_mode',
                           'save_best_model']


def train(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    local_rank, world_size = get_dist_info()
    #
    cfg = Config.fromfile(args.config)
    predefined_keys = ['datasets', 'models', 'control', 'train', 'test']
    old_keys = list(cfg._cfg_dict.keys())
    for key in old_keys:
        if not key in predefined_keys:
            del cfg._cfg_dict[key]
    # check control keys are allowable
    control_cfg = cfg.control
    for key in control_cfg.keys():
        assert key in Predefined_Control_Keys, '{} is not allowed appeared in control keys'.format(key)
    # set default values for control keys
    max_iters = control_cfg.get('max_iters',100000)
    log_interval = control_cfg.get('log_interval', 100)
    val_interval = control_cfg.get('val_interval', 5000)
    save_interval = control_cfg.get('save_interval', 5000)
    max_save_num = control_cfg.get('max_save_num', 1)
    cudnn_deter_flag = control_cfg.get('cudnn_deterministic', False)
    test_mode = control_cfg.get('test_mode', False)
    save_best_model = control_cfg.get('save_best_model', True)
    # create log dir
    run_id = random.randint(1, 100000)
    run_id_tensor = torch.ones((1,), device='cuda:{}'.format(local_rank)) * run_id
    torch.distributed.broadcast(run_id_tensor, src=0)
    run_id = int(run_id_tensor.cpu().item())
    logdir = os.path.join('runs', os.path.basename(args.config)[:-3],
                          'job_' + args.job_id + '_exp_' + str(run_id))
    # save source code, and config files to logdir
    if local_rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        #
        shutil.copy(args.config, logdir)  #
        shutil.copytree('./{}'.format(args.source_code_name), os.path.join(logdir, 'source_code'))
        #
        cfg_save_path = os.path.join(logdir, 'config.py')
        cfg.dump(cfg_save_path)
    # create logger and tensorboard writer
    tb_writer = get_root_writer(log_dir=logdir)
    timestamp = time.strftime('runs_%Y_%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(logdir, f'rank_{local_rank}_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=args.log_level)
    logger.info('log dir is {}'.format(logdir))
    logger.info('Let the games begin')
    logger.info('Experiment identifier is {}'.format(args.job_id))
    # log env information
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # Setup random seeds, and cudnn_deterministic mode
    seed = control_cfg.get('seed', None)
    random_seed = init_random_seed(seed)
    logger.info(f'Set random random_seed to {random_seed}, '
                f'deterministic: {cudnn_deter_flag}')
    set_random_seed(random_seed, deterministic=cudnn_deter_flag)
    #
    # build dataloader
    train_loaders, test_loaders = parse_args_for_multiple_datasets(cfg['datasets'],
                                                                   random_seed=random_seed, data_root=args.data_root)
    #
    # build model and corresponding optimizer, scheduler
    model_related_results = parse_args_for_models(cfg['models'])
    model_dict, optimizer_dict, scheduler_dict = model_related_results
    #
    # gather trainer
    logger.info('Trainer class is {}'.format(args.trainer))
    basic_training_parameters = {
        'local_rank': args.local_rank,
        'model_dict': model_dict,
        'optimizer_dict': optimizer_dict,
        'scheduler_dict': scheduler_dict,
        'train_loaders': train_loaders,
        'logdir': logdir,
        'log_interval': log_interval,
    }
    training_args = cfg.train
    training_hook_args = training_args.pop('custom_hooks', None)
    training_args.update({
        'type': args.trainer,
        'basic_parameters': basic_training_parameters,
    })
    # build trainer
    trainer = build_trainer(training_args)
    trained_iteration = 0
    #
    # load pretrained weights
    pretrained_model = control_cfg.get('pretrained_model', None)
    checkpoint_file = control_cfg.get('checkpoint', None)
    if pretrained_model is not None:
        if '~' in pretrained_model:
            pretrained_model = os.path.expanduser(pretrained_model)
        assert os.path.isfile(pretrained_model), '{} is not a weight file'.format(pretrained_model)
        logger.info('Load pretrained model in {}'.format(pretrained_model))
        trainer.load_pretrained_model(pretrained_model)
    # resume training from checkpoint
    if checkpoint_file is not None:
        if '~' in checkpoint_file:
            checkpoint_file = os.path.expanduser(checkpoint_file)
        trainer.resume_training(checkpoint_file)
        trained_iteration = trainer.get_trained_iteration_from_scheduler()
    #
    # build validator
    test_args = cfg.test
    broadcast_bn_buffer = test_args.pop('broadcast_bn_buffer', True)
    basic_validation_parameters = {
        'local_rank': args.local_rank,
        'model_dict': model_dict,
        'test_loaders': test_loaders,
        'logdir': logdir,
        'broadcast_bn_buffer': broadcast_bn_buffer,
        'trainer': trainer,
    }
    test_hook_args = test_args.pop('custom_hooks', None)
    test_args.update(
        {
            'type': args.validator,
            'basic_parameters': basic_validation_parameters,
        }
    )
    # build evaluator
    validator = build_validator(test_args)
    # build custom validator hooks
    build_custom_hooks(test_hook_args, validator)
    # test mode: only conduct test process
    if test_mode:
        validator(trainer.iteration)
        exit(0)
    ########################################
    # register training hooks

    updater_iter = control_cfg.get('update_iter', 1)
    train_time_recoder = TrainTimeLogger(log_interval)
    trainer.register_hook(train_time_recoder)
    scheduler_step = SchedulerStep(updater_iter)
    lr_recoder = LrLogger(log_interval)
    trainer.register_hook(lr_recoder, priority='HIGH')
    trainer.register_hook(scheduler_step, priority='VERY_LOW')
    save_model_hook = SaveCheckpoint(max_save_num=max_save_num, save_interval=save_interval)
    trainer.register_hook(save_model_hook,
                          priority='LOWEST')  # save model after scheduler step to get the right iteration number
    ########################################
    # build custom training hooks
    build_custom_hooks(training_hook_args, trainer)
    # deal with val_interval
    val_point_list = deal_with_val_interval(val_interval, max_iters=max_iters,
                                            trained_iteration=trained_iteration)
    # start training and testing
    last_val_point = trained_iteration
    for val_point in val_point_list:
        # train
        trainer(train_iteration=val_point - last_val_point)
        time.sleep(2)
        # test
        save_flag, early_stop_flag = validator(trainer.iteration)
        #
        if save_flag and save_best_model:
            save_path = os.path.join(trainer.logdir, "best_model.pth".format(trainer.iteration))
            torch.save(trainer.state_dict(), save_path)
        # early stop
        if early_stop_flag:
            logger.info("Early stop as iteration {}".format(val_point))
            break
        #
        last_val_point = val_point
    #
    tb_writer.close()
