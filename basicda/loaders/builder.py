#
# ----------------------------------------------
from mmcv.utils import Registry
from torch.utils.data import DataLoader
import numpy as np
import random
from mmcv.parallel import collate
from functools import partial
from mmcv.runner import get_dist_info
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler
from mmcv.utils import build_from_cfg
from basicda.utils import get_root_logger

DATASETS = Registry('basicda_datasets')
DATABUILDERS = Registry('basicda_databuilders')


@DATABUILDERS.register_module(name='default')
class DefaultDataBuilder(object):
    def __init__(self, dataset, samples_per_gpu, num_workers, shuffle, drop_last, seed, **kwargs):
        sampler = self.build_sampler(dataset, shuffle, samples_per_gpu, seed)
        collate_fn = self.build_collate_fn(samples_per_gpu)
        worker_init_fn = self.build_init_fn(num_workers, seed)
        self.dataloader = DataLoader(dataset, batch_size=samples_per_gpu, num_workers=num_workers, shuffle=False,
                                     sampler=sampler,
                                     drop_last=drop_last, collate_fn=collate_fn,
                                     worker_init_fn=worker_init_fn,**kwargs)

    def build_sampler(self, dataset, shuffle, samples_per_gpu=None, seed=None):
        return DistributedSampler(dataset, shuffle=shuffle)

    def build_collate_fn(self, samples_per_gpu):
        return partial(collate, samples_per_gpu=samples_per_gpu)

    def build_init_fn(self, num_workers, seed):
        rank, world_size = get_dist_info()
        return partial(self.worker_init_fn, num_workers=num_workers, rank=rank,
                       seed=seed) if seed is not None else None

    def get_dataloader(self):
        return self.dataloader

    def worker_init_fn(self, worker_id, num_workers, rank, seed):
        # The seed of each worker equals to
        # num_worker * rank + worker_id + user_seed
        worker_seed = num_workers * rank + worker_id + seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def process_one_dataset(dataset_args, databuilder_args, pipelines, data_root, samplers_per_gpu, n_workers, shuffle,
                        drop_last=True, random_seed=None):
    logger = get_root_logger()
    #
    dataset_params = deepcopy(dataset_args)
    #
    if 'pipeline' not in dataset_args:
        dataset_params['pipeline'] = pipelines
    #
    if 'data_root' not in dataset_params:
        dataset_params['data_root'] = data_root
    #
    dataset = build_from_cfg(dataset_params, DATASETS)
    # check if dataset has name and split attribute
    assert hasattr(dataset,'name'), 'Please add "name" attribute for dataset'
    assert hasattr(dataset,'split'), 'Please add "split" attribute for {} dataset'.format(dataset.name)
    #
    temp_samples_per_gpu = dataset_params.pop('samples_per_gpu', None)
    #
    temp_databuilder_args = dict(
        dataset=dataset,
        samples_per_gpu=temp_samples_per_gpu,
        num_workers=n_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=random_seed,
    )
    temp_databuilder_args.update(databuilder_args)
    if 'type' not in temp_databuilder_args:
        temp_databuilder_args['type'] = 'default'
        logger.info("You are using the DEFAULT data builder in basicda")
    loader = build_from_cfg(temp_databuilder_args, DATABUILDERS).get_dataloader()
    return loader


def parse_args_for_multiple_datasets(dataset_args, data_root, random_seed=None):
    """

    :param data_root:
    :param random_seed:
    :param dataset_args:
    :return: 返回一个list
    """
    logger = get_root_logger()
    #
    dataset_args = deepcopy(dataset_args)
    trainset_args = dataset_args['train']
    testset_args = dataset_args['test']
    train_pipeline = trainset_args.get('pipeline', None)
    test_pipeline = testset_args.get('pipeline', None)
    # other global args for dataloader
    train_samples_per_gpu = trainset_args.get('samples_per_gpu', None)
    test_samples_per_gpu = testset_args.get('samples_per_gpu', None)
    n_workers = dataset_args['n_workers']
    # 训练集
    train_loaders = []
    for i in range(100):
        if i in trainset_args.keys():
            temp_data_builder_args = trainset_args[i].pop('builder', None)
            assert temp_data_builder_args is not None, "You should specify builder for {} train dataset".format(i)
            temp_train_loader = process_one_dataset(trainset_args[i], temp_data_builder_args,
                                                    pipelines=train_pipeline,
                                                    data_root=data_root,
                                                    samplers_per_gpu=train_samples_per_gpu, n_workers=n_workers,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    random_seed=random_seed)
            train_loaders.append(temp_train_loader)
        else:
            break

    # 测试集
    test_loaders = []
    for i in range(100):
        if i in testset_args.keys():
            temp_data_builder_args = testset_args[i].pop('builder', None)
            assert temp_data_builder_args is not None, "You should specify builder for {} test dataset".format(i)
            temp_test_loader = process_one_dataset(testset_args[i], temp_data_builder_args, pipelines=test_pipeline,
                                                   data_root=data_root,
                                                   samplers_per_gpu=test_samples_per_gpu,
                                                   n_workers=n_workers,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   random_seed=random_seed,
                                                   )
            test_loaders.append(temp_test_loader)
        else:
            break
    #
    for i, loader in enumerate(train_loaders):
        logger.info('{} train loader has {} images'.format(i, len(loader.dataset)))
    return train_loaders, test_loaders
