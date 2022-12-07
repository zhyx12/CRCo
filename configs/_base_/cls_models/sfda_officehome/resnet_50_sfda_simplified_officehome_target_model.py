backbone_optimizer = dict(
    type='SGD',
    lr=0.0001,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='sfda_simplified_contrastive_model',
    model_dict=dict(
        type='SFDAResNetBase',
        resnet_name='ResNet50',
        bottleneck_dim=256,
    ),
    classifier_dict=dict(
        type='SFDAClassifier',
        num_class=65,
        bottleneck_dim=256,
    ),
    num_class=65,
    low_dim=256,
    model_moving_average_decay=0.99,
    optimizer=backbone_optimizer,
)


scheduler = dict(
    type='ConstantLR',
    total_iters=1000000,
    factor=1.0,
)

models = dict(
    base_model=backbone,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
