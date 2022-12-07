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
        resnet_name='ResNet101',
        bottleneck_dim=256,
    ),
    classifier_dict=dict(
        type='SFDAClassifier',
        num_class=12,
        bottleneck_dim=256,
    ),
    num_class=12,
    low_dim=256,
    model_moving_average_decay=0.99,
    optimizer=backbone_optimizer,
)


scheduler = dict(
    type='InvLR',
    gamma=0.0004,
    power=0.75,
)

models = dict(
    base_model=backbone,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
