backbone_optimizer = dict(
    type='SGD',
    lr=0.0004,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='sfda_simplified_contrastive_model',
    model_dict=dict(
        type='vit_sfda_basenet',
        use_bottleneck=True,
        bottleneck_dim=1024,
        width=1024,
        sfda_feat_width=256,
    ),
    classifier_dict=dict(
        type='vit_sfda_classifier',
        width=256,
        class_num=65,
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
