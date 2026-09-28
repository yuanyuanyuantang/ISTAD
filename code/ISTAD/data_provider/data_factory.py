from torch.utils.data import DataLoader

ANOMALY_DATASETS = {
    'PSM': 'PSMSegLoader',
    'MSL': 'MSLSegLoader',
    'SMAP': 'SMAPSegLoader',
    'SMD': 'SMDSegLoader',
    'SWAT': 'SWATSegLoader',
    'EXATHLON': 'EXATHLONSegLoader',
}


def _get_anomaly_dataset_class(dataset_name):
    from data_provider import data_loader as loader

    if dataset_name not in ANOMALY_DATASETS:
        supported = ', '.join(sorted(ANOMALY_DATASETS.keys()))
        raise ValueError(f"Unsupported anomaly dataset '{dataset_name}'. Supported: {supported}")
    return getattr(loader, ANOMALY_DATASETS[dataset_name])


def data_provider(args, flag):
    if args.task_name != 'anomaly_detection':
        raise NotImplementedError("This branch only supports task_name='anomaly_detection'.")

    Data = _get_anomaly_dataset_class(args.data)

    score_train = flag == 'score_train'
    dataset_flag = 'train' if score_train else flag
    shuffle_flag = False if (dataset_flag == 'test' or dataset_flag == 'TEST' or score_train) else True
    drop_last = False
    batch_size = args.batch_size

    dataset_kwargs = dict(
        args=args,
        root_path=args.root_path,
        win_size=args.seq_len,
        flag=dataset_flag,
    )
    eval_step = int(getattr(args, 'istad_eval_step', 0) or 0)
    if eval_step > 0 and (dataset_flag in {'test', 'TEST'} or score_train):
        dataset_kwargs['step'] = eval_step
    data_set = Data(**dataset_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
