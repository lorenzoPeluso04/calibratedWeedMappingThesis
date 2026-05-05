from calweed.weedmap import WeedMapDataset
from torch.utils.data import DataLoader


def get_data():
    # Parameters sets for TRAINING
    root = "RoWeeder/dataset/patches/512"
    channels = ["RGB"]
    fields_TRAIN = ["000", "002", "004"]
    gt_folder = "RoWeeder/dataset/patches/512"
    input_transform = lambda x: x / 255.0
    target_transform=lambda x: x

    # WeedMapDataset istancing
    train_dataset = WeedMapDataset(
        root=root,
        channels=channels,
        fields=fields_TRAIN,
        gt_folder=gt_folder,
        transform=input_transform,
        target_transform=target_transform,
        return_path=True
        )


    # Parameters sets for VALIDATION
    fields_EVAL = ["001"]
    # Almost the same...

    # WeedMapDataset istancing
    eval_dataset = WeedMapDataset(
        root=root,
        channels=channels,
        fields=fields_EVAL,
        gt_folder=gt_folder,
        transform=input_transform,
        target_transform=target_transform,
        return_path=True
        )

    # Parameters sets for TEST
    fields_TEST = ["003"]
    # Almost the same...

    # WeedMapDataset istancing
    test_dataset = WeedMapDataset(
        root=root,
        channels=channels,
        fields=fields_TEST,
        gt_folder=gt_folder,
        transform=input_transform,
        target_transform=target_transform,
        return_path=True
        )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_dataloader, eval_dataloader, test_dataloader