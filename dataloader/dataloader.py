import os
from .dataset import SegDataset
from torch.utils.data import DataLoader

def get_dataset(args):

    train_img_dir = os.path.join(args.data_directory, "train", "images")
    train_msk_dir = os.path.join(args.data_directory, "train", "masks")

    valid_img_dir = os.path.join(args.data_directory, "valid", "images")
    valid_msk_dir = os.path.join(args.data_directory, "valid", "masks")

    train_img_paths = [os.path.join(train_img_dir, i) for i in os.listdir(train_img_dir)]
    train_msk_paths = [os.path.join(train_msk_dir, i) for i in os.listdir(train_msk_dir)]

    valid_img_paths = [os.path.join(valid_img_dir, i) for i in os.listdir(valid_img_dir)]
    valid_msk_paths = [os.path.join(valid_msk_dir, i) for i in os.listdir(valid_msk_dir)]

    train_ds = SegDataset(img_paths=train_img_paths, mask_paths=train_msk_paths, data_type="train")
    valid_ds = SegDataset(img_paths=valid_img_paths, mask_paths=valid_msk_paths, data_type="valid")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=True)

    return train_loader, valid_loader

    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, args):
        self.dl = dl
        self.device = args.device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)