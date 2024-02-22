import torch
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

def average_checkpoints(checkpoint1_path, checkpoint2_path):
    # Load the model
    model = deeplabv3_mobilenet_v3_large(num_classes=2, aux_loss=True)

    # Load the state dictionaries of the checkpoints
    checkpoint1 = torch.load(checkpoint1_path, map_location=torch.device('cpu'))
    print(checkpoint1.keys())
    checkpoint2 = torch.load(checkpoint2_path, map_location=torch.device('cpu'))

    averaged_state_dict = {}
    for key in checkpoint1.keys():
        averaged_state_dict[key] = (checkpoint1[key] + checkpoint2[key]) / 2

    # Load the averaged state_dict into the model
    model.load_state_dict(averaged_state_dict)

    # Save the averaged model
    torch.save(model.state_dict(), 'model_repository/mbv3_averaged_averaged.pth')

if __name__ == "__main__":
    # for averaged
    # checkpoint1_path = '/home/aittgp/vutt/workspace/Document-Scanner/model_repository/model_mbv3_iou_mix_2C049.pth'
    # checkpoint2_path = '/home/aittgp/vutt/workspace/Document-Scanner/model_repository/mbv3_15k.pth'
    
    # for averaged averaged
    checkpoint1_path = '/home/aittgp/vutt/workspace/Document-Scanner/model_repository/mbv3_averaged.pth'
    checkpoint2_path = '/home/aittgp/vutt/workspace/Document-Scanner/model_repository/mbv3_301.pth'

    average_checkpoints(checkpoint1_path, checkpoint2_path)
