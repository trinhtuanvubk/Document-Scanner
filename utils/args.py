import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1234)
    # parser.add_argument('--scenario', type=str, default='test_output_model')
    parser.add_argument('--scenario', type=str, default='train')

    # generate doc set
    parser.add_argument('--dst_img_dir', type=str, default="DOCUMENTS/CHOSEN/images")
    parser.add_argument('--dst_msk_dir', type=str, default="DOCUMENTS/CHOSEN/masks")
    parser.add_argument('--raw_img_dir', type=str, default="all_images/simple_images")

    #resizer
    parser.add_argument("-s", "--source-dir", type=str, default="DOCUMENTS/CHOSEN/images", help="Input Source folder path")
    parser.add_argument("-d", "--destination-dir", type=str, default="DOCUMENTS/CHOSEN/resized_images", help="Output destination folder path")
    parser.add_argument("-x", "--img-size", type=int, default=640, help="size of resized Image")

    #create dataset
    parser.add_argument('--doc_img_path', type=str, default="DOCUMENTS/CHOSEN/resized_images")
    parser.add_argument('--doc_msk_path', type=str, default="DOCUMENTS/CHOSEN/resized_masks")
    parser.add_argument('--gen_img_dir', type=str, default="final_set/images")
    parser.add_argument('--gen_msk_dir', type=str, default="final_set/masks")
    parser.add_argument('--brg_img_dir', type=str, default="all_images/simple_background")

    # split data
    parser.add_argument('--ori_img_dir', type=str, default="final_set/images")
    parser.add_argument('--ori_msk_dir', type=str, default="final_set/masks")
    parser.add_argument('--train_img_dir', type=str, default="document_dataset_resized/train/images")
    parser.add_argument('--train_mask_dir', type=str, default="document_dataset_resized/train/masks")
    parser.add_argument('--val_img_dir', type=str, default="document_dataset_resized/valid/images")
    parser.add_argument('--val_mask_dir', type=str, default="document_dataset_resized/valid/masks")
    parser.add_argument('--img_per_doc', type=int, default=6)
    parser.add_argument('--max_dim_size', type=int, default=480)


    # model
    parser.add_argument("--backbone_model", type=str, default='mbv3')
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default='document_dataset_resized')
    parser.add_argument("--metric_name", type=str, default='iou', help='iou, dice')
    parser.add_argument("--pretrained_path", type=str, default="model_repository/mbv3_averaged.pth")
    parser.add_argument("--checkpoint_path", type=str, default="model_repository/mbv3.pth")

    # infer
    parser.add_argument("--data_infer", type=str, default="./test_AI")
    parser.add_argument("--data_result", type=str, default="./output_test_AI")
    
    parser.add_argument("--image_size", type=int, default=384)

    # params
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--shuffle', action='store_false')

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args