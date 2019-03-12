import argparse
from models.AlexNet import alexnet_model
from models.Vgg16Net import vgg16net_model
from models.ResNet import resnet_model
from models.ResNetBottleneck import resnet_bottleneck_model
from models.ResNeXt import resnext_model
from models.DenseNet import densenet_model
from models.GoogLeNet import googlenet_model
from models.SENet import senet_model

from data_utils import check_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of batch')
    parser.add_argument('--dataset_path', type=str, default='./data/',
                        help='Directory name to load data')
    parser.add_argument('--save_model_path', type=str, default='./save/model/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--model', type=str, default='alexnet',
                        help='alexnet or vgg16net or resnet or resnet_bottleneck or resnext or densenet or googlenet')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--device', type=str, default='cuda',
                        help='train on GPU or CPU')
    parser.add_argument('--save_history_path', type=str, default='./save/history/',
                        help='Directory name to save the loss and accuracy history')
    return parser.parse_args()


def main():
    args = parse_args()

    check_folder(args.dataset_path)
    check_folder(args.save_model_path)
    check_folder(args.save_history_path)


    if args.model == 'alexnet':
        model = alexnet_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "vgg16net":
        model = vgg16net_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "resnet":
        model = resnet_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "resnet_bottleneck":
        model = resnet_bottleneck_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "resnext":
        model = resnext_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "densenet":
        model = densenet_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "googlenet":
        model = googlenet_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    elif args.model == "senet":
        model = senet_model(args.dataset_path,
               args.save_model_path,
               args.save_history_path,
               args.epochs,
               args.batch_size,
               args.device,
               args.mode)
    else:
        model = None
        print("model is not exist")

    if model is not None:
        if args.mode == "train":
            model.train()
        else:
            model.test()    

if __name__ == "__main__":
    main()