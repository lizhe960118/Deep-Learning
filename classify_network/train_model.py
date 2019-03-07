import argparse
from conditional_gan import CGAN
from utils import check_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=20, help='The size of batch')
    parser.add_argument('--dataset_path', type=str, default='./data/',
                        help='Directory name to load data')
    parser.add_argument('--save_model_path', type=str, default='./save/model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--model', type=str, default='alexnet',
                        help='alexnet or vgg16net or resnet or resnet_bottleneck or resnext or densenet or googlenet')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--device', type=str, default='cuda',
                        help='train on GPU or CPU')
    parser.add_argument('--save_training_history', type=str, default='./save/training_history/',
                        help='Directory name to save the training history')
    parser.add_argument('--save_testing_history', type=str, default='./save/testing_history/',
                        help='Directory name to save the testing history')
    return parser.parse_args()


def main():
    args = parse_args()

    check_folder(args.dataset_path)
    check_folder(args.save_path)


    if args.model == 'alexnet':
        model = alexnet_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == "vgg16":
        model = vgg16net_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == "resnet":
        model = resnet_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == "resnet_bottleneck":
        model = resnet_bottleneck_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == " resnext":
        model =  resnext_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == "densenet":
        model = dense_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    elif args.model == "googlenet":
        model = googlenet_model(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    else:
        model = None
        print("model is not exist")

    if model is not None:
        if args.mode == "train":
            check_folder(args.save_training_img_path)
            model.train()
        else:
         # args.mode == "test":
            check_folder(args.save_testing_img_path)
            model.test()    

if __name__ == "__main__":
    main()