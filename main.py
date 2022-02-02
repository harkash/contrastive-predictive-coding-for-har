from arguments import parse_args
from dataset import load_dataset
from model import CPC
from trainer import learn_model
from utils import save_model, set_all_seeds


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    set_all_seeds(args)
    print(args)

    # Data Loader
    data_loaders, dataset_sizes = load_dataset(args)

    # Creating the model
    model = CPC(args).to(args.device)

    # Training the model
    print('Training the model')
    model = learn_model(model, data_loaders=data_loaders,
                        dataset_sizes=dataset_sizes, args=args)

    # Save the model
    print('Saving the trained model!')
    save_model(model, args)

    print('---------Training complete!---------')
