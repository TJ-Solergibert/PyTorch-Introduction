import argparse


def get_args():
    parser = argparse.ArgumentParser(description="SA-MIRI | Lab 8")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation per GPU")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for the DataLoaders")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--epochs_eval", type=int, default=5, help="How often we compute the accuracy on the validation set"
    )
    parser.add_argument("--iteration_logging", type=int, default=5, help="How often we log the loss during training")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["custom", "resnet50", "vit"],
        default="custom",
        help="Current supported models: ['custom', 'resnet50', 'vit']",
    )
    parser.add_argument(
        "--intermidiate_dimensions",
        nargs="+",
        type=int,
        default=None,
        help="Intermediate dimension of the custom dense model. e.g.: --intermidiate_dimensions 512 256 128 64",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Load ResNet50 pretrained weights",
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer type: SGD, Adam or AdamW")
    parser.add_argument("--learning_rate", type=float, default="5e-4", help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the input images")

    _args, _ = parser.parse_known_args()
    return _args
