import argparse
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-ViT arg parser")

    parser.add_argument("--run_name", type=str, help="name of run")
    parser.add_argument("--base_path", type=str, help="path to pretrained vit_b_32 model", default=None)
    parser.add_argument("--agent_path", type=str, help="path to pretrained agent model", default=None)
    parser.add_argument("--nclasses", type=int, help="number of classes", default=10)
    parser.add_argument("--nepochs", type=int, help="number of epochs", default=300)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-8)
    parser.add_argument("--batch", type=int, help="batch size", default=32)

    parser.add_argument("--img_size", type=int, help="height (width) of images)", default=224)
    parser.add_argument("--logging", type=int, help="number of epochs until loggin starts", default=5)
    parser.add_argument("--warmup", type=int, help="number of warmup", default=10)
    parser.add_argument("--verbose", type=bool, help="use verbose mode", default=True)
    parser.add_argument("--use_rl", type=bool, help="use rl mode", default=True)
    parser.add_argument("--use_baseline", type=bool, help="use baseline for rewards", default=False)
    parser.add_argument("--alternate", type=bool, help="alternate model and agent training", default=False)
    args = parser.parse_args()
    print(args)
    train(args.run_name, args.nclasses, args.nepochs, args.base_path, reinforce=args.use_rl,
          verbose=args.verbose, img_size=args.img_size, batch_size=args.batch, agent_model=args.agent_path,warmup=args.warmup)