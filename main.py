import argparse
from train import train
import torch.multiprocessing as mp
import torch
import os
def main(rank, world_size, args):
    train(args.run_name, args.nclasses, args.nepochs, args.base_path, reinforce=args.use_rl,
          verbose=args.verbose, img_size=args.img_size, batch_size=args.batch, agent_model=args.agent_path,
          warmup=args.warmup, use_baseline=args.baseline
          , logging=args.logging, alternate=args.alternate, rank=rank, world_size=world_size)

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
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument("--use_rl", action='store_true')
    parser.add_argument('--no-use_rl', dest='use_rl', action='store_false')
    parser.set_defaults(use_rl=True)

    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--no-baseline', dest='baseline', action='store_false')
    parser.set_defaults(baseline=True)

    parser.add_argument('--alternate', action='store_true')
    parser.add_argument('--no-alternate', dest='alternate', action='store_false')
    parser.set_defaults(alternate=True)

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='alternate', action='store_false')
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
    print(args)
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    )


