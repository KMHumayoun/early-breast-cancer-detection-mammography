import argparse
import json

from .cohort import build_cohort
from .runtime import benchmark, train
from .evaluation import evaluate


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="RSNA screening mammography study")
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--csv", required=True)
    prep.add_argument("--images", required=True)
    prep.add_argument("--output", default="runs/index.json")
    run = subs.add_parser("train")
    run.add_argument("--model", choices=["single","paired"], required=True)
    run.add_argument("--epochs", type=positive, default=10)
    run.add_argument("--seed", type=int, default=83)
    run.add_argument("--random-init", action="store_true", help="Use only for software tests")
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--single", required=True)
    evaluation.add_argument("--paired", required=True)
    evaluation.add_argument("--bootstrap", type=positive, default=1000)
    evaluation.add_argument("--seed", type=int, default=83)
    for command in (run, evaluation):
        command.add_argument("--index", default="runs/index.json")
        command.add_argument("--images", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--workers", type=int, default=0)
    bench = subs.add_parser("benchmark")
    bench.add_argument("--steps", type=positive, default=8)
    for command in (run, evaluation, bench):
        command.add_argument("--batch", type=positive, default=2)
    for command in (run, bench):
        command.add_argument("--height", type=positive, default=768)
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_cohort(args.csv,args.images,args.output)
    elif args.command == "benchmark":
        result = benchmark(args.height,args.batch,args.steps)
    elif args.command == "train":
        train(args)
        return
    else:
        result = evaluate(args)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
