import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--foo", nargs="*", type=float, default=[3])
args = parser.parse_args()

print(args.foo)
