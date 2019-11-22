from pathlib import Path

from self_supervised_3d_tasks.train_and_eval import train_and_eval
import json

def main():
    with open(Path(__file__).parent / "ukb3d.json", 'r') as f:
        args = json.load(f)
    train_and_eval(args)


if __name__ == "__main__":
    main()
