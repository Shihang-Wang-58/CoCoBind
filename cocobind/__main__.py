"""CoCoBind Package - CLI entry point"""
from .train import main as train_main
from .eval import main as eval_main


def main():
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "train":
            sys.argv = sys.argv[1:]
            train_main()
        elif cmd == "eval":
            sys.argv = sys.argv[1:]
            eval_main()
        else:
            print("Usage: python -m cocobind [train|eval] [options]")
            print("  train - Train the CoCoBind model")
            print("  eval  - Evaluate a trained model")
    else:
        print("CoCoBind - RNA-Drug Interaction & Binding Site Prediction")
        print("Usage: python -m cocobind [train|eval] [options]")


if __name__ == "__main__":
    main()
