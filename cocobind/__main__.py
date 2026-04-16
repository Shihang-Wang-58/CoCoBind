"""CoCoBind Package - CLI entry point"""
from .train import main as train_main
from .eval import main as eval_main
from .screen import main as screen_main


def main():
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "train":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            train_main()
        elif cmd == "eval":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            eval_main()
        elif cmd == "screen":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            screen_main()
        else:
            print("Usage: python -m cocobind [train|eval|screen] [options]")
            print("  train - Train the CoCoBind model")
            print("  eval  - Evaluate a trained model")
            print("  screen - Rank a compound library against one RNA sequence")
    else:
        print("CoCoBind - RNA-Drug Interaction & Binding Site Prediction")
        print("Usage: python -m cocobind [train|eval|screen] [options]")
        print("  train  - Train the CoCoBind model")
        print("  eval   - Evaluate a trained model")
        print("  screen - Rank a compound library against one RNA sequence")


if __name__ == "__main__":
    main()
