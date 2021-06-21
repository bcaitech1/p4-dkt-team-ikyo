import os
import argparse

def model_select(args):
    if args.model_name == 'QUA':
        if args.train:
            from QuestionUserAttention.train import main
        else:
            from QuestionUserAttention.inference_kfold import main 

        from QuestionUserAttention.args import parse_args
        args = parse_args(mode="train")
        main(args)
    elif args.model_name == 'simple':
        from simple.train import main
        from simple.args import parse_args
        
        args = parse_args(mode='train')
        os.makedirs(args.model_dir, exist_ok=True)
        main(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="simple", type=str, help="Model Name")
    parser.add_argument("--train", default=False, type=bool, help="Train Inference")
    args = parser.parse_args()
    model_select(args)