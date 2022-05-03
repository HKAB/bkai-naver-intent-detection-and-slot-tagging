import argparse

from trainer_tune import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples, concat_train_dev_and_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = None #load_and_cache_examples(args, tokenizer, mode="test")

    if (args.concat_and_slit_train_dev):
        train_dataset, dev_dataset = concat_train_dev_and_split(args, [train_dataset, dev_dataset])

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "intent_loss_coef": tune.quniform(0.1, 1.0, 0.1),
        "slot_loss_coef": tune.quniform(0.1, 1.0, 0.1),
        "batch_size": tune.choice([16, 32, 64]),
        "args": args,
        "train_dataset": train_dataset,
        "dev_dataset": dev_dataset,
        "test_dataset": test_dataset
    }

    # trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    scheduler = ASHAScheduler()
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "sementic_frame_acc", "training_iteration"]
    )

    result = tune.run(
        Trainer,
        metric="sementic_frame_acc",
        mode="max",
        stop={
            "training_iteration": 1
        },
        resources_per_trial={"cpu": 2, "gpu": int(args.no_cuda)},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    print("Best trial config: {}".format(result.best_config))
    print("Best trial : {}".format(result.best_trial))
    print("Best checkpoint dir is:", result.best_checkpoint)

    # if args.do_eval:
    #     trainer.load_model()
    #     trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")


    # Data option
    parser.add_argument("--concat_and_slit_train_dev", action="store_true", help="Concat train and dev data, then random split again")
    parser.add_argument('--dev_size', type=int, default=500, help="Use when --concat_and_slit_train_dev is True. Size of the dev set.")
    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
