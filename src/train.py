# src/train.py
from pathlib import Path
import tensorflow as tf
import argparse
from transformers import create_optimizer
from data_preprocessing import build_tf_dataset, RNG_SEED
from model import get_model
from datasets import load_from_disk


def main(args):
    tf.keras.utils.set_random_seed(RNG_SEED)
    ds_path = Path(args.dataset).resolve()
    tfdata = tf.data.Dataset.load((ds_path / "tf_dataset").as_posix())

    model = get_model()
    steps_per_epoch = tf.data.experimental.cardinality(tfdata).numpy()

    optimizer, _ = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=steps_per_epoch * args.epochs,
    )
    model.compile(
        optimizer=optimizer,
        loss=model.compute_loss,
        metrics=["accuracy"],
    )

    history = model.fit(tfdata, epochs=args.epochs)

    ckpt_dir = Path("outputs/checkpoints/local_run")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"▶ Saving model to {ckpt_dir.resolve()}")
    model.save_pretrained(ckpt_dir.as_posix())

    # Save metrics
    (ckpt_dir / "training_history.json").write_text(
        tf.keras.utils.serialize_keras_object(history.history)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/raid_train_3000",
        help="Path produced by data_preprocessing.py",
    )
    parser.add_argument("--epochs", type=int, default=2)
    main(parser.parse_args())
