# test.py

import os
import json

from tqdm import tqdm

import evaluate

from PIL import Image

from dataset import (
    load_dataframe,
    create_splits
)

from inference import (
    CaptionGenerator
)


RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


class Evaluator:

    def __init__(
        self,
        checkpoint_path,
        caption_file,
        img_dir
    ):

        self.generator = (
            CaptionGenerator(
                checkpoint_path
            )
        )

        self.image_dir = img_dir

        df = load_dataframe(
            caption_file
        )

        (
            _,
            _,
            self.test_df
        ) = create_splits(df)

        self.bleu = evaluate.load(
            "bleu"
        )

        self.meteor = evaluate.load(
            "meteor"
        )

        self.rouge = evaluate.load(
            "rouge"
        )

    def evaluate(self):

        predictions = []

        references = []

        sample_outputs = []

        print(
            "Running Evaluation..."
        )

        for _, row in tqdm(
            self.test_df.iterrows(),
            total=len(self.test_df)
        ):

            image_name = row["image"]

            reference = row["caption"]

            image_path = os.path.join(
                self.image_dir,
                image_name
            )

            prediction = (
                self.generator.predict(
                    image_path
                )
            )

            predictions.append(
                prediction
            )

            references.append(
                [reference]
            )

            sample_outputs.append(
                {
                    "image":
                    image_name,

                    "prediction":
                    prediction,

                    "reference":
                    reference
                }
            )

        bleu_score = (
            self.bleu.compute(
                predictions=predictions,
                references=references
            )
        )

        meteor_score = (
            self.meteor.compute(
                predictions=predictions,
                references=[
                    r[0]
                    for r in references
                ]
            )
        )

        rouge_score = (
            self.rouge.compute(
                predictions=predictions,
                references=[
                    r[0]
                    for r in references
                ]
            )
        )

        metrics = {

            "BLEU":
                bleu_score,

            "METEOR":
                meteor_score,

            "ROUGE":
                rouge_score
        }

        with open(
            f"{RESULTS_DIR}/metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        with open(
            f"{RESULTS_DIR}/predictions.json",
            "w"
        ) as f:

            json.dump(
                sample_outputs[:100],
                f,
                indent=4
            )

        print(
            "\nEvaluation Complete"
        )

        print(
            json.dumps(
                metrics,
                indent=4
            )
        )

        return metrics


def main():

    evaluator = Evaluator(

        checkpoint_path=
        "checkpoints/best_model.pt",

        caption_file=
        "data/captions.txt",

        image_dir=
        "data/images"
    )

    evaluator.evaluate()


if __name__ == "__main__":

    main()