from typing import Tuple
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Adafactor,
    AutoModelForNextSentencePrediction as AutoNSPModel,
    AutoTokenizer,
)
from data_caching import CACHE_DIR, CACHE_CAPTION_DIR, CACHE_PLOT_DIR
from libs.preprocess import load_data

DATA_DIR = "/data/coherence_evaluation"
DATA_NAME = {
    "train": "desc_and_plot_train.csv",
    "validation": "desc_and_plot_val.csv",
    "test": "desc_and_plot_val.csv",
}
MODEL_NAME_OR_PATH = "bert-base-uncased"


class PlotCaptionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
    ):
        self.split = split
        self.data_df = load_data(
            data_dir=DATA_DIR,
            data_filename=DATA_NAME[split],
            rename_dict={
                "videoid": "video_id",
                "imdbid": "movie_id",
                "caption_from_old": "caption",
                "synopsis": "plot",
            },
        )
        self.cls_vector = torch.load(CACHE_DIR / "cls.pt", map_location="cpu")
        self.sep_vector = torch.load(CACHE_DIR / "sep.pt", map_location="cpu")

    def __len__(self) -> int:
        # Positive sample, Negative sample
        return len(self.data_df) * 2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, bool]:
        row = self.data_df.iloc[idx // 2]
        label = True

        # Negative sample
        if idx % 2 == 1:
            label = False
            movie_id = row["movie_id"]
            while movie_id == row["movie_id"]:
                row = self.data_df.sample(n=1).iloc[0]

        cached_plot_path = CACHE_PLOT_DIR / f"{row['movie_id']}.pt"
        cached_caption_path = CACHE_CAPTION_DIR / f"{row['video_id']}.pt"

        # Cache
        if cached_plot_path.exists() and cached_caption_path.exists():
            plot_vector = torch.load(cached_plot_path, map_location="cpu")
            caption_vector = torch.load(cached_caption_path, map_location="cpu")

        sentence_vectors = torch.cat(
            (self.cls_vector, plot_vector, self.sep_vector, caption_vector),
            dim=0,
        )
        return sentence_vectors, label


class PlotCaptionDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        # stage: fit, validate, test, predict
        self.dataset = dict()
        if stage == "fit":
            self.dataset["train"] = PlotCaptionDataset(split="train")
            self.dataset["validation"] = PlotCaptionDataset(split="validation")
        elif stage == "validate":
            self.dataset["validation"] = PlotCaptionDataset(split="validation")
        elif stage == "test":
            self.dataset["test"] = PlotCaptionDataset(split="test")
        else:
            raise ValueError(f"Stage must be in [fit, validate, test] - {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            pin_memory=True,
        )


class PlotCaptionNSPModel(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.model = AutoNSPModel.from_pretrained(model_name_or_path)

    def forward(self, sentence_vectors):
        output = self.model(inputs_embeds=sentence_vectors)
        output = F.softmax(output.logits, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        # TODO collate_fn -> stack expects each tensor to be equal size, but got [214, 768] ... [43, 768] ...
        sentence_vectors, labels = batch
        outputs = self(sentence_vectors)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sentence_vectors, labels = batch
        outputs = self(sentence_vectors)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, axis=1)
        acc = torch.sum(preds == labels)

        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adafactor(
            model.parameters(),
            lr=self.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )


if __name__ == "__main__":
    seed_everything(42)

    dm = PlotCaptionDataModule(model_name_or_path=MODEL_NAME_OR_PATH)
    dm.setup("fit")
    model = PlotCaptionNSPModel(
        model_name_or_path=MODEL_NAME_OR_PATH,
    )

    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model, datamodule=dm)
