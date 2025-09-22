import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import Trainer


class NoShuffleTrainer(Trainer):

    def get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with shuffling disabled.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Use the overridden sampler to preserve sorted order
        train_sampler = self.get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )

    def training_step(self, model, inputs, num_items_in_batch: int | None = None):
        # Prepare inputs once here; avoid double work in super by passing them through.
        inputs = self._prepare_inputs(inputs)
        print(inputs["labels"].shape)

        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(f"{k}: {v.shape}, {v.dtype}, {v.device}")
            else:
                print(f"{k}: {type(v)}, {v}")
        print("------------------------------------")

        return super().training_step(model, inputs)
