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
        """Preserve HF training semantics (backward/AMP) but add a small hook.
        Avoid torch.no_grad() and model.eval() to keep gradients flowing.
        """
        # Prepare inputs once here; avoid double work in super by passing them through.
        inputs = self._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
            print(inputs["labels"].shape)
        # Delegate to the base implementation so accelerator + scaler work correctly
        return super().training_step(model, inputs)
