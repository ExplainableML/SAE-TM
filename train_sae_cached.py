import logging
import torch
import numpy as np
import argparse
import os

from torch.utils.data import IterableDataset, DataLoader
from libraries.dictionary_learning.training import trainSAE
from libraries.dictionary_learning.trainers import StandardTrainer
from libraries.dictionary_learning.trainers import JumpReluTrainer
from libraries.dictionary_learning.trainers import BatchTopKTrainer
from libraries.dictionary_learning.trainers import TopKTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sparse_autoencoder_trainers = {
    "StandardTrainer": StandardTrainer,
    "JumpReluTrainer": JumpReluTrainer,
    "BatchTopKTrainer": BatchTopKTrainer,
    "TopKTrainer": TopKTrainer
}


class CachedEmbeddingsDataset(IterableDataset):
    def __init__(self, path_to_embeddings: str):
        self.path_to_embeddings = path_to_embeddings
        if os.path.isdir(self.path_to_embeddings):
            self.embedding_files = os.listdir(self.path_to_embeddings)
            self.embedding_files = [file for file in self.embedding_files if file.endswith(".pt")]
        elif os.path.isfile(self.path_to_embeddings):
            self.embedding_files = [os.path.basename(self.path_to_embeddings)]
            self.path_to_embeddings = os.path.dirname(self.path_to_embeddings)
        else:
            raise ValueError(f"Path {path_to_embeddings} is neither a directory nor a file.")

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            embedding_files = self.embedding_files[worker_info.id::worker_info.num_workers]
            generator = np.random.default_rng(seed=worker_info.id)
        else:
            embedding_files = self.embedding_files
            generator = np.random.default_rng(seed=42)

        # Cycle over embedding files indefinitely so that training can run
        # for the full number of requested steps (trainSAE breaks at step >= steps)
        while True:
            if len(embedding_files) > 1:
                generator.shuffle(embedding_files)
            for file in embedding_files:
                embeddings = torch.load(os.path.join(self.path_to_embeddings, file), map_location="cpu").to(dtype=torch.float8_e4m3fn)
                permuted_indices = generator.permutation(len(embeddings))
                for idx in permuted_indices:
                    yield embeddings[idx].to(dtype=torch.float32)
    
    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default='StandardTrainer')
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--l1-penalty", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-path", type=str, default='results/trained_models/')
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=112233)
    parser.add_argument("--path-to-embeddings", type=str, default="data/embedding_datasets/")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    dataset = CachedEmbeddingsDataset(path_to_embeddings=args.path_to_embeddings)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=CachedEmbeddingsDataset.collate_fn
    )

    embedding_dim = next(iter(dataset)).shape[-1]

    # Make save path
    experiment_name = f"trainer={args.trainer}_expansion_factor={args.expansion_factor}"
    hparams_str = f"lr={args.lr}_top_k={args.top_k}_l1_penalty={args.l1_penalty}_batch_size={args.batch_size}_steps={args.steps}_warmup_ratio={args.warmup_ratio}"
    save_path = os.path.join(args.save_path, experiment_name, hparams_str)
    os.makedirs(save_path, exist_ok=True)

    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train SAE
    steps = args.steps
    warmup_steps = args.warmup_ratio * steps
    save_steps = list(range(0, steps, args.save_interval)) + [steps-1]
    trainer = sparse_autoencoder_trainers[args.trainer]

    trainer_cfg = {
        "trainer": trainer,
        "activation_dim": embedding_dim,
        "dict_size": args.expansion_factor * embedding_dim,
        "lr": args.lr,
        "device": device,
        "steps": steps,
        "lm_name": "cached_embeddings",
        "layer": "embedding",
        "warmup_steps": warmup_steps,
    }

    if args.trainer in ["StandardTrainer", "JumpReluTrainer"]:
        trainer_cfg["sparsity_warmup_steps"] = warmup_steps
        trainer_cfg["l1_penalty"] = args.l1_penalty
    elif args.trainer in ["BatchTopKTrainer", "TopKTrainer"]:
        trainer_cfg["k"] = args.top_k

    trainSAE(
        data=dataloader,
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_steps=save_steps,
        save_dir=save_path,
        verbose=True,
        log_steps=200,
        device=device
    )

