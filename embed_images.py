import argparse
import json
import os
import sys
from functools import partial
from typing import Dict, List

import torch
import open_clip
from tqdm import tqdm
from io import BytesIO
from PIL import Image as PILImage, ImageFile
from datasets import Image as HFImage
from torch.utils.data import DataLoader
from datasets import Dataset, VerificationMode, load_dataset as hf_load_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def ds_has_valid_image(ex: dict, key: str = "jpg") -> bool:
    """
    Open the image ourselves (without triggering HF's EXIF logic) and verify.
    Works whether the column is decoded (PIL.Image) or raw (dict with bytes/path).
    """
    try:
        img = ex[key]
        if isinstance(img, dict):  # decode=False: {'bytes': ..., 'path': ...}
            if img.get("bytes") is not None:
                im = PILImage.open(BytesIO(img["bytes"]))
            else:
                im = PILImage.open(img["path"])
        else:  # already a PIL.Image
            im = img
        im.verify()  # cheap corruption check; doesn't parse EXIF
        return True
    except Exception:
        return False

def subsample_dataset(args: argparse.Namespace, dataset: Dataset) -> Dataset:
    """
    Subsamples the dataset based on a start index and a specified number of samples.
    """
    if args.start_idx is not None and args.num_samples is not None:
        start = min(args.start_idx, len(dataset))
        end = min(args.start_idx + args.num_samples, len(dataset))
        if start >= end:
            print(f"Error: start_idx ({start}) is greater than or equal to end_idx ({end}). Exiting.")
            sys.exit()
        print(f"Subsampling dataset from index {start} to {end}.")
        dataset = dataset.select(range(start, end))

    return dataset

def get_image_key(args: argparse.Namespace) -> str:
    """
    Returns the key for the image column based on the dataset name.
    """
    key_map = {
        "cc3m": "jpg",
        "cc12m": "jpg",
        "yfcc": "images",
        "imagenette2": "image",
        "food101": "image",
        "cifar100": "image",
        "SUN397": "image",
        "wikiart": "image"
    }
    if args.dataset in key_map:
        return key_map[args.dataset]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

datasets_dict = {
    "cc3m": "pixparse/cc3m-wds",
    "cc12m": "pixparse/cc12m-wds",
    "yfcc": "Kaichengalex/YFCC15M",
    "imagenette2": "johnowhitaker/imagenette2-320",
    "cifar100": "tanganke/cifar100",
    "food101": "ethz/food101",
    "SUN397": "1aurent/SUN397",
    "wikiart": "huggan/wikiart"
}

def load_dataset(args: argparse.Namespace) -> Dataset:
    key = get_image_key(args)
    dataset_name = datasets_dict[args.dataset]
    print(f"Loading dataset: {dataset_name} with image key: '{key}'")

    dataset = hf_load_dataset(dataset_name, split="train", verification_mode=VerificationMode.NO_CHECKS)
    dataset = dataset.add_column("index", list(range(len(dataset))))

    dataset = subsample_dataset(args, dataset)

    if args.dataset != "cifar100":
        print("Filtering for valid images...")

        # 👇 disable HF auto-decoding (and therefore EXIF parsing) during filtering
        dataset = dataset.cast_column(key, HFImage(decode=False))

        original_size = len(dataset)
        # (optional) speed up: num_proc=os.cpu_count()//2 or similar
        dataset = dataset.filter(partial(ds_has_valid_image, key=key), num_proc=8)  # , num_proc=8
        print(f"Filtered dataset from {original_size} to {len(dataset)} samples.")

    return dataset


class CollateFN:
    def __init__(self, image_key: str, preprocess) -> None:
        self.image_key = image_key
        self.preprocess = preprocess

    def _open_image(self, img_field):
        # img_field is either a dict (decode=False) or a PIL.Image
        if isinstance(img_field, dict):
            if img_field.get("bytes") is not None:
                return PILImage.open(BytesIO(img_field["bytes"]))
            # fall back to path on disk
            return PILImage.open(img_field["path"])
        # already a PIL.Image (in case some datasets still decode upstream)
        return img_field

    def __call__(self, batch):
        tensors, metas = [], []
        for item in batch:
            try:
                im = self._open_image(item[self.image_key]).convert("RGB")
                tensors.append(self.preprocess(im))
                metas.append({k: v for k, v in item.items() if k != self.image_key})
            except Exception:
                # skip the one bad sample; the batch will just be smaller
                continue

        if not tensors:
            # if everything in the batch failed, signal to re-fetch next batch
            raise RuntimeError("All images in batch failed to load")

        return torch.stack(tensors), metas


def main(args: argparse.Namespace):
    """
    Main function to run the image encoding process.
    """
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)
    if args.start_idx is not None and args.num_samples is not None:
        embeddings_path = os.path.join(args.save_path, f"embeddings_{args.start_idx}_{args.start_idx + args.num_samples}.pt")
        metadata_path = os.path.join(args.save_path, f"metadata_{args.start_idx}_{args.start_idx + args.num_samples}.jsonl")
    else:
        embeddings_path = os.path.join(args.save_path, "embeddings.pt")
        metadata_path = os.path.join(args.save_path, "metadata.jsonl")

    # --- Load OpenCLIP Model ---
    print(f"Loading OpenCLIP model: {args.model_name} with pretrained weights: {args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    model.eval()

    # --- Load Dataset ---
    dataset = load_dataset(args)
    image_key = get_image_key(args)

    collate_fn = CollateFN(image_key=image_key, preprocess=preprocess)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important to keep the original order for mapping.
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Encoding Loop ---
    all_embeddings = []
    all_metadata = []

    print("Starting image encoding...")
    with torch.no_grad():
        for images, metadata in tqdm(dataloader, desc="Encoding batches"):
            images = images.to(device)

            # Encode images and normalize the resulting feature vectors.
            image_features = model.encode_image(images)
            # Store results, moving embeddings to CPU to save GPU memory.
            all_embeddings.append(image_features.cpu())
            all_metadata.extend(metadata)

    # --- Save Results ---
    if not all_embeddings:
        print("Warning: No embeddings were generated. Check dataset and filtering.")
        return

    print("Concatenating all embedding batches...")
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Saving {final_embeddings.shape[0]} embeddings to {embeddings_path}")
    torch.save(final_embeddings, embeddings_path)

    print(f"Saving {len(all_metadata)} metadata entries to {metadata_path}")
    with open(metadata_path, 'w') as f:
        for item in all_metadata:
            # The 'index' key is preserved, allowing reconstruction of the original order.
            f.write(json.dumps(item) + '\n')

    print("Script finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a dataset's images using OpenCLIP's vision encoder.")

    # --- Command-line Arguments ---
    parser.add_argument("--dataset", type=str, required=True, choices=datasets_dict.keys(),
                        help="Name of the dataset to process.")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Starting index for subsampling the dataset.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process from the start index.")

    parser.add_argument("--model-name", type=str, default="ViT-B-32",
                        help="Name of the OpenCLIP model architecture.")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="Pretrained dataset for the OpenCLIP model.")

    parser.add_argument("--save-path", type=str, required=True,
                        help="Directory to save the embeddings and metadata files.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for the DataLoader.")

    args = parser.parse_args()
    main(args)