import os
import sys
import time
import json
import torch
import argparse
import logging

from tqdm import tqdm
from PIL import Image
from copy import copy, deepcopy
from functools import partial
from datasets import VerificationMode
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset
from lmdeploy.vl import load_image as lmdeploy_load_image
from lmdeploy import pipeline, TurbomindEngineConfig, VisionConfig, GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dense_caption_prompt = """You are an expert image analyst and descriptive writer specializing in creating "dense captions." Your task is to generate a single, continuous paragraph of highly detailed and comprehensive text that describes the provided image. Your description must be objective and based solely on visual evidence.

Follow this multi-step process for your analysis:

1. Holistic Overview:
Begin by establishing the overall scene. Describe the setting (e.g., urban street, natural landscape, indoor room), the time of day (e.g., midday, golden hour, night), the overall atmosphere or mood (e.g., bustling, serene, melancholic), and the general color palette.

2. Primary Subjects and Actions:
Identify and describe the primary subject(s) in detail. If they are people, describe their apparent age, gender, clothing, posture, expression, and any actions they are performing. If they are objects or animals, describe their type, condition, color, and position. Describe the interactions between primary subjects.

3. Secondary Elements and Background:
Detail the secondary subjects, significant objects, and the immediate background. Describe architectural elements, furniture, vehicles, flora, and fauna that populate the scene but are not the central focus. Describe their spatial relationship to the primary subjects.

4. Fine-Grained Details and Textures:
Scrutinize the image for fine-grained details. Mention specific textures (e.g., the rough bark of a tree, the smooth surface of a metal table, the fabric weave of a coat), small, easily missed objects, text or symbols visible on signs or clothing, reflections in windows or water, and the quality of light and shadow (e.g., sharp, defined shadows indicating harsh light, or soft, diffuse light).

5. Synthesis and Composition:
Conclude by synthesizing all observations. Briefly describe the photographic composition, such as the framing, perspective, and depth of field (e.g., a shallow depth of field blurring the background, a wide-angle shot capturing a vast landscape).

Formatting and Style Constraints:

- Output Format: Your entire output must be one single, continuous paragraph.
- No Line Breaks: Do not use any line breaks, newlines, or paragraph breaks (\\n).
- Style: Write in a descriptive, objective, and formal tone.
- Exclusions:
    - Do not start with phrases like "This is an image of," "The picture shows," or any similar introductory statement.
    - Do not include personal opinions, judgments, or interpretations that are not directly supported by visual evidence.
    - Do not use bullet points, lists, or headers in your final output. Your entire response must be the caption itself.

Example of Desired Output:

A vibrant and crowded marketplace unfolds under the bright, hazy sun of midday, characterized by a dominant palette of warm ochres, deep reds, and earthen browns. The central focus is a male vendor in his late fifties, wearing a light blue djellaba and a straw hat, who is carefully arranging a pyramid of colorful spices on a rough wooden stall; his face is weathered and creased in concentration. In front of him, a tourist with a backpack slung over one shoulder, clad in a khaki shirt, points at a specific spice mound while a young child clings to her hand, looking with wide eyes at a nearby stall selling intricately woven leather bags. The background is a dense tapestry of activity, with other shoppers and vendors creating a soft-focus blur of movement, set against the backdrop of ancient, reddish-pink plaster walls and arched doorways. Fine details abound, from the coarse texture of the burlap sacks holding grains and the gleam of polished brass lanterns hanging from a wooden beam, to the subtle shadows cast by the woven canopy overhead, dappling the ground in a shifting pattern of light. The composition is tight and layered, creating a deep sense of immersion and chaotic energy, capturing the scene from a slightly low, eye-level perspective that places the viewer directly within the bustling alleyway.

Your Task:

Now, analyze the following image and generate the dense caption, strictly adhering to all instructions above.
"""

def ds_has_valid_image(ex, key="jpg"):
    try:
        ex[key].verify()  # cheap sanity check
        return True
    except Exception as e:
        print("Error checking image: ", e)
        return False

def subsample_dataset(args, dataset):
    if args.start_idx is not None and args.num_samples is not None:
        start = min(args.start_idx, len(dataset))
        end = min(args.start_idx + args.num_samples, len(dataset))
        if start >= end:
            sys.exit()
        dataset = dataset.select(range(start, end))

    return dataset

def get_image_key(args):
    if args.dataset == "cc3m":
        return "jpg"
    elif args.dataset == "cc12m":
        return "jpg"
    elif args.dataset == "yfcc":
        return "images"
    elif args.dataset == "imagenette2":
        return "image"
    elif args.dataset == "food101":
        return "image"
    elif args.dataset == "cifar100":
        return "image"
    elif args.dataset == "SUN397":
        return "image"
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

def load_dataset(args) -> Dataset:
    key = get_image_key(args)
    dataset_name = datasets_dict[args.dataset]

    dataset = hf_load_dataset(dataset_name, split="train", verification_mode=VerificationMode.NO_CHECKS)
    dataset = dataset.add_column("index",list(range(len(dataset))))
    dataset = subsample_dataset(args, dataset)
    if args.dataset == "cifar100":
        return dataset
    else:
        dataset = dataset.filter(partial(ds_has_valid_image, key=key))

    return dataset


class CollateFN:
    def __init__(self, image_key: str, preprocess) -> None:
        self.image_key = image_key
        self.preprocess = preprocess

    def __call__(self, batch: list[dict]) -> tuple[list[Image.Image], list[dict]]:
        images = [item[self.image_key] for item in batch]
        images = [self.preprocess(image) for image in images]

        metadata = [{key: value for key, value in item.items() if key != self.image_key} for item in batch]
        return images, metadata



if __name__ == "__main__":
    # Get start time
    start_time = time.time()

    # Parse cmd args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B")
    parser.add_argument("--dataset", type=str, required=True, choices=["cc3m", "cc12m", "yfcc", "imagenette2", "cifar100", "food101", "SUN397", "wikiart", "rijksmuseum"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-path", type=str, default="results/recaption_images")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    # Make output path
    true_model_name = args.model.split("/")[-1]
    output_path = os.path.join(args.output_path, args.dataset, true_model_name)
    os.makedirs(output_path, exist_ok=True)

    # Check if output path already exists
    if os.path.exists(os.path.join(output_path, f"responses_{args.start_idx}_{args.start_idx + args.num_samples}.json")):
        print(f"Output path {output_path} already exists")
        sys.exit(1)

    # Warn user if using cpu although gpu is available
    if args.device == "cpu" and torch.cuda.is_available():
        logging.warning("Using CPU althouth GPU is available")

    # Load model
    backend_config = TurbomindEngineConfig(session_len=8192)
    vision_config = VisionConfig(max_batch_size=16)
    generation_config = GenerationConfig(max_new_tokens=1024, do_sample=False)
    captioning_model = pipeline(
        args.model,
        backend_config=backend_config,
        vision_config=vision_config,
    )

    # Load dataset
    dataset = load_dataset(args)
    image_key = get_image_key(args)
    collate_fn = CollateFN(image_key, lmdeploy_load_image)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Embed images
    responses = []

    for images, metadata in tqdm(dataloader):
        prompts = [copy(dense_caption_prompt) for _ in images]
        prompts = [(prompt, image) for prompt, image in zip(prompts, images)]
        answers = captioning_model(prompts, gen_config=generation_config)
        answers = [answer.text for answer in answers]

        for answer, meta in zip(answers, metadata):
            meta = deepcopy(meta)
            meta["answer"] = answer
            meta["global_idx"] = (args.start_idx or 0) + len(responses)
            responses.append(meta)

    if len(responses) > 0:
        if args.start_idx is not None and args.num_samples is not None:
            filename = f"responses_{args.start_idx}_{args.start_idx + args.num_samples}.json"
        else:
            filename = "responses.json"
        with open(os.path.join(output_path, filename), "w") as f:
            json.dump(responses, f)

    logging.info(f"Saved {len(responses)} responses to {output_path}")
    logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")