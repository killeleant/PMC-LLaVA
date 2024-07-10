import re
import torch
import argparse
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    InstructBlipQFormerConfig,
    InstructBlipQFormerModel,
    InstructBlipProcessor,
)

QformerProcessor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
device = "cuda" if torch.cuda.is_available() else "cpu"

def creat_prompt(qs, model, model_name, caption=None):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    if caption:
        conv.append_message(conv.roles[1], caption)
    else:
        conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if isinstance(image_file, str):
      if image_file.startswith("http") or image_file.startswith("https"):
          response = requests.get(image_file)
          image = Image.open(BytesIO(response.content)).convert("RGB")
      else:
          image = Image.open(image_file).convert("RGB")
    elif isinstance(image_file, Image.Image):
        image = image_file
    else:
        raise ValueError(f"Unsupported image file type: {type(image_file)}")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def process_and_prepare_image(image_files, model, image_processor, device):
    images = load_images(image_files)
    images_tensor = process_images(images, image_processor, model.config)

    images_tensor_to_device = [image_tensor.to(device, dtype=torch.bfloat16) for image_tensor in images_tensor]

    image_sizes = [image.size for image in images]
    return images_tensor_to_device, image_sizes

def preprocess_text(strs, model):
    tokenized_text = QformerProcessor(text=strs, padding=True, return_tensors="pt")
    qformer_ids = tokenized_text["qformer_input_ids"].to(model.device)
    attention_mask = tokenized_text["qformer_attention_mask"].to(model.device)
    return qformer_ids, attention_mask

def eval_model(tokenizer, model, image_processor, image_files, qs, use_q, model_name, sep=',',temperature=1.0, num_beams=1, max_new_tokens=512):
    disable_torch_init()

    qformer_ids, qformerattention_mask = preprocess_text(qs, model)

    prompt = creat_prompt(qs, model, model_name)
    print(f"Prompt: {prompt}")

    images_tensor, image_sizes = process_and_prepare_image(image_files, model, image_processor, model.device) # image_files should be a str list

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            qformer_ids,
            qformerattention_mask,
            images=images_tensor,
            image_sizes=image_sizes,
            use_q=use_q,
            do_sample=True if temperature != 1.0 else False,
            temperature=temperature,
            #top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    return outputs


def model_inference(model_path, image_path, qs, use_q):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        #use_flash_attn=True,
        #    cache_dir = '' # download the model in current directory
    )
    model.to(device)

    eval_model(tokenizer, model, image_processor, [image_path], qs, use_q, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/workspace/PMC_llava-v1.6-mistral-qformer")
    parser.add_argument("--image_path", type=str, default="./images/1.jpg")
    parser.add_argument("--qs", type=str, default="Can you briefly describe this image?")
    parser.add_argument("--use_q", type=bool, default=True)
    args = parser.parse_args()

    model_inference(args.model_path, args.image_path, args.qs, args.use_q)