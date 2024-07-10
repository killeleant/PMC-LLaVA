import wandb
import torch
import os
import random
import argparse
from torch.nn.utils.rnn import pad_sequence

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    IGNORE_INDEX,
)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from inference import creat_prompt, process_and_prepare_image, preprocess_text, QformerProcessor
from preprocess_data import prepare_ROCO
from torch.optim import AdamW
from transformers import AutoProcessor, Trainer, BitsAndBytesConfig, TrainingArguments



device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_API_KEY"] = "Your_key"
os.environ["WANDB_PROJECT"] = "Your_project_name"


concise_describe_instructions = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the given image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the provided picture.",
    "Briefly describe this image.",
]

def tokenize_and_create_label(example_batch, image_processor, tokenizer, model, model_name, device):
    pad_token_id = tokenizer.pad_token_id
    image_files = example_batch["name"]

    images_tensor, image_sizes = process_and_prepare_image(image_files, model, image_processor, model.device)

    tokenized_conversation_with_caption = []
    tokenized_conversation_without_caption = []
    query_list = []
    for caption in example_batch["caption"]:
        query = random.choice(concise_describe_instructions)
        query_list.append(query)
        prompt_without_caption = creat_prompt(query, model, model_name, None)
        prompt_with_caption = creat_prompt(query, model, model_name, caption)

        tokenized_without_caption = tokenizer_image_token(prompt_without_caption, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt")
        tokenized_with_caption = tokenizer_image_token(prompt_with_caption, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt")

        tokenized_conversation_without_caption.append(tokenized_without_caption)
        tokenized_conversation_with_caption.append(tokenized_with_caption)

    input_ids = pad_sequence([tcwc.squeeze(0) for tcwc in tokenized_conversation_with_caption], batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long().to(device)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    for i, tcwc in enumerate(tokenized_conversation_without_caption):
        input_id_without_caption = tcwc.squeeze(0)
        labels[i, len(input_id_without_caption):] = input_ids[i, len(input_id_without_caption):]

    qformer_ids_list, qformerattention_list = preprocess_text(query_list)

    inputs = {
        "input_ids": input_ids,
        "qformer_inputids": qformer_ids_list,
        "qfromer_attention_mask": qformerattention_list,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images_tensor,
        "image_sizes": image_sizes,
    }

    return inputs


def train_model_on_ROCO(model_path, data_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        use_flash_attn=True,
        #    cache_dir = '' # download the model in current directory
    )
    model.to(device)
    train_ds, eval_ds = prepare_ROCO(data_path)

    def transform_batch(batch):
        return tokenize_and_create_label(batch, image_processor, tokenizer, model, model_name, device)

    train_ds.set_transform(transform_batch)
    eval_ds.set_transform(transform_batch)

    for param in model.parameters():
        param.requires_grad = False

    # FIXME: Turn on if you use qformer version
    #for param in model.model.vision_tower.projection.parameters():
    #    param.requires_grad = True

    #model.model.vision_tower.query_tokens.requires_grad = True

    #for param in model.model.vision_tower.post_projection.parameters():
    #    param.requires_grad = True

    for param in model.model.mm_projector.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total param num: {total_params}")
    print(f"trainable param: {trainable_params}")
    print(f"trainable rate: {100 * trainable_params / total_params:.2f}%")

    optimizer = AdamW(model.parameters(), lr=4e-5, foreach=False)
    output_model_name = f"FinetuneonROCO_{model_name}"

    training_args = TrainingArguments(
        output_dir="./" + output_model_name,
        learning_rate=4e-5,
        bf16=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=False,
        save_total_limit=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=1,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        report_to="wandb",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        optimizers=(optimizer, None)
    )
    torch.set_default_dtype(torch.bfloat16)

    trainer.train()
    #trainer.push_to_hub()

    new_model_dir = './ROCO_train_model/'
    trainer.save_model(new_model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Theon1130/RSV_llava-v1.6-mistral_PMC")
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    wandb.login()
    train_model_on_ROCO(args.model_path, args.data_path)
    wandb.finish()