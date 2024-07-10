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
from preprocess_data import prepare_ROCO, prepare_SLAKE
from torch.optim import AdamW
from transformers import AutoProcessor, Trainer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_API_KEY"] = "Your_key"
os.environ["WANDB_PROJECT"] = "Your_project_name"


config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj", "v_proj", "k_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
    modules_to_save=["mm_projector"],
#    modules_to_save=["mm_projector", "query_tokens", "post_projection", "projection"], # FIXME: Turn on if you use qformer version
)


def tokenize_and_create_label(example_batch, image_processor, tokenizer, model, model_name, device, dataset):
    pad_token_id = tokenizer.pad_token_id
    #print(f"pad_token_id: {pad_token_id}")
    if dataset == 'VQA-RAD':
        image_files = example_batch["image"]
    elif dataset == 'SLAKE':
        image_files = example_batch["img_name"]
    else:
        raise ValueError("Dataset not supported")

    images_tensor, image_sizes = process_and_prepare_image(image_files, model, image_processor, model.device)

    tokenized_conversation_with_caption = []
    tokenized_conversation_without_caption = []
    query_list = []
    for query, answer in zip(example_batch["question"], example_batch["answer"]):

        query_list.append(query)
        prompt_without_caption = creat_prompt(query, model, model_name, None)
        prompt_with_caption = creat_prompt(query, model, model_name, answer)

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


def finetune_model(model_path, dataset='VQA-RAD'):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        use_flash_attn=True,
        #    cache_dir = '' # download the model in current directory
    )
    model.to(device)
    def transform_batch(batch):
        return tokenize_and_create_label(batch, image_processor, tokenizer, model, model_name, device, dataset)
    if dataset == 'VQA-RAD':
        train_ds, eval_ds = prepare_ROCO()
    elif dataset == 'SLAKE':
        train_ds, eval_ds = prepare_SLAKE()

    train_ds.set_transform(transform_batch)
    eval_ds.set_transform(transform_batch)

    model = get_peft_model(model, config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total param num: {total_params}")
    print(f"trainable param: {trainable_params}")
    print(f"trainable rate: {100 * trainable_params / total_params:.2f}%")

    optimizer = AdamW(model.parameters(), lr=2e-5, foreach=False)
    output_model_name = f"Finetuneon{dataset}_{model_name}"

    training_args = TrainingArguments(
        output_dir="./" + output_model_name,
        learning_rate=2e-5,
        bf16=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=False,
        save_total_limit=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=40,
        logging_steps=1,
        num_train_epochs=5,
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
    # trainer.push_to_hub()

    new_model_dir = f'./{dataset}_train_model/'
    trainer.save_model(new_model_dir)
    querytoken = model.base_model.model.model.vision_tower.query_tokens
    torch.save(querytoken, f'./{dataset}query_tokens.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Theon1130/RSV_llava-v1.6-mistral_PMC")
    parser.add_argument("--dataset", type=str, default="SLAKE")
    args = parser.parse_args()
    wandb.login()
    finetune_model(args.model_path, args.dataset)
    wandb.finish()