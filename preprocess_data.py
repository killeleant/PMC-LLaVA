from datasets import Dataset, load_dataset
from PIL import Image
import pandas as pd
import os
import re
import json
from tqdm import tqdm

pre_prompt = {
        "short": "Based on the image, respond to this question with a word or phrase: ",
        "long": "Based on the image, respond to this question with a short answer: "
    }
slake_file_path = "./SLAKE/images/"
VQA_file_path="./VQA_RAD_Dataset_Public.json"

def check_images(directory):
    unidentifiable_files = []
    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        try:
            with Image.open(file_path) as img:
                img.convert('RGB')
        except Exception as e:
            unidentifiable_files.append(file_path)
    for file in unidentifiable_files:
        print(file)

def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', '', text)
    return text.lower()

def add_image_name_and_answer_type(example):
    question = normalize_text(example['question'])
    answer = normalize_text(example['answer'])
    with open(VQA_file_path, 'r', encoding='utf-8') as file:
        vqa_data = json.load(file)

    for item in vqa_data:
        if normalize_text(item['question']) == question and normalize_text(item['answer']) == answer:
            example['image_name'] = item['image_name']
            example['answer_type'] = item['answer_type']
            return example

    raise ValueError(f"No matching question-answer pair found for question: {example['question']}, answer: {example['answer']}")


def map_vqarad_data(example):

    if example["answer_type"] == "CLOSED":
        example["question"] = pre_prompt["short"] + example["question"]
    elif example["answer_type"] == "OPEN":
        example["question"] = pre_prompt["long"] + example["question"]
    else:
        raise ValueError(f"Invalid answer type: {example['answer_type']}")

    return example

def map_slake_data(example):
    example['img_name'] = os.path.join(slake_file_path, example['img_name'])

    if example["answer_type"] == "CLOSED":
        example["question"] = pre_prompt["short"] + example["question"]
    elif example["answer_type"] == "OPEN":
        example["question"] = pre_prompt["long"] + example["question"]
    else:
        raise ValueError(f"Invalid answer type: {example['answer_type']}")

    return example


# You need to download the ROCO dataset from kaggle
def prepare_ROCO(data_path):
    train_datas = os.path.join(data_path, '/train/radiologytraindata.csv')
    val_datas = os.path.join(data_path, '/validation/radiologytraindata.csv')
    train_df = pd.read_csv(train_datas)
    val_df = pd.read_csv(val_datas)

    train_prefix = os.path.join(data_path,"/train/radiology/images/")
    train_df["name"] = train_df["name"].apply(lambda x: train_prefix + x)

    val_prefix = os.path.join(data_path,"/validation/radiology/images/")
    val_df["name"] = val_df["name"].apply(lambda x: val_prefix + x)

    # convert pandas dataframe to Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(val_df)

    return train_ds, eval_ds


def prepare_VQARAD():
    cache_dir = "./VQA-RAD"
    VQA_RADdataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)
    train_datas = VQA_RADdataset['train'].map(add_image_name_and_answer_type)
    train_ds, eval_ds = train_datas.train_test_split(test_size=0.2).values()
    train_ds = train_ds.map(map_vqarad_data)
    eval_ds = eval_ds.map(map_vqarad_data)

    return train_ds, eval_ds


def prepare_SLAKE():
    cache_dir = "./SLAKE"
    slake_dataset = load_dataset("BoKelvin/SLAKE", cache_dir=cache_dir)
    train_ds = slake_dataset['train'].filter(lambda example: example['q_lang'] == 'en')
    eval_ds = slake_dataset['validation'].filter(lambda example: example['q_lang'] == 'en')

    train_ds = train_ds.map(map_slake_data)
    eval_ds = eval_ds.map(map_slake_data)

    return train_ds, eval_ds