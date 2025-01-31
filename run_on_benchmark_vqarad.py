import torch
import argparse
from datasets import Dataset, load_dataset
import os
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from inference import eval_model
from preprocess_data import add_image_name_and_answer_type
from peft import LoraConfig, get_peft_model, PeftModel
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_eval_on_vqa_rad(model_path, save_path, use_q=False):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        #use_flash_attn=True,
        #    cache_dir = '' # download the model in current directory
    )

    #if use_q==True:
    #    ckpt_list = ["./loraqformervqarad/VQARADQformer"]
    #else:
    #    ckpt_list = ["./2ndvqarad/2ndvqarad"]
    #for ckpt in ckpt_list:
    #    model = PeftModel.from_pretrained(model, ckpt)
    #    model = model.merge_and_unload()

    model.to(device)

    # Load the dataset
    cache_dir = "./VQA-RAD"
    VQA_RADdataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)
    test_ds = VQA_RADdataset['test'].map(add_image_name_and_answer_type)
    counter = 0
    json_data = []
    candidates = {}
    ans = []

    for row in test_ds:

        image = row['image']
        answer = row['answer']

        if row['answer_type'] == 'OPEN':
            qs = f"Based on the image, respond to this question with a short answer:{row['question']}"
            ans.append(answer)
        elif row['answer_type'] == 'CLOSED':
            qs = f"Based on the image, respond to this question with a word or phrase:{row['question']}"
            #ans.append(answer)
        else:
            raise ValueError(f"Invalid answer_type: {row['answer_type']}")

        generate_answer = eval_model(tokenizer, model, image_processor, context_len, [image], qs, use_q, model_name)
        print(f"{counter}.Img: {image}\n Question: {row['question']}\n Answer: {generate_answer}\n GT: {answer}")

        counter += 1

        new_json_data = {
            "image_name": row['image_name'],
            "question": row['question'],
            "prompt": qs,
            "generated": generate_answer,
            "answer": answer,
            "mode": "test",
            "answer_type": row['answer_type']
        }
        json_data.append(new_json_data)

    with open(os.path.join(save_path, "vqarad_prediction_answer.json"), "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"VQA-RAD generated results saved to {save_path}")

    candidates["0"] = list(set(ans))

    with open(os.path.join(save_path, "vqarad_candidates.json"), "w") as json_file:
        json.dump(candidates, json_file, indent=4)

    print(f"VQA-RAD generated candidates saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model", default="./")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results", default="./")
    parser.add_argument("--use_q", type=bool, default=False)
    args = parser.parse_args()

    run_eval_on_vqa_rad(args.model_path, args.save_path, args.use_q)


