import os
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from PIL import Image

from inference import eval_model
from llava.mm_utils import (
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model

load_dotenv()
models = {}
m_name = None


@asynccontextmanager
async def load_model(app: FastAPI):
    global m_name
    global models
    model_name = get_model_name_from_path(os.getenv("MODEL_PATH"))
    m_name = model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=os.getenv("MODEL_PATH"),
        model_base=None,
        model_name=model_name,
        use_flash_attn=True,
        #    cache_dir = '' # download the model in current directory
    )
    model.to(device)
    models[model_name] = (tokenizer, model, image_processor, context_len)
    yield
    models.clear()


app = FastAPI(lifespan=load_model)
# app = FastAPI()


@app.post("/query")
def Chat_model(
    image: Annotated[str, Form()],
    query: Annotated[str, Form()],
):
    use_q = True
    img = Image.open(image)
    print(type(img))
    # print(type(image.file))
    tokenizer, model, image_processor, context_len = models[m_name]
    res = eval_model(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        image_files=[img],
        qs=query,
        use_q=use_q,
        model_name=m_name,
    )
    return {"query": query, "response": res}


# uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
