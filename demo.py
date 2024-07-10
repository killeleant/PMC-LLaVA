import gradio as gr
import requests
from PIL import Image


def echo(message, history):
    """
    message["text"] is the text input from the user
    message["files"] is a list of files uploaded by the user
    """
    response = None
    files = {"file": open(message["files"][0], "rb")} if message["files"] else {}
    print("files------------------", message["files"])
    print(f"types of files: {type(files)}")
    try:
        img = Image.open(files["file"])
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"
    print(f"types of img: {type(img)}")
    try:
        response = requests.post(
            "http://localhost:8000/query",
            data={
                "query": message["text"],
                "image": message["files"][0] if message["files"][0] else None,
            },
        )
        print(f"Received response: {response.json()}")
    except requests.exceptions.RequestException as e:
        mesg = e
        print(f"An error occurred: {e}")

    print(f"Received message: {message}")

    return (
        response.json()["response"]
        if response
        else f"No response received, this is a default message, check for the errors.{mesg}"
    )


demo = gr.ChatInterface(
    fn=echo,
    title="Model Demo",
    multimodal=True,
)
demo.launch()
