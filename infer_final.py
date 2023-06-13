# coding=utf-8
# Implement stream chat in command line for ChatGLM fine-tuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py


import os
import signal
import platform
import sys

sys.path.append("ChatGLM-Efficient-Tuning/src")
from utils import ModelArguments, load_pretrained
from transformers import HfArgumentParser

sys.path.append("../..")
from wenda2.plugins.zhishiku_rtst import find

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False
welcome = "欢迎使用 ChatGLM-6B-history 模型"


# 输入内容即可对话，clear清空对话历史，stop终止程序
def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


parser = HfArgumentParser(ModelArguments)
model_args, = parser.parse_args_into_dataclasses(args="--checkpoint_dir /kaggle/working/model/ChatGLM-Efficient"
                                                      "-Tuning/checkpoint4/checkpoint-5000")
model, tokenizer = load_pretrained(model_args)
model = model.cuda()
model.eval()


def infer_final(query):
    global stop_stream

    history = []
    resultJSON = find(query)
    result = ""
    for item in resultJSON:
        result += item["content"]
    query = ("请你回答一个问题，下面是一些可能相关的信息。" + result + " 问题如下 "+ query)

    count = 0
    for _, history in model.stream_chat(tokenizer, query, history=history):
        if stop_stream:
            stop_stream = False
            break
        else:
            count += 1
            if count % 8 == 0:
                signal.signal(signal.SIGINT, signal_handler)
    if history:
        return history[-1][-1]
    return "unknown error"


if __name__ == "__main__":
    print(infer_final("中国古代第一个爱国诗人是（）"))
