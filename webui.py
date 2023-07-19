import os
import torch
import scipy.io.wavfile as wavf
from torch import no_grad, LongTensor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, Generator
import lib.commons as commons
import lib.utils as utils
from lib.models import SynthesizerTrn
from lib.text import text_to_sequence
import gradio as gr
import webbrowser

# 指定模型目录
MODEL_PATH="model"
# 识别是否支持CUDA，否则使用CPU推理
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 发音人音色名称
SPEAKER = "MOSS"

print(f"use device: {DEVICE}")

model = None

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, speaker_ids, sampling_rate):
    def tts_fn(text, speed):
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(DEVICE)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(DEVICE)
            sid = LongTensor([speaker_ids[SPEAKER]]).to(DEVICE)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (sampling_rate, audio)
    return tts_fn

if __name__ == "__main__":
    hps = utils.get_hparams_from_file(f"{MODEL_PATH}/config.json")

    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(DEVICE)
    _ = model.eval()

    _ = utils.load_checkpoint(f"{MODEL_PATH}/moss.pth", model, None)

    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(model, hps.speakers, hps.data.sampling_rate)
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="输入合成的文本（仅支持中文）",
                                          value="我是载人轨道空间站控制系统，550W离线版本，量子体积8192，人类迄今为止算力最强大的硬件。", elem_id=f"tts-input")
                    # select character
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                label='速度 Speed')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="输出音频", elem_id="tts-audio")
                    btn = gr.Button("合成")
                    btn.click(tts_fn,
                              inputs=[textbox, duration_slider],
                              outputs=[text_output, audio_output])

    webbrowser.open("http://127.0.0.1:5011")
    app.launch(share=False)