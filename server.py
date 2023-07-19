import setproctitle
import uvicorn
import traceback
import os
import re
import uuid
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

# 设置进程名称
setproctitle.setproctitle("moss-tts-service")

# 指定模型目录
MODEL_PATH="model"
# 识别是否支持CUDA，否则使用CPU推理
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 发音人音色名称
SPEAKER = "MOSS"

print(f"use device: {DEVICE}")

model = None
speaker_ids = {}
sampling_rate = None


if os.path.exists("tmp") is False:
    os.mkdir("tmp")

def prepare_text(text):
    return re.sub(r"moss", "mos", text, flags=re.I)

def get_text(text, hps, is_symbol):
    text = prepare_text(text)
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def predict(text, speed):
    stn_tst = get_text(text, hps, False)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(DEVICE)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(DEVICE)
        sid = LongTensor([speaker_ids[SPEAKER]]).to(DEVICE)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
            length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    torch_gc()
    return audio

app = FastAPI()

class TextToSpeechParams(BaseModel):
    text: str
    speechRate: Optional[float] = 1

@app.post('/text_to_speech')
async def text_to_speech(params: TextToSpeechParams):
    tmp_file_path = f"tmp/{uuid.uuid4()}.wav"
    audio = predict(params.text, params.speechRate)
    wavf.write(tmp_file_path, sampling_rate, audio)
    async def generate() -> Generator:
        with open(tmp_file_path, "rb") as file_like:
            yield file_like.read()
        os.remove(tmp_file_path)
    return StreamingResponse(generate(), media_type="audio/mp3")

def handle_exception(request: Request, e: Exception):
    err = {
        "error": type(e).__name__,
        "detail": vars(e).get('detail', ''),
        "body": vars(e).get('body', ''),
        "errors": str(e),
    }
    print(f"API error: {request.method}: {request.url} {err}")
    if not isinstance(e, HTTPException): # do not print backtrace on known httpexceptions
        traceback.print_exc()
    return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

@app.middleware("http")
async def exception_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return handle_exception(request, e)

@app.exception_handler(Exception)
async def fastapiExceptionHandler(request: Request, e: Exception):
    return handle_exception(request, e)

@app.exception_handler(HTTPException)
async def httpExceptionHandler(request: Request, e: HTTPException):
    return handle_exception(request, e)

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

    speaker_ids = hps.speakers
    sampling_rate = hps.data.sampling_rate
    uvicorn.run(app, host="0.0.0.0", port=5010, log_config="uvicorn_config.json")