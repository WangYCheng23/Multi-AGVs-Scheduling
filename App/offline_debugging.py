import os, sys
from matplotlib.pyplot import flag
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import uvicorn
from fastapi import FastAPI
from fastapi.websockets import WebSocket
from App.preprocessing import *
from App.postprocessing import *


app = FastAPI()

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # await websocket.send_text("accepted")
    # debug = await websocket.receive_json()
    # print(debug)
    # file_path = '/home/WangC/work/RL_SIMULATION/log/eval_log/20221117_141139/resources_log/sources'
    # name = os.listdir(file_path)
    # data = []
    # for n in name:
    #     with open(file_path+n, 'r') as f:
    #         data += json.load(f) 
    await WebSocket.send_text("haha")
    # await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        app='main:app',
        host="0.0.0.0",
        port=4040,
        # log_level = "error",
        # use_colors = 1,
        reload=True,
        # debug=True,
        # lifespan="off",
    )