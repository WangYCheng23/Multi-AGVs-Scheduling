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
    await websocket.send_text("accepted")

    
    # data = await websocket.receive_json()
    # params = init_params()
    # add_env_info(params, data)
    # add_agent_info(params)
    # add_trainer_hypeparameters(params)
    # runner(params)
    # print("prepare sending")
    # await websocket.send_json(log2json(params))
    # flag = await websocket.receive_json()
    # print(flag)
    await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        app='main:app',
        host="0.0.0.0",
        port=60090,
        # log_level = "error",
        use_colors = 1,
        reload=False,
        debug=True,
        # lifespan="off",
    )