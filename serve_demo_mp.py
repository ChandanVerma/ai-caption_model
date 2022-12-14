import os
import sys
import requests
import logging
import json
import pickle
import traceback
import datetime
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

# loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")

df = pd.read_csv(
    "./test/test_data/test_batch_data.csv",
    usecols=["ID", "CREATED", "VIDEO", "COUNTRY"],
)
df.columns = df.columns.str.lower()
records = df.to_dict("records")
# print(records[0])


from multiprocessing import Pool


def mp_send_req(i):
    kinesis_event = {}
    kinesis_event["lomotif"] = records[i]

    # kinesis_event = pickle.load(
    #     open("./test/test_data/sample_event_{}.pkl".format(i), "rb")
    # )
    video = kinesis_event["lomotif"]["video"]
    # video_tup = os.path.splitext(video)
    # new_video = "".join([video_tup[0], "-vs", video_tup[-1]])
    # kinesis_event["lomotif"]["video"] = new_video
    resp = requests.get(
        "http://0.0.0.0:8000/ai_caption_lomotif_pipeline",
        json=kinesis_event,
        timeout=60 * 10,
    )
    if resp.status_code == 200:
        output = resp.json()
    else:
        print(i, "\n")


if __name__ == "__main__":
    start = datetime.datetime.now()
    # for x in range(3):
    #     mp_send_req(x)
    num = 40
    total = 40
    with Pool(processes=num) as pool:
        pool.starmap(mp_send_req, [(x,) for x in range(total)])
    end = datetime.datetime.now()
    print("time taken: ", ((end - start).total_seconds() / total))
