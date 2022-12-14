from distutils.log import warn
import warnings
warnings.filterwarnings(action='ignore')
import os
import sys
import requests
import logging
import json
import pickle
import traceback

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

# loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")

if __name__ == "__main__":
    i = 0
    kinesis_event = pickle.load(
        open("./test/test_data/sample_event_{}.pkl".format(i), "rb")
    )
    video = kinesis_event["lomotif"]["video"]
    video_tup = os.path.splitext(video)
    new_video = "".join([video_tup[0], "-vs", video_tup[-1]])
    kinesis_event["lomotif"]["video"] = new_video

    logger.info("Reading sample data.")
    try:
        resp = requests.get(
            "http://0.0.0.0:8000/ai_caption_lomotif_pipeline",
            json=kinesis_event,
            timeout=10,
        )
        if resp.status_code == 200:
            output = resp.json()
            print(output)

        else:
            logger.error(
                "Error in rayserve tasks. Status code: {} \nTraceback: {}".format(
                    resp.status_code, resp.text
                )
            )
    except:
        assert False, logger.error(
            "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                kinesis_event["lomotif"]["id"], traceback.format_exc()
            )
        )
