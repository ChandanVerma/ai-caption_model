import os
import sys
import pandas as pd
import clip
import torch
import logging
import traceback
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))
from sentence_transformers import SentenceTransformer
from src.utils.predict_utils import get_device


class SentenceEmbeddings:
    """Main class for generating sentence embeddings given some text."""

    def __init__(self, is_gpu=True):
        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")

            self.is_gpu = is_gpu
            self.device_2 = "cuda" if self.is_gpu else "cpu"

            assert self.device_2 != "cpu", self.logger.error("CPU used instead of GPU.")

            self.sent_model = SentenceTransformer(
                "./models/version_1/all-MiniLM-L6-v2"
            )
            self.sent_model.encode([""], device=self.device_2)
            self.logger.info("Sentence embedding model loaded.")

        except Exception as e:
            self.logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def reset(
        self,
    ):
        # function not used yet
        pass

    def generate_sent_embeddings(self, captions, lomotif_id):
        try:
            output_dict = {}
            output_dict["LOMOTIF_ID"] = lomotif_id
            start_time = str(pd.Timestamp.utcnow())
            output_dict["EMBED_PROCESS_START_TIME"] = start_time

            captions_joined = " ".join(list(captions.values()))

            self.logger.debug("[{}] Generating embeddings.".format(lomotif_id))
            if len(captions) > 0:
                torch.cuda.empty_cache()
                embedding = self.sent_model.encode(
                    [captions_joined], device=self.device_2
                )
                pred_time = str(pd.Timestamp.utcnow())
                self.logger.info("[{}] Done generating embeddings.".format(lomotif_id))

                output_dict["EMBED_PREDICTION_TIME"] = pred_time
                output_dict["EMBED_STATUS"] = 0

            else:
                embedding = ""
                output_dict["EMBED_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
                output_dict["EMBED_STATUS"] = 3
                self.logger.info("[{}] Length of captions = 0".format(lomotif_id))

        except Exception as e:
            embedding = ""
            output_dict["EMBED_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
            output_dict["EMBED_STATUS"] = 4
            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

        return embedding, output_dict


# if __name__ == "__main__":
#     import pickle
#     captions = pickle.load(open("sample_captions.pkl", "rb"))
#     sent = SentenceEmbeddings()
#     emb, _ = sent.generate_sent_embeddings(captions=captions,
#                                         lomotif_id='59e9aa40e8044f8fbcaaa9dc637c5d65')
#     print(emb.shape)
