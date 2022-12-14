"""
Sample pipeline for indexing
- Download lomotif
- Get Clip captions
- Send outputs to queue

Outputs include: \
    ai generated captions, ai generated caption embeddings,
    key frames, num_frames, fps, and other timestamps logged.
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback
import requests
import shutil
import time
import ray
import json
import boto3
from botocore.exceptions import ClientError

from ray import serve
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())) + "/serve_tasks")
sys.path.append(str(Path(os.getcwd())))

from src.download_lomotif import LomotifDownloader
from src.generate_captions import CaptionImageFrames
from src.utils.download_models import download_models_helper
from src.generate_sentence_embeddings import SentenceEmbeddings
from src.utils.generate_outputs import output_template
from src.utils.data_processing import get_mime, get_interest_frames_from_video

# comment out before commit
# from dotenv import load_dotenv

# load_dotenv("./.env")

# loggers
logger = logging.getLogger("ray")
logger.setLevel("INFO")

# Set up env variables
AWS_ROLE_ARN = os.environ.get("AWS_ROLE_ARN")
AWS_WEB_IDENTITY_TOKEN_FILE = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
# AWS_REGION = os.environ.get("AWS_REGION")

# Set up queues
results_queue_url = os.environ.get("SnowflakeResultsQueue")
raw_results_queue_url = os.environ.get("RawResultsQueue")


@serve.deployment(
    route_prefix="/download_lomotif",
    max_concurrent_queries=os.environ["DownloadMaxCon"],
    num_replicas=os.environ["DownloadNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["DownloadNumCPUPerReplica"]),
    },
)
class LomotifDownloaderServe:
    def __init__(
        self,
    ):
        try:
            self.downloader = LomotifDownloader(
                save_folder_directory="./downloaded_lomotifs"
            )
            self.num_retries = 5
            self.delay_between_retries = 30  # in seconds
            # logging.basicConfig(level=logging.INFO)
            logger.info("Downloading models from S3...")
            download_models_helper(models_path="models/")
            logger.info("All model files downloaded.")
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, video_url, lomotif_id):
        start = time.time()
        for retry_number in range(self.num_retries):
            logger.info(
                "[{}] Download retry {}/{}...".format(
                    lomotif_id, retry_number, self.num_retries
                )
            )
            result, save_file_name = self.downloader.download(
                video_url=video_url, lomotif_id=lomotif_id
            )
            if result:
                end = time.time()
                logger.info(
                    "Download complete, lomotif_id:{}, filename: {}, duration: {}".format(
                        lomotif_id, save_file_name, end - start
                    )
                )
                break
            else:
                time.sleep(self.delay_between_retries)

        return result, save_file_name


@serve.deployment(
    route_prefix="/process_lomotif",
    max_concurrent_queries=os.environ["PreprocessMaxCon"],
    num_replicas=os.environ["PreprocessNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["PreprocessNumCPUPerReplica"]),
    },
)
class PreprocessLomotifServe:
    def __init__(
        self,
    ):
        try:
            pass
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, save_file_name, lomotif_id):
        start = time.time()
        if os.path.exists(save_file_name):
            mime = get_mime(save_file_name)
            if mime in ["video", "image"]:
                (
                    key_frames,
                    fps,
                    num_frames,
                    selected_frame_indices,
                ) = get_interest_frames_from_video(save_file_name)
                logger.info("[{}] Key frames generated.".format(lomotif_id))
                end = time.time()
                logger.info(
                    "Preprocess complete, save_file_name: {}, duration: {}".format(
                        save_file_name, end - start
                    )
                )
                return mime, key_frames, fps, num_frames, selected_frame_indices
            else:
                mime = None
                logger.warning(
                    "[{}] File is not video or image. File not processed and defaults to to-be-moderated.".format(
                        lomotif_id
                    )
                )
                return mime, [], -1, -1, []


@serve.deployment(
    route_prefix="/generate_captions",
    max_concurrent_queries=os.environ["CaptionMaxCon"],
    num_replicas=os.environ["CaptionNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["CaptionNumCPUPerReplica"]),
        "num_gpus": float(os.environ["CaptionNumGPUPerReplica"]),
    },
)
class CaptionLomotifServe:
    def __init__(
        self,
    ):
        try:
            self.caption_model = CaptionImageFrames()

        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, lomotif_id):
        start = time.time()
        self.caption_model.reset()
        caption_results, caption_dict = self.caption_model.get_captions(
            key_frames=key_frames, lomotif_id=lomotif_id
        )
        end = time.time()
        logger.info(
            "CaptionLomotifServe complete, save_file_name: {}, duration: {}".format(
                save_file_name, end - start
            )
        )
        return caption_results, caption_dict


@serve.deployment(
    route_prefix="/generate_embeddings",
    max_concurrent_queries=os.environ["EmbedMaxCon"],
    num_replicas=os.environ["EmbedNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["EmbedNumCPUPerReplica"]),
        "num_gpus": float(os.environ["EmbedNumGPUPerReplica"]),
    },
)
class EmbeddingLomotifServe:
    def __init__(
        self,
    ):
        try:
            self.sent_model = SentenceEmbeddings()

        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, caption_results, save_file_name, lomotif_id):
        start = time.time()
        self.sent_model.reset()
        embed_results, embed_dict = self.sent_model.generate_sent_embeddings(
            captions=caption_results, lomotif_id=lomotif_id
        )
        end = time.time()
        logger.info(
            "EmbeddingLomotifServe complete, save_file_name: {}, duration: {}".format(
                save_file_name, end - start
            )
        )
        return embed_results, embed_dict


@serve.deployment(
    route_prefix="/ai_caption_lomotif_pipeline",
    max_concurrent_queries=os.environ["PipelineMaxCon"],
    num_replicas=os.environ["PipelineNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["PipelineNumCPUPerReplica"]),
    },
)
class CaptionPipelineServe:
    def __init__(self):
        try:
            self.download_engine = LomotifDownloaderServe.get_handle()
            self.preprocess_engine = PreprocessLomotifServe.get_handle()
            self.model_caption = CaptionLomotifServe.get_handle()
            self.model_embedding = EmbeddingLomotifServe.get_handle()
            self.sqs_client = boto3.client("sqs", region_name=AWS_DEFAULT_REGION)
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    async def __call__(self, starlette_request):
        start = time.time()
        kinesis_event = await starlette_request.json()
        message_receive_time = str(pd.Timestamp.utcnow())
        logger.info("Message received: {}".format(kinesis_event["lomotif"]["id"]))
        video = kinesis_event["lomotif"]["video"]
        lomotif_id = kinesis_event["lomotif"]["id"]

        output_dict = output_template(
            kinesis_event,
            video,
            message_receive_time,
        )

        download_result, save_file_name = await self.download_engine.remote(
            video, lomotif_id
        )

        if download_result:
            (
                mime,
                key_frames,
                fps,
                num_frames,
                selected_frame_indices,
            ) = await self.preprocess_engine.remote(save_file_name, lomotif_id)
            if mime is not None:
                if len(key_frames) == 0:
                    output_dict["CAPTION_STATUS"] = 5
                    logger.info("[{}] No key frames generated.".format(lomotif_id))
                else:
                    output_dict["KEY_FRAMES"] = ", ".join(
                        [str(x) for x in selected_frame_indices]
                    )
                    output_dict["NUM_FRAMES"] = int(num_frames)
                    output_dict["FPS"] = fps
                    logger.info(
                        "[{}] Sending request to caption model.".format(lomotif_id)
                    )
                    caption_results, caption_dict = await self.model_caption.remote(
                        key_frames, save_file_name, lomotif_id
                    )

                    logger.info("[{}] Getting caption results.".format(lomotif_id))
                    logger.info(
                        "[{}] Sending request to embedding model.".format(lomotif_id)
                    )
                    embed_results, embed_dict = await self.model_embedding.remote(
                        caption_results, save_file_name, lomotif_id
                    )

                    raw_ouput = {}
                    raw_ouput["LOMOTIF_ID"] = output_dict["LOMOTIF_ID"]
                    raw_ouput["CAPTIONS"] = " ".join(list(caption_results.values()))
                    raw_ouput["EMBEDDINGS"] = str(list(embed_results[0]))
                    raw_ouput["CREATION_TIME"] = output_dict["CREATION_TIME"]

                    logger.info("[{}] Aggregating results...".format(lomotif_id))
                    for k, v in caption_dict.items():
                        output_dict[k] = v
                    for k, v in embed_dict.items():
                        output_dict[k] = v
                    logger.info("[{}] Results aggregated.".format(lomotif_id))

            else:
                output_dict["CAPTION_STATUS"] = 1
                output_dict["EMBED_STATUS"] = 1
                raw_ouput = {}
                raw_ouput["LOMOTIF_ID"] = output_dict["LOMOTIF_ID"]
                raw_ouput["CAPTIONS"] = ""
                raw_ouput["EMBEDDINGS"] = ""
                raw_ouput["CREATION_TIME"] = output_dict["CREATION_TIME"]
                logger.info("[{}] Mime is None.".format(lomotif_id))

            os.remove(save_file_name)
            save_file_name_rewrite = (
                os.path.splitext(save_file_name)[0]
                + "-rewrite"
                + os.path.splitext(save_file_name)[1]
            )
            if os.path.exists(save_file_name_rewrite):
                os.remove(save_file_name_rewrite)

        else:
            output_dict["CAPTION_STATUS"] = 403
            output_dict["EMBED_STATUS"] = 403
            raw_ouput = {}
            raw_ouput["LOMOTIF_ID"] = output_dict["LOMOTIF_ID"]
            raw_ouput["CAPTIONS"] = ""
            raw_ouput["EMBEDDINGS"] = ""
            raw_ouput["CREATION_TIME"] = output_dict["CREATION_TIME"]
            logger.info(
                "[{}] Lomotif file does not exist or download has failed.".format(
                    lomotif_id
                )
            )

        logger.info("[{}] {}".format(lomotif_id, output_dict))
        # logger.info("[{}] {}".format(lomotif_id, raw_ouput))
        model_version_dict = {
            "VERSION": os.environ.get("version"),
            "SENTENCE_TRANSFORMERS": os.environ.get("sentence_transformers"),
            "CLIP_MODEL_VERSION": os.environ.get("clip"),
            "TOKENIZER": os.environ.get("tokenizer"),
        }
        output_dict["TOTAL_TIME_TO_PROCESS"] = round(time.time() - start, 2)
        data_attributes = output_dict.copy()
        data_attributes.update(model_version_dict)
        output_dict["DATA_ATTRIBUTES"] = data_attributes

        try:
            # Send message to SQS queue
            logger.info(
                "Attempting to send output to SQS: {}.".format(results_queue_url)
            )
            msg = json.dumps(output_dict)
            response = self.sqs_client.send_message(
                QueueUrl=results_queue_url, DelaySeconds=0, MessageBody=msg
            )
            logger.info("Sent outputs to SQS: {}.".format(results_queue_url))

            logger.info(
                "Attempting to send output to SQS: {}.".format(raw_results_queue_url)
            )
            msg = json.dumps(raw_ouput)
            response = self.sqs_client.send_message(
                QueueUrl=raw_results_queue_url, DelaySeconds=0, MessageBody=msg
            )
            end = time.time()
            logger.info("Sent raw outputs to SQS: {}.".format(raw_results_queue_url))
            logger.info(
                "ComposedModel complete, save_file_name: {}, duration: {}".format(
                    save_file_name, round(end - start, 2)
                )
            )
            return output_dict

        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script


if __name__ == "__main__":
    import torch

    # if not os.path.exists('models'):
    #     os.system('sudo mkdir models')
    #     # os.system(f'aws s3 cp s3://{AWS_BUCKET_NAME}/data_science/ai_model_files/content-captioning/ ./models/ --recursive')
    #     os.system(f'aws s3 cp s3://lomotif-datalake-prod/data_science/ai_model_files/content-captioning/ models/ --recursive')

    env_vars = {
        "AWS_ROLE_ARN": AWS_ROLE_ARN,
        "AWS_WEB_IDENTITY_TOKEN_FILE": AWS_WEB_IDENTITY_TOKEN_FILE,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        "AWS_ROLE_ARN": AWS_ROLE_ARN,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        "AWS_BUCKET_NAME": AWS_BUCKET_NAME,
        "results_queue_url": results_queue_url,
        "raw_results_queue_url": raw_results_queue_url,
        "version": os.environ.get("version"),
        "sentence_transformers": os.environ.get("sentence_transformers"),
        "clip": os.environ.get("clip"),
        "tokenizer": os.environ.get("tokenizer"),
    }
    runtime_env = {"env_vars": {}}

    for key, value in env_vars.items():
        if value is not None:
            runtime_env["env_vars"][key] = value

    ## used for debugging
    # ray.shutdown()
    # os.system('ray start --head --port=6300')

    ray.init(address="auto", namespace="serve", runtime_env=runtime_env)
    serve.start(detached=True, http_options={"host": "0.0.0.0"})
    logger.info("The environment variables in rayserve are: {}".format(runtime_env))
    logger.info("All variables are: {}".format(env_vars))
    logger.info("Starting rayserve server.")
    logger.info("Deploying modules.")

    LomotifDownloaderServe.deploy()
    PreprocessLomotifServe.deploy()
    CaptionLomotifServe.deploy()
    EmbeddingLomotifServe.deploy()
    CaptionPipelineServe.deploy()

    logger.info("Deployment completed.")
    logger.info("Waiting for requests...")
