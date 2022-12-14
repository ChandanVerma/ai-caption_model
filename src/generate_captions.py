import os
import sys
import clip
import numpy as np
import pandas as pd
import torch
import logging
import traceback
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))
from transformers import GPT2Tokenizer
from PIL import Image
from src.utils.predict_utils import (
    get_device,
    ClipCaptionModel,
    generate_beam,
    generate_nobeam,
)

## to be commented after testing
from src.utils.data_processing import get_mime, get_interest_frames_from_video

class CaptionImageFrames:
    """Main class for generating captions for each clip."""

    def __init__(
        self,
        max_key_frames=10,
        model_name="conceptual",
        is_gpu=True,
        model_directory="./models/version_1/",
    ):
        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")

            assert model_name in ["conceptual", "coco"], self.logger.error(
                "model_name must be conceptual or coco."
            )

            self.logger.info("Caption engine starting.")
            self.is_gpu = is_gpu
            self.model_name = model_name
            self.use_beam_search = True
            self.prefix_length = 10
            self.max_key_frames = max_key_frames

            self.device_1 = "cuda" if self.is_gpu else "cpu"
            assert self.device_1 != "cpu", self.logger.error("CPU used instead of GPU.")

            if model_name == "coco":
                model_file = "coco_weights.pt"
            if model_name == "conceptual":
                model_file = "conceptual_weights.pt"

            self.model_path = os.path.join(model_directory, model_file)
            # self.logger.info(f"Model directory: {os.listdir(os.getcwd())}")
            # self.logger.info(f"Current working directory: {os.getcwd()}")
            # self.logger.info(f"All files: {os.listdir('/app/models/')}")
            # self.logger.info(f"Model directory: {os.system('readlink -f model_file')}")
            # self.logger.info(f"Model directory: {os.getcwd()}")

            assert os.path.exists(self.model_path), self.logger.error("Model path does not exist.")

            torch.cuda.empty_cache()
            self.clip_model, self.preprocess = clip.load(
                os.path.join(model_directory, "clip/ViT-B-32.pt"),
                device=self.device_1,
                jit=False,
                # download_root=os.path.join(model_directory, "clip"),
            )
            self.logger.info("CLIP model loaded.")
            self.tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(model_directory, "gpt_tokenizer"))
            self.logger.info("GPT2 tokenizer loaded.")

            self.model = ClipCaptionModel(self.prefix_length)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device_1)
            )
            self.model = self.model.eval()
            self.model = self.model.to(self.device_1)
            self.logger.info("Caption model loaded.")

        except Exception as e:
            self.logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def reset(
        self,
    ):
        # function not used yet
        pass

    def run_img_arr(self, img_arr):
        """Run caption model on 1 image array.

        Args:
            img_arr ([np.ndarray]): image array, channel-last

        Returns:
            [str]: generated caption.
        """
        pil_image = Image.fromarray(img_arr)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device_1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device_1, dtype=torch.float32
            )
            prefix_embed = self.model.clip_project(prefix).reshape(
                1, self.prefix_length, -1
            )
        if self.use_beam_search:
            generated_text_prefix = generate_beam(
                self.model, self.tokenizer, embed=prefix_embed
            )[0]
        else:
            generated_text_prefix = generate_nobeam(
                self.model, self.tokenizer, embed=prefix_embed
            )

        return generated_text_prefix

    # def run_img_arr_batch(self, frames):
    #     """Run caption model on a batch of image arrays.

    #     Args:
    #         frames ([list]): list of image arrays, channel-last

    #     Returns:
    #         [str]: generated captions.
    #     """
    #     pil_image_stack = torch.concat(
    #         [preprocess(Image.fromarray(img_arr)).unsqueeze(0) for img_arr in frames]
    #     )
    #     pil_image_stack = pil_image_stack.to(device)
    #     with torch.no_grad():
    #         prefix = self.clip_model.encode_image(pil_image_stack).to(
    #             self.device_1, dtype=torch.float32
    #         )
    #         prefix_embed = self.model.clip_project(prefix).reshape(
    #             len(frames), self.prefix_length, -1
    #         )

    #     generated_texts = []
    #     for i in range(len(frames)):
    #         if self.use_beam_search:
    #             generated_text_prefix = generate_beam(
    #                 self.model, self.tokenizer, embed=prefix_embed[i : i + 1, ...]
    #             )[0]
    #         else:
    #             generated_text_prefix = generate_nobeam(
    #                 self.model, self.tokenizer, embed=prefix_embed[i : i + 1, ...]
    #             )

    #         generated_texts.append(generated_text_prefix)

    #     return " ".join(generated_texts)

    def get_captions(self, key_frames, lomotif_id):

        try:
            output_dict = {}
            output_dict["LOMOTIF_ID"] = lomotif_id
            start_time = str(pd.Timestamp.utcnow())
            output_dict["CAPTION_PROCESS_START_TIME"] = start_time

            # if len(key_frames) > self.max_key_frames:
            #     selected = np.array(
            #         np.arange(
            #             0, len(key_frames), len(key_frames) / self.max_key_frames
            #         ),
            #         np.int32,
            #     )
            #     selected = np.unique(selected)
            #     selected = selected[selected < len(key_frames)]
            #     key_frames = [key_frames[x] for x in selected]

            #     frame_indices = [frame_indices[x] for x in selected]

            self.logger.debug("[{}] Generating captions.".format(lomotif_id))
            captions = {}
            for i in range(len(key_frames)):
                caption = self.run_img_arr(key_frames[i])
                captions["key_frame_{}".format(i)] = caption

            pred_time = str(pd.Timestamp.utcnow())
            self.logger.debug("[{}] Done generating captions.".format(lomotif_id))

            output_dict["CAPTION_PREDICTION_TIME"] = pred_time
            output_dict["CAPTION_STATUS"] = 0

        except Exception as e:
            captions = ""
            output_dict["CAPTION_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
            output_dict["CAPTION_STATUS"] = 4
            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

        return captions, output_dict


# if __name__ == "__main__":
#     def ProcessData(save_file_name):
#         if os.path.exists(save_file_name):
#             mime = get_mime(save_file_name)
#             if mime in ["video", "image"]:
#                 (
#                     key_frames,
#                     fps,
#                     num_frames,
#                     selected_frame_indices,
#                 ) = get_interest_frames_from_video(save_file_name)              
#                 return mime, key_frames, fps, num_frames, selected_frame_indices
#             else:
#                 mime = None
#                 return mime, [], -1, -1, []

#     from src.download_lomotif import LomotifDownloader
#     # from src.process_lomotif import ProcessData

#     ld = LomotifDownloader("./downloaded_lomotifs")
#     video_url = "https://lomotif-prod.s3.amazonaws.com/lomotifs/2020/11/20/59e9aa40e8044f8fbcaaa9dc637c5d65/59e9aa40e8044f8fbcaaa9dc637c5d65-20201120-0247-video.mp4"
#     result, save_file_name = ld.download(video_url=video_url, lomotif_id='59e9aa40e8044f8fbcaaa9dc637c5d65')
#     if result:
#         # pd = ProcessData()
#         mime, key_frames, fps, video_length, indices = ProcessData(
#             save_file_name
#         )

#     # ld.remove_downloaded_file(save_file_name)
#     # ld.remove_downloads_folder()

#     if result:
#         if mime is not None:
#             cif = CaptionImageFrames()
#             captions, frame_indices = cif.get_captions(
#                 key_frames=key_frames, 
#                 # frame_indices=indices,
#                 lomotif_id='59e9aa40e8044f8fbcaaa9dc637c5d65'
#             )

#             print(captions, frame_indices)
#             # " ".join(list(captions.values()))
#             import pickle

#             pickle.dump(captions, open("sample_captions.pkl", "wb"))