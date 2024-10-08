#!/usr/bin/env python

from PIL import Image
import numpy as np
import pandas as pd
import os
from tifffile import imread
from tiler import Tiler
from torchvision import transforms
import torch
import timm
import argparse
import logging

# Configure logging
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ImageCropTileFilter:
    def __init__(self, imageLoc, hf_token:str):
        self.img = imread(imageLoc)
        self.manual_count = {}
        self.h, self.w, self.channels = self.img.shape
        self.hf_token = hf_token
        self.total_pixels = self.w * self.h
        self.filtered_tiles = []
        self.cancer_type = imageLoc.split("/")[-2]
        self.image_file_name = imageLoc.split("/")[-1].split(".")[0] + '/'
        self.subid = self.image_file_name.split("_")[1].split("/")[0]
        self.temp_tile_save_path = "/home/exacloud/gscratch/CEDAR/sivakuml/ellrott-proj/temp-file.tif"

        logging.info(f"Initialized ImageCropTileFilter for {imageLoc}")

    def crop_and_tile(self):
        nrows, h_rem = divmod(self.h, 256)
        ncols, w_rem = divmod(self.w, 256)

        y = int(self.h) - h_rem
        x = int(self.w) - w_rem

        self.cropped = self.img[:y, :x, :]
        self.cropped_h, self.cropped_w, self.cropped_d = self.cropped.shape

        self.tiler = Tiler(data_shape=self.cropped.shape,
                           tile_shape=(256, 256, 3),
                           channel_dimension=None)

        logging.info(f"Cropped image: {self.cropped_w}x{self.cropped_h}")

    def other_pixel_var(self):
        self.u, count_unique = np.unique(self.tile, return_counts=True)
        tile_1d = self.tile.ravel()
        self.per_5 = np.percentile(tile_1d, 5)
        self.per_50 = np.percentile(tile_1d, 50)

    def load_gp_tile_encoder(self):
        os.environ["HF_TOKEN"] = self.hf_token

        try:
            self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            logging.info("Loaded tile encoder model successfully.")
        except Exception as e:
            logging.error(f"Error loading tile encoder: {e}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def filter_and_save(self):
        self.load_gp_tile_encoder()
        self.crop_and_tile()
        x = -256
        y = 0
        for tile_id, self.tile in self.tiler.iterate(self.cropped):
            x += 256

            if x > self.cropped_w:
                x = 256
                y += 256
            else:
                continue

            self.tile_pos = str(x) + "x_" + str(y) + "y"
            self.other_pixel_var()

            if self.u[0] < 135 and self.u[-1] >= 255 and self.per_5 < 162 and self.per_50 < 225:
                try:
                    tile_save = Image.fromarray(self.tile)
                    tile_save.save(self.temp_tile_save_path)

                    gp_input = self.transform(Image.open(self.temp_tile_save_path).convert("RGB")).unsqueeze(0)

                    self.tile_encoder.eval()
                    with torch.no_grad():
                        self.model_output = self.tile_encoder(gp_input).squeeze()

                    t_np = self.model_output.numpy()  # convert to Numpy array
                    df = pd.DataFrame(t_np)  # convert to a dataframe
                    df_transposed = df.transpose()
                    df_transposed['submitter_id'] = self.subid
                    df_transposed['cancer_type'] = self.cancer_type
                    df_transposed['tile_position'] = self.tile_pos
                    df_transposed.to_csv("/home/exacloud/gscratch/CEDAR/sivakuml/ellrott-proj/trial2.tsv",
                                         sep="\t",
                                         mode='a',
                                         index=False, header=False)  # append row to existing tsv

                    logging.info(f"Tile saved successfully: {self.tile_pos}")
                except Exception as e:
                    logging.error(f"Error processing tile {self.tile_pos}: {e}")
            else:
                logging.info(f"Tile filtered out: {self.tile_pos}")
                continue


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images and filter tiles for cancer type.")
    parser.add_argument('--image_directory',"-id", type=str, help='Path to the directory containing images', required=True)
    parser.add_argument("--hftoken", "-hf", type=str, help="Hugging Face token", required=True)

    # Parse arguments
    args = parser.parse_args()

    hugging_face_token = args.hftoken

    # Process images in the provided directory
    try:
        for cancer_type in os.listdir(args.image_directory):
            logging.info(f"Processing cancer type: {cancer_type}...")
            cancer_path = os.path.join(args.image_directory, cancer_type)  # path to original cancer type directory

            for image in os.listdir(cancer_path):
                image_path = os.path.join(cancer_path, image)  # path to original tcga wsi

                logging.info(f"Processing image: {image_path}")
                og_img = ImageCropTileFilter(image_path, hf_token=hugging_face_token)
                og_img.filter_and_save()

    except Exception as e:
        logging.error(f"Error processing images in directory {args.image_directory}: {e}")
