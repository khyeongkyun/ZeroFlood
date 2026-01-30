from terramesh import build_terramesh_dataset, timestamp_to_str

from torch.utils.data import DataLoader
from datetime import datetime
import json
import argparse
import csv
from pathlib import Path
from utils import utm_crs_from_lonlat, get_info, process

parser = argparse.ArgumentParser(
    description="Check Image Quality from TerraMesh and Get metadata for the future usage.")

parser.add_argument(
    "--split",
    "-s",
    choices=["train", "val", "all"],
    default="all",
    help="Which split info are you interested? train, val, or all (default).",
)
parser.add_argument(
    "--root",
    "-r",
    default='.',
    help="Root path where dataset exist (default: current directory).",
)

if __name__ == "__main__":

    args = parser.parse_args()
    root = args.root

    if args.split == 'all':
        splits = ['train', 'val']
    else:
        splits = [args.split]

    for split in splits:

        # dataset = build_terramesh_dataset(
        #     path="/dss/dsstbyfs02/scratch/07/di54rur/zeroflood/TerraMesh",  # Streaming or local path
        #     modalities=['S1RTC', "S2L2A", 'S2RGB', "LULC"], 
        #     split=split,
        #     shuffle=False,  # Set false for split="val"
        #     batch_size=1,
        #     return_metadata=True,
        #     time_dim=True,
        # )
        # dataloader = DataLoader(dataset, 
        #                         batch_size=None, 
        #                         num_workers=1, 
        #                         persistent_workers=True, 
        #                         prefetch_factor=1)
        # get_info(dataloader, split)

        process(split, root, save_img=True, save_metadata=True)
