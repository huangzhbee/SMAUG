import argparse
import torch
import datetime
import json
import yaml
import os
import setproctitle
import pandas as pd
import ast

from main_model import Model_Shanghai
from dataset_shanghai import get_dataloader
from utils import *


parser = argparse.ArgumentParser(description="Session Genration")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/shanghai_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader, all_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"]
)


location_emb = pd.read_csv('../data/shanghai/Basestation/cluster_embedding.txt', sep='\t', header=0)
loc_emb_dict = [ast.literal_eval(location_emb.iloc[i]['embedding']) for i in range(len(location_emb))]
loc_emb_dict.append([0.0 for _ in range(32)])
loc_emb_dict = torch.tensor(np.array(loc_emb_dict)).to(args.device).float()  # 共1001个（包括无地点）

model = Model_Shanghai(config, args.device, loc_emb_dict).to(args.device)


if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

generate(model, all_loader, nsample=args.nsample, scaler=1, foldername=foldername)
