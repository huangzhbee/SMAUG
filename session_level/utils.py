import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import json


def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=20, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)

                loss.backward()

                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best cate_loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def evaluate(model, test_loader, nsample=10, scaler=1, foldername=""):
    with torch.no_grad():
        model.eval()

        all_emb_targets = []
        all_emb_samples = []

        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                emb_sample, emb_target, _ = output

                emb_sample = emb_sample.mean(dim=1)

                all_emb_targets.append(emb_target)
                all_emb_samples.append(emb_sample)

                B = emb_target.shape[0]
                evalpoints_total += B

                mse_current = ((emb_sample - emb_target) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((emb_sample - emb_target))) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no,
                        "rmse": np.sqrt(mse_total / evalpoints_total),
                        "mae": mae_total / evalpoints_total,
                    },
                    refresh=True,
                )

            print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            print("MAE:", mae_total / evalpoints_total)

            with open(
                    foldername + "/generated_outputs.pk", "wb"
            ) as f:
                all_emb_targets = torch.cat(all_emb_targets, dim=0).cpu().numpy()
                all_emb_samples = torch.cat(all_emb_samples, dim=0).cpu().numpy()

                pickle.dump(
                    [all_emb_targets, all_emb_samples],
                    f,
                )


def generate(model, all_loader, nsample=10, scaler=1, foldername=""):
    with torch.no_grad():
        model.eval()

        all_emb_targets = []
        all_emb_samples = []
        all_session_ids = []

        with tqdm(all_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                output = model.evaluate(batch, nsample)

                emb_sample, emb_target, session_id = output

                emb_sample = emb_sample.mean(dim=1)

                all_emb_targets.append(emb_target)
                all_emb_samples.append(emb_sample)
                all_session_ids.append(session_id)

                it.set_postfix(
                    ordered_dict={"batch_no": batch_no},
                    refresh=True,
                )

            with open(
                    foldername + "/generated_outputs.pk", "wb"
            ) as f:
                all_emb_targets = torch.cat(all_emb_targets, dim=0).cpu().numpy()
                all_emb_samples = torch.cat(all_emb_samples, dim=0).cpu().numpy()
                all_session_ids = torch.cat(all_session_ids, dim=0).cpu().numpy()

                pickle.dump(
                    [all_session_ids, all_emb_targets, all_emb_samples],
                    f,
                )
