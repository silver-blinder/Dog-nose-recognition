import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from network_training.model import ContrastiveLoss, SiameseNetwork
from network_training.utils import DogNosePrintDataset, evaluate_model_with_metrics


@dataclass(frozen=True)
class SweepConfig:
    data_dir: str
    test_size: float
    seed: int
    batch_size: int
    epochs: int
    lr: float
    margin: float
    threshold: float
    backbone_trainable: bool
    num_workers: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(
    data_dir: str,
    test_size: float,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    train_folders, test_folders = train_test_split(folders, test_size=test_size, random_state=seed)

    train_ds = DogNosePrintDataset(root_dir=data_dir, transform=transform, folders=train_folders)
    test_ds = DogNosePrintDataset(root_dir=data_dir, transform=transform, folders=test_folders)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl, test_dl


def configure_model(backbone_trainable: bool) -> SiameseNetwork:
    model = SiameseNetwork()
    for p in model.backbone.parameters():
        p.requires_grad = backbone_trainable
    return model


def train_one(
    cfg: SweepConfig,
    out_dir: Path,
) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = get_device()

    train_dl, test_dl = build_dataloaders(
        data_dir=cfg.data_dir,
        test_size=cfg.test_size,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = configure_model(cfg.backbone_trainable).to(device)
    criterion = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    per_epoch_path = out_dir / "per_epoch.csv"
    with per_epoch_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "test_loss", "test_acc"],
        )
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            for img1, img2, labels in train_dl:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(img1, img2).view(-1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running += loss.item() * img1.size(0)

            train_loss = running / len(train_dl.dataset)
            test_loss, test_acc = evaluate_model_with_metrics(
                model,
                test_dl,
                criterion,
                threshold=cfg.threshold,
                device=device,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "test_loss": f"{test_loss:.6f}",
                    "test_acc": f"{test_acc:.6f}",
                }
            )
            print(
                f"[epoch {epoch:02d}/{cfg.epochs}] "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    # Save model
    torch.save(model.state_dict(), out_dir / "siamese_network.pth")

    # Best (last) metrics for summary
    return {"test_loss": float(test_loss), "test_acc": float(test_acc)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Siamese param sweep (ablation) runner")
    p.add_argument("--data-dir", required=True, help="Path to dir_train (class folders)")
    p.add_argument("--out-root", default="network_training/outputs", help="Output root directory")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--batch-sizes", default="16,32", help="Comma-separated batch sizes")
    p.add_argument("--lrs", default="1e-4,1e-3", help="Comma-separated learning rates")
    p.add_argument("--margins", default="0.5,1.0,2.0", help="Comma-separated contrastive margins")
    p.add_argument("--thresholds", default="0.5", help="Comma-separated decision thresholds")
    p.add_argument(
        "--backbone-trainable",
        default="false,true",
        help="Comma-separated booleans controlling backbone finetuning",
    )
    return p.parse_args()


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_bools(s: str) -> List[bool]:
    out: List[bool] = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif x in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool: {x}")
    return out


def write_markdown_report(out_root: Path, rows: List[Dict[str, str]]) -> None:
    md = out_root / "comparison.md"
    lines = []
    lines.append("# 参数对照实验结果\n")
    lines.append("本目录为同一训练/评估协议下，不同超参数组合的对照结果汇总。\n")
    lines.append("## 汇总表\n")
    lines.append("| run_id | batch_size | lr | margin | threshold | backbone_trainable | test_acc | test_loss |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| `{r['run_id']}` | {r['batch_size']} | {r['lr']} | {r['margin']} | {r['threshold']} | {r['backbone_trainable']} | {r['test_acc']} | {r['test_loss']} |\n"
        )
    md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    batch_sizes = parse_csv_ints(args.batch_sizes)
    lrs = parse_csv_floats(args.lrs)
    margins = parse_csv_floats(args.margins)
    thresholds = parse_csv_floats(args.thresholds)
    backbone_trainable_list = parse_csv_bools(args.backbone_trainable)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_root = out_root / f"sweep-{stamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    summary_path = sweep_root / "summary.csv"
    rows: List[Dict[str, str]] = []

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "batch_size",
                "lr",
                "margin",
                "threshold",
                "backbone_trainable",
                "test_acc",
                "test_loss",
            ],
        )
        writer.writeheader()

        run_idx = 0
        for bs in batch_sizes:
            for lr in lrs:
                for margin in margins:
                    for thr in thresholds:
                        for bb_trainable in backbone_trainable_list:
                            run_idx += 1
                            run_id = f"run-{run_idx:03d}"
                            run_dir = sweep_root / run_id
                            run_dir.mkdir(parents=True, exist_ok=True)

                            cfg = SweepConfig(
                                data_dir=args.data_dir,
                                test_size=args.test_size,
                                seed=args.seed,
                                batch_size=bs,
                                epochs=args.epochs,
                                lr=lr,
                                margin=margin,
                                threshold=thr,
                                backbone_trainable=bb_trainable,
                                num_workers=args.num_workers,
                            )
                            (run_dir / "config.json").write_text(
                                json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )

                            print(f"\n=== {run_id} ===")
                            metrics = train_one(cfg, run_dir)

                            row = {
                                "run_id": run_id,
                                "batch_size": str(bs),
                                "lr": f"{lr:g}",
                                "margin": f"{margin:g}",
                                "threshold": f"{thr:g}",
                                "backbone_trainable": str(bb_trainable).lower(),
                                "test_acc": f"{metrics['test_acc']:.6f}",
                                "test_loss": f"{metrics['test_loss']:.6f}",
                            }
                            writer.writerow(row)
                            rows.append(row)

    write_markdown_report(sweep_root, rows)
    print(f"\nSaved sweep results to: {sweep_root}")


if __name__ == "__main__":
    main()

