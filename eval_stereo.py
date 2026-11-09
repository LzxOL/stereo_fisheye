"""从 checkpoint 恢复同一相机对、深度轴和局部球面配置进行评估。"""

import argparse
import json
from types import SimpleNamespace
import torch
from module_baseline import StereoBaseline
from module_baseline.training import make_dataset, make_loader, check_device, evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", help="Only relocate data; geometry must still match the checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output", default="results/stereo_evaluation.json")
    cli = parser.parse_args()
    try:
        device = check_device(cli.device)
        checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**checkpoint["training_args"])
        args.root = cli.root if cli.root is not None else args.root
        args.device, args.batch_size, args.num_workers = cli.device, cli.batch_size, cli.num_workers
        args.augment = False
        (args.sweep_azimuth_center_deg, args.sweep_azimuth_span_deg,
         args.sweep_elevation_min_deg, args.sweep_elevation_max_deg) = checkpoint["geometry"]["angles"]
        data = make_dataset(args, cli.split)
        if not len(data):
            raise ValueError("Validation split is empty; create a validation subset or explicitly use --split test for final evaluation")
        if data.metadata != checkpoint["geometry"]:
            raise ValueError("Evaluation calibration/masks/candidate axis/sweep differs from checkpoint")
        model_config = dict(checkpoint["model_config"])
        model_config["mixed_precision"] &= device.type == "cuda"
        model = StereoBaseline(**model_config).to(device)
        state = checkpoint.get("net_state_dict", checkpoint.get("model"))
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        report = evaluate(model, make_loader(data, args), device, data.axis,
                          args.valid_iters if cli.iters is None else cli.iters, cli.max_batches)
        report.update(split=cli.split, geometry=data.metadata, checkpoint=cli.checkpoint)
        write_json(cli.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
