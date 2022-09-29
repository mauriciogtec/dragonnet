import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2", default=0.01, type=float, nargs="+")
    ncuda = torch.cuda.device_count()
    args = parser.parse_args()


    for i, l2 in enumerate(args.l2):
        fname = f"serialjob_l2-{l2}.sh"
        dev = "cpu" if ncuda == 0 else i % ncuda

        lines = [
            "#! /usr/bin/bash",
            "sousrce ~/.bashrc",
            "conda activate cuda116",
            f"export l2={l2}",
        ]

        for scenario in range(7):
            s = f"python train.py --rdir=results_l2-${{l2}} --scenario={scenario} --l2=${{l2}} --device {dev} --silent --nseeds=200"
            lines.append(s)
        lines.append("")

        with open(fname, "w") as io:
            io.writelines("\n".join(lines))
