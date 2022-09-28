import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.01, type=float, nargs="+")
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    for lr in args.lr:
        fname = f"slurmjob_lr-{lr}-serial.sh"

        lines = [
            "#! /usr/bin/bash",
            "source ~/.bashrc",
            "conda activate cuda116",
            f"export lr={lr}",
        ]

        for scenario in range(7):
            # start = "001" if scenario != 4 else "190"
            start = "001"
            s = f"for i in {{{start}..200}}; do python train.py --rdir=results_lr-${{lr}} --scenario={scenario} --seed=1${{i}}001 --lr=${{lr}} --device {args.device} --silent; done"
            lines.append(s)
        lines.append("")

        with open(fname, "w") as io:
            io.writelines("\n".join(lines))
