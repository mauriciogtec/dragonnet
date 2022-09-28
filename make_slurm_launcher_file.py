import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.01, type=float, nargs="+")
    args = parser.parse_args()

    for lr in args.lr:
        fname = f"slurmjob_lr-{lr}.sh"

        lines = [
            "#! /usr/bin/bash",
            "",
            "source ~/.bashrc",
            "conda activate cuda116",
            f"export lr={lr}",
            "",
            "#SBATCH -t 1:0:0",
            "#SBATCH -N 1",
            "#SBATCH -n 4",
            "#SBATCH -p fasse_gpu",
            # "#SBATCH --cpus-per-task 8",
            "#SBATCH --gres=gpu:4",
            # "#SBATCH --mem-per-gpu 32G",
            "#SBATCH --gpus-per-task 1",
            "#SBATCH --array=1-200",
            "",
            "conda activate cuda116",
            ""
        ]

        for scenario in range(7):
            s = f"python train.py --rdir=results_lr-${{lr}} --scenario={scenario} --seed=1${{SLURM_ARRAY_TASK_ID}}001 --lr=${{lr}} --silent"
            lines.append(s)
        lines.append("")

        with open(fname, "w") as io:
            io.writelines("\n".join(lines))
