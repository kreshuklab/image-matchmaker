#!/bin/bash
#SBATCH --job-name=brain_matching
#SBATCH --partition=standard96s:shared
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00020
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem 500G
#SBATCH --mail-user=marei.freitag@zentr.uni-goettingen.de
#SBATCH --mail-type=all

source ~/.bashrc
micromamba activate matchmaker_env

cd /mnt/vast-nhr/home/mfreita/u19382/matchmaker

snakemake -s workflows/registration.smk --configfile /mnt/lustre-grete/usr/u19382/matchmaker/iso_brain_matching_config.yaml --cores 16