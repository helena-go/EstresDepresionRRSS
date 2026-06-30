#!/bin/bash
#SBATCH --job-name=bert_bose
#SBATCH --output=bert_bose-%j.out
#SBATCH --error=bert_bose-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=viz

cd $SLURM_SUBMIT_DIR

STORE=/mnt/netapp2/Store_uni/home/usc/cursos/curso1488

module load python/3.10

export PYTHONPATH=$STORE/bert_env:$PYTHONPATH

mkdir -p $STORE/logs
mkdir -p $STORE/results

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodos: $SLURM_JOB_NODELIST"
echo "Inicio: $(date)"
echo "================================"

python - <<EOF
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "==============================="
echo "Ejecutando el script de entrenamiento..."
echo "==============================="

python train_bert_vanila.py \
    --train_csv dreaddit-train.csv \
    --test_csv dreaddit-test.csv \
    --bose_train bose_train.npy \
    --bose_test bose_test.npy \
    --output_dir results/bert_bose_$(date +%Y%m%d_%H%M%S)

echo "================================"
echo "Fin del trabajo: $(date)"
echo "================================" 
   