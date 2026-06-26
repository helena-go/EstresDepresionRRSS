"""TRANSFERENCIA DE COÑECEMENTO DO DOMINIO DE ESTRÉS AO DOMINIO DA DEPRESIÓN"""

"""Helena García Osorio - 2025/2026 - TFG - Grao en Intelixencia Artificial - Universidade de Santiago de Compostela"""

"""EXPLICACIÓN
--------------------------------
Dado o noso modelo BERT preentrenado con Dreaddit no dominio de estres,
o obxectivo é facer inferencia co dominio de depresión, para poder ver se realmente existe unha relación lingüística entre ambos dominios,
o modelo debe poder clasificar como estrés aqueles posts que teñan unha etiqueta de depresión, ademais, según o nivel de depresión a probabilidade
de ser clasificado coma estrés tamén debería aumentar, é dicir, os posts con depresión leve deberían ser clasificados coma estrés con menor probabilidade que os posts con depresión grave.

  1. Carga os posts de depresión (df_nivelPosts_etiquetado.csv)
  2. Pasaos polo modelo de estrés 
  3. Obtén prob_estres por post (softmax sobre los logits)
  4. Agrega por usuario (id_subject) a probabilidade media de estrés
  5. Cruza co nivel de depresión real (label_depresion 0-3)
  6. Guarda dos CSVs:
       - depression_posts_stress_probs.csv  → nivel post
       - depression_users_stress_probs.csv  → nivel usuario (agregado)
"""


""""""

"""imports necesarios"""
from email import parser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from bert_wrapper_vanila import BertWithFeatures, CustomDataset

# -------------------------------------------------------------------------------
# Configuración — igual que en entrenamiento
# -------------------------------------------------------------------------------
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "seed": 1234,
}

# -------------------------------------------------------------------------------
# Función principal de inferencia
# -------------------------------------------------------------------------------

def inference(model, loader,bose, device):
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["text"].to(device)
            attention_mask = batch["attention"].to(device)
            
            start = i * CONFIG['batch_size']
            end = start + input_ids.size(0)
            
            # Tenemos bose en sparse, así que convertimos a denso
            features = torch.tensor(bose[start:end].toarray(), dtype=torch.float32
                                    ).to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.argmax(all_probs, axis=1)
    return all_probs, all_preds

# -------------------------------------------------------------------------------
# args
# -------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transferencia")
    parser.add_argument("--depresion_csv", type=str, default="df_nivelPosts_etiquetado.csv", help="CSV con los posts de depresión")
    parser.add_argument("--bose_depression", type=str, default="bose_depression_sparse.npz", help="BoSE de depresión en sparse (.npz), generado con el TF-IDF de estrés")
    parser.add_argument("--model_dir",       type=str, default="bert_bose_model", help="Directorio con best_model.pt")
    parser.add_argument("--num_labels",      type=int, default=2, help="Número de clases del modelo de estrés (2: estres/no-estres)")
    parser.add_argument("--output_dir",      type=str, default="results_transfer")
    return parser.parse_args()
    
# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------
def main():
    args = parse_args()
    
    # 1. configuración y semilla
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # 2. Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Cargamos datos de depresión
    df = pd.read_csv(args.depresion_csv)
    df["text"] = df["text"].fillna("").astype(str)
    print(f"  Posts: {len(df)} | Usuarios: {df['id_subject'].nunique()}")
    print(f"  Distribución label_depresion:\n{df['label_depresion'].value_counts().sort_index()}\n")
    
    # 4. Cargamos BoSE de depresión (sparse)
    print(f"Cargando BoSE de depresión (sparse) desde: {args.bose_depression}")
    bose = sparse.load_npz(args.bose_depression)
    print(f"  BoSE shape: {bose.shape} | Tipo: {type(bose)}\n")
    
    feature_dim = bose.shape[1]
    
    # 5. Cargamos modelo BERT+BoSE entrenado en estrés
    print(f"Cargando modelo BERT+BoSE desde: {args.model_dir}")
    model_path = os.path.join(args.model_dir, "best_model.pt")
    model = BertWithFeatures(CONFIG["model_name"], feature_dim, args.num_labels, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Modelo cargado correctamente.\n")
    
    # 6. Dataset y DataLoader
    dataset = CustomDataset(df["text"].tolist(), bose, df["label_depresion"].tolist(), CONFIG["max_length"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 7. Inferencia
    print("Realizando inferencia...")
    all_probs, all_preds = inference(model, loader, bose, device)
    df["prob_no_estres"] = all_probs[:, 0]  # Probabilidad de no ser estrés
    df["prob_estres"] = all_probs[:, 1]  # Probabilidad de ser estrés
    df["pred_estres"] = all_preds

    print("Inferencia completada.\n")
    
    # 8. Guardar resultados
    # A nivel de POSTS


    posts_out = os.path.join(args.output_dir, "posts_depression_probs.csv")
    df[["id_subject", "label_depresion", "pred_estres", "prob_estres", "prob_no_estres"]].to_csv(
        posts_out, index=False
    )
    print(f"CSV posts guardado: {posts_out}")
    
    # A nivel de USUARIOS (agregando por id_subject)
    df_user = df.groupby("id_subject").agg(
        nivel_depresion  = ("label_depresion", "first"), # asumimos que el nivel de depresión es el mismo para todos los posts de un usuario
        prob_estres_mean = ("prob_estres", "mean"), # probabilidad media de estrés por usuario
        prob_estres_max  = ("prob_estres", "max"), # probabilidad máxima de estrés por usuario
        pred_estres_pct  = ("pred_estres", "mean"),  #% posts clasificados como estrés
        n_posts          = ("pred_estres", "count"), # número de posts por usuario
    ).reset_index()

    users_out = os.path.join(args.output_dir, "users_depression_probs.csv")
    df_user.to_csv(users_out, index=False)
    print(f"CSV usuarios guardado: {users_out}")
  
    """# 9. Resumen
    print("Probabilidad de estrés por nivel de depresión a nivel de POSTS:")
    print(df.groupby("label_depresion")["prob_estres"].describe())
    
    print("\nProbabilidad de estrés por nivel de depresión a nivel de USUARIOS:")
    print(df_user.groupby("nivel_depresion")["prob_estres_mean"].describe())
    
    # Reporte de clasificación a nivel de posts
    if "grupo" in df.columns:
        df["label"] = df["grupo"].map({"control": 0, "positivo": 1})
        print("\nReporte de clasificación (grupo como ground truth):")
        print(classification_report(df["label"], df["pred_estres"]))
    else:
        print("\nNo se encontró la columna 'grupo' para el reporte de clasificación.")"""
        
if __name__ == "__main__":
    main()