"""En este train de bert utilizo el bert wrapper directamente del profesor sin hacer ningún cambio"""

"""Helena García Osorio - TFG - 2026

BertWrapper con BoSE

- dreaddit/dreaddit_train.csv
- dreaddit/dreaddit_test.csv
- MASK/X_train_masked.json
- MASK/X_test_masked.json
"""

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------
import os
import argparse
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from bert_wrapper_vanila import BertWithFeatures, CustomDataset

# -------------------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------------------
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "num_epochs": 3,
    "lr": 2e-5,
    "seed": 1234,
}

# -------------------------------------------------------------------------------
# TRAIN / EVALUACIÓN
# -------------------------------------------------------------------------------
def train_epoch(model, loader, bose, optimizer, criterion, device, batch_size):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(loader):
        input_ids = batch["text"].to(device)
        attention_mask = batch["attention"].to(device)
        labels = batch["label"].to(device)
        
        # BOSE
        batch_start = i * batch_size
        batch_end = batch_start + input_ids.size(0)
        
        features = torch.tensor(bose[batch_start:batch_end], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def eval_epoch(model, loader, bose, criterion, device, batch_size):
    model.eval()
    total_loss = 0
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["text"].to(device)
            attention_mask = batch["attention"].to(device)
            labels = batch["label"].to(device)
            
            # BOSE
            batch_start = i * batch_size
            batch_end = batch_start + input_ids.size(0)
            features = torch.tensor(bose[batch_start:batch_end], dtype=torch.float32).to(device)
        
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(loader)
    
    return avg_loss, all_preds, all_labels

# -------------------------------------------------------------------------------
# args
# -------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT with BoSE")
    parser.add_argument("--train_csv", type=str, default="dreaddit_train.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="dreaddit_test.csv", help="Path to test CSV")
    parser.add_argument("--bose_train", type=str, default="bose_train.npy", help="Path to BoSE train features")
    parser.add_argument("--bose_test", type=str, default="bose_test.npy", help="Path to BoSE test features")
    parser.add_argument("--output_dir", type=str, default="bert_bose_model", help="Directory to save the model")
    
    return parser.parse_args()


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main():
    args = parse_args()
    
    # 1. configuración y semilla
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # 2. Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 4. Cargar datos
    df_train = pd.read_csv(args.train_csv)
    y_train = df_train["label"].tolist()
    df_bose_train = np.load(args.bose_train)

    df_test = pd.read_csv(args.test_csv)
    y_test = df_test["label"].tolist()
    df_bose_test = np.load(args.bose_test)
    
    # -------------------------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------------------------
    indices = np.arange(len(df_train))
    
    y_full = df_train["label"].tolist()
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        stratify=y_full,
        random_state=CONFIG["seed"]
    )
    
    X_train = df_train["text"].iloc[train_idx].tolist()
    y_train = [y_full[i] for i in train_idx]
    bose_train = df_bose_train[train_idx]
    
    X_val = df_train["text"].iloc[val_idx].tolist()
    y_val = [y_full[i] for i in val_idx]
    bose_val = df_bose_train[val_idx]
    
    print(f"Datos cargados: {len(X_train)} ejemplos de entrenamiento, {len(X_val)} ejemplos de validación, {len(df_test)} ejemplos de test")
    
    # 5. Datasets y DataLoaders
    train_dataset = CustomDataset(X_train, y_train, CONFIG)
    val_dataset = CustomDataset(X_val, y_val, CONFIG)
    test_dataset = CustomDataset(df_test["text"].tolist(), y_test, CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 6. Modelo, optimizador, criterio
    
    feature_dim = bose_train.shape[1]
    model = BertWithFeatures(
        feature_dim=feature_dim,
        num_labels=len(set(y_train)),
        dropout=0.1
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    print("Modelo, optimizador y criterio inicializados.")
    
    # 7. Entrenamiento
    best_f1 = 0.0
    
    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_epoch(
            model, train_loader, bose_train,
            optimizer, criterion, device, CONFIG["batch_size"]
        )
        
        val_loss, val_preds, val_labels = eval_epoch(
            model, val_loader, bose_val,
            criterion, device, CONFIG["batch_size"]
        )
        
        f1 = f1_score(val_labels, val_preds, average="weighted")
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {f1:.4f}")
    
        # Guardar el mejor modelo
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("Mejor modelo guardado.")
        
    # 8. Evaluación final en test
    print("Evaluando en test...")
        
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
        
    _, preds, labels = eval_epoch(
            model, test_loader, df_bose_test,
            criterion, device, CONFIG["batch_size"]
        )
        
    print("Reporte de clasificación en test:")
    print(classification_report(labels, preds))
        
if __name__ == "__main__":
    main()