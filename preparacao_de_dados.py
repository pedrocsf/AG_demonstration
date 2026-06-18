import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def AlgoritmoGenetico_preparacao_de_dados():
    """Prepara os dados do dataset, retornando os splits, limites e metadados."""
    print("Baixando e Preparando Dataset Pima Indians Diabetes...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)

    X = df.drop('Outcome', axis=1).values.astype(float)
    y = df['Outcome'].values.astype(int)
    feature_names = columns[:-1]

    # Split Estratificado 80/20
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Imputação da Mediana (exclusiva do treino)
    cols_to_impute = [1, 2, 3, 4, 5]
    medians = {}

    for c in cols_to_impute:
        mask_train = X_train_raw[:, c] == 0
        X_train_raw[mask_train, c] = np.nan
        medians[c] = np.nanmedian(X_train_raw[:, c])
        X_train_raw[np.isnan(X_train_raw[:, c]), c] = medians[c]

    # Teste usa a mediana do treino para evitar vazamento de dados
    for c in cols_to_impute:
        mask_test = X_test_raw[:, c] == 0
        X_test_raw[mask_test, c] = np.nan
        X_test_raw[np.isnan(X_test_raw[:, c]), c] = medians[c]

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Limites para inicialização de 't' e desvio padrão para mutação
    t_bounds = [(np.min(X_train_scaled[:, i]), np.max(X_train_scaled[:, i])) for i in range(8)]
    stds_train = np.std(X_train_scaled, axis=0)

    # Prepara Folds para Validação Cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = list(skf.split(X_train_scaled, y_train))

    return X_train_scaled, X_test_raw, y_train, y_test, cv_folds, t_bounds, stds_train, medians, scaler, feature_names

def calcular_aptidao(s, t, d, k, w, X, y, cv_folds):
    """Calcula a aptidão (F1-score macro média) usando Validação Cruzada Estratificada. (Função Objetivo)"""
    if np.sum(s) == 0:
        return 0.0  # Indivíduos com todos os bits de seleção zero recebem aptidão zero

    # Pesos efetivos apenas das regras ativas; soma total para normalizar a inferência em [0, 1]
    active_w = w * (s == 1)
    total_weight = np.sum(active_w)
    if total_weight <= 0:
        return 0.0  # Todas as regras ativas têm peso zero: sem evidência possível

    f1_scores = []
    for train_idx, val_idx in cv_folds:
        X_val, y_val = X[val_idx], y[val_idx]
        met1 = (X_val > t) & (d == 1)
        met2 = (X_val < t) & (d == 0)
        met = (met1 | met2) & (s == 1)
        # Soma ponderada das regras satisfeitas, normalizada -> fração de evidência em [0, 1]
        score = (met * active_w).sum(axis=1) / total_weight
        preds = (score >= k).astype(int)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        f1_scores.append(f1)
    return np.mean(f1_scores)