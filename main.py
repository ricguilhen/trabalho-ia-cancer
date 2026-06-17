"""
============================================================================
 Trabalho Final - Disciplina de Inteligencia Artificial
 Professor Munif Gebara Junior - Unicesumar - 2026

 Tema: Classificacao de Tumores de Mama (Maligno x Benigno)
 Metodos: SVM (Parte 1)  e  Rede Neural Artificial / RNA (Parte 2)

 Integrante:
   - Ricardo Guilhen Melo - RA: 23013569-2
============================================================================

VISAO GERAL DO PIPELINE
-----------------------
 1. Carrega o dataset (CSV) de exames de tumores de mama.
 2. Pre-processa e DIVIDE em treino/teste de forma estratificada.
 3. Normaliza os atributos (StandardScaler).
 4. BALANCEIA as classes do conjunto de treino (oversampling).
 5. Treina o modelo da Parte 1 (SVM) e o da Parte 2 (RNA).
 6. Avalia com metricas e matriz de confusao (detalhando VP/VN/FP/FN).
 7. Gera os graficos e salva os modelos treinados.
"""

import os
import pickle                       # salva/carrega os modelos treinados em arquivo
import warnings
# A RNA e treinada epoca por epoca (max_iter=1), o que gera avisos de
# "convergencia" esperados e inofensivos. Silenciamos para a saida ficar limpa.
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # backend que salva imagem sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample          # usado no balanceamento (oversampling)
from sklearn.svm import SVC                  # PARTE 1 - SVM
from sklearn.neural_network import MLPClassifier   # PARTE 2 - Rede Neural (MLP)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, log_loss
)

SEED = 42                          # fixa a aleatoriedade -> resultados reproduziveis
os.makedirs("graficos", exist_ok=True)
os.makedirs("modelos", exist_ok=True)
COR_SVM, COR_RNA = "#2E86C1", "#E67E22"


# ===========================================================================
# 1. CARREGAMENTO DO DATASET
# ===========================================================================
print("=" * 60)
print("1. CARREGANDO O DATASET")
print("=" * 60)

df = pd.read_csv("cancer_mama.csv")
print(f"   Registros (exames): {df.shape[0]}")
print(f"   Atributos          : {df.shape[1] - 1} preditores + 1 alvo")

# Verificacao de balanceamento das classes ANTES de qualquer tratamento
n_benigno = (df["diagnostico"] == 0).sum()
n_maligno = (df["diagnostico"] == 1).sum()
print("\n   Distribuicao do alvo (o dataset E desbalanceado):")
print(f"     Benigno (0): {n_benigno}  ({n_benigno/len(df)*100:.1f}%)")
print(f"     Maligno (1): {n_maligno}  ({n_maligno/len(df)*100:.1f}%)")


# ===========================================================================
# 2. PRE-PROCESSAMENTO E DIVISAO TREINO/TESTE
# ===========================================================================
print("\n" + "=" * 60)
print("2. PRE-PROCESSAMENTO E DIVISAO TREINO/TESTE")
print("=" * 60)

X = df.drop(columns=["diagnostico"])   # atributos preditores
y = df["diagnostico"]                  # variavel alvo (0=benigno, 1=maligno)

# DIVISAO ESTRATIFICADA: stratify=y mantem a MESMA proporcao de classes tanto
# no conjunto de TREINO quanto no de TESTE. Ou seja, ja garante que treino e
# teste estejam "balanceados na mesma proporcao" um em relacao ao outro.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"   Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# NORMALIZACAO: coloca todos os atributos na mesma escala (media 0, desvio 1).
# O scaler aprende a escala APENAS com o treino (fit_transform) e apenas
# aplica essa mesma escala no teste (transform). Isso evita data leakage
# (vazamento de informacao do teste para o treino).
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# ===========================================================================
# 3. BALANCEAMENTO DAS CLASSES (somente no TREINO)
# ===========================================================================
print("\n" + "=" * 60)
print("3. BALANCEAMENTO DAS CLASSES (oversampling no treino)")
print("=" * 60)

def balancear_classes(X_dados, y_dados, semente=SEED):
    """
    Balanceia as classes por OVERSAMPLING da classe minoritaria.

    Como funciona: descobre qual classe tem menos exemplos e sorteia copias
    dela (com reposicao) ate que as duas classes fiquem com o MESMO numero de
    amostras. Assim o modelo nao aprende a "chutar" sempre a classe majoritaria.

    Importante: e aplicado SO no conjunto de TREINO. O conjunto de TESTE e
    mantido na proporcao real, para a avaliacao refletir o mundo real.
    """
    y_dados = np.asarray(y_dados)
    classes, contagens = np.unique(y_dados, return_counts=True)
    n_maior = contagens.max()                # tamanho da classe majoritaria

    X_bal_list, y_bal_list = [], []
    for c in classes:
        idx = np.where(y_dados == c)[0]      # indices das amostras da classe c
        X_c = X_dados[idx]
        y_c = y_dados[idx]
        if len(idx) < n_maior:               # se for a classe minoritaria...
            X_c, y_c = resample(             # ...sorteia copias ate igualar
                X_c, y_c, replace=True,
                n_samples=n_maior, random_state=semente
            )
        X_bal_list.append(X_c)
        y_bal_list.append(y_c)

    X_bal = np.vstack(X_bal_list)
    y_bal = np.concatenate(y_bal_list)
    # embaralha para nao ficar todas as classes em blocos
    ordem = np.random.RandomState(semente).permutation(len(y_bal))
    return X_bal[ordem], y_bal[ordem]

print(f"   Antes  -> Benigno(0): {(y_train==0).sum()} | Maligno(1): {(y_train==1).sum()}")
X_train_bal, y_train_bal = balancear_classes(X_train_sc, y_train)
print(f"   Depois -> Benigno(0): {(y_train_bal==0).sum()} | Maligno(1): {(y_train_bal==1).sum()}")
print("   (as duas classes ficaram com o mesmo numero de amostras no treino)")


# ===========================================================================
# 4. PARTE 1 - SVM (Support Vector Machine)
# ===========================================================================
print("\n" + "=" * 60)
print("4. TREINANDO SVM (Parte 1)")
print("=" * 60)

# O SVM procura o "hiperplano" que separa as duas classes com a maior margem
# possivel. O kernel RBF permite separar classes que nao sao linearmente
# separaveis. C controla o quanto o modelo tolera erros de classificacao.
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)
svm.fit(X_train_bal, y_train_bal)          # treina com os dados JA balanceados
y_pred_svm = svm.predict(X_test_sc)        # preve no teste (proporcao real)

with open("modelos/svm_modelo.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("modelos/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("   Modelo SVM treinado e salvo em modelos/svm_modelo.pkl")


# ===========================================================================
# 5. PARTE 2 - RNA (Rede Neural Artificial / MLP)
# ===========================================================================
print("\n" + "=" * 60)
print("5. TREINANDO RNA - Rede Neural (Parte 2)")
print("=" * 60)

# ARQUITETURA DA REDE:
#   Entrada: 30 neuronios (um para cada atributo do exame)
#   Camada oculta 1: 32 neuronios, ativacao ReLU
#   Camada oculta 2: 16 neuronios, ativacao ReLU
#   Saida: 1 neuronio (probabilidade de ser maligno)
# Otimizador Adam, treinada por varias "epocas" (passagens pelos dados).
#
# Treinamos epoca por epoca (max_iter=1 + warm_start=True) para conseguir
# registrar a perda (loss) a cada epoca e desenhar a curva de perda.
rna = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1,
    warm_start=True,
    random_state=SEED
)

perda_treino, perda_teste = [], []
N_EPOCAS = 200
for epoca in range(N_EPOCAS):
    rna.fit(X_train_bal, y_train_bal)            # treina 1 epoca a mais
    perda_treino.append(rna.loss_)               # perda no treino
    prob_teste = rna.predict_proba(X_test_sc)
    perda_teste.append(log_loss(y_test, prob_teste))  # perda no teste

y_pred_rna = rna.predict(X_test_sc)

with open("modelos/rna_modelo.pkl", "wb") as f:
    pickle.dump(rna, f)
print(f"   Rede treinada por {N_EPOCAS} epocas. Arquitetura: 30 -> 32 -> 16 -> 1")
print("   Modelo RNA salvo em modelos/rna_modelo.pkl")


# ===========================================================================
# 6. AVALIACAO - METRICAS E MATRIZ DE CONFUSAO
# ===========================================================================
print("\n" + "=" * 60)
print("6. AVALIACAO DOS MODELOS")
print("=" * 60)

def explicar_matriz(nome, y_real, y_pred):
    """Imprime a matriz de confusao e explicita VP, VN, FP e FN, mostrando
    de onde saem as metricas. (Classe positiva = 1 = MALIGNO.)"""
    cm = confusion_matrix(y_real, y_pred)   # [[VN, FP], [FN, VP]]
    vn, fp, fn, vp = cm.ravel()
    print(f"\n   --- {nome} ---")
    print(f"   Verdadeiro Negativo (VN): {vn}  (benigno classificado como benigno)")
    print(f"   Falso Positivo      (FP): {fp}  (benigno classificado como maligno)")
    print(f"   Falso Negativo      (FN): {fn}  (MALIGNO classificado como benigno) <- pior erro")
    print(f"   Verdadeiro Positivo (VP): {vp}  (maligno classificado como maligno)")
    return {
        "Modelo": nome,
        "Acuracia": accuracy_score(y_real, y_pred) * 100,
        "Acuracia_Balanceada": balanced_accuracy_score(y_real, y_pred) * 100,
        "Precisao": precision_score(y_real, y_pred) * 100,
        "Revocacao": recall_score(y_real, y_pred) * 100,
        "F1": f1_score(y_real, y_pred) * 100,
    }

m_svm = explicar_matriz("SVM (Parte 1)", y_test, y_pred_svm)
m_rna = explicar_matriz("RNA (Parte 2)", y_test, y_pred_rna)

print("\n   " + "-" * 52)
print(f"   {'Metrica':<22}{'SVM (P1)':>12}{'RNA (P2)':>14}")
print("   " + "-" * 52)
for chave, rotulo in [("Acuracia","Acuracia"), ("Acuracia_Balanceada","Acur. Balanceada"),
                      ("Precisao","Precisao"), ("Revocacao","Revocacao"), ("F1","F1-Score")]:
    print(f"   {rotulo:<22}{m_svm[chave]:>11.2f}%{m_rna[chave]:>13.2f}%")
print("   " + "-" * 52)


# ===========================================================================
# 7. GRAFICOS
# ===========================================================================
print("\n" + "=" * 60)
print("7. GERANDO GRAFICOS")
print("=" * 60)
cats = ["Acuracia", "Precisao", "Revocacao", "F1"]
rot_cats = ["Acuracia", "Precisao", "Revocacao", "F1-Score"]

# --- Grafico 1: distribuicao do alvo (antes x depois do balanceamento) ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["Benigno (0)", "Maligno (1)"], [n_benigno, n_maligno],
            color=[COR_SVM, COR_RNA], alpha=0.85)
axes[0].set_title("Dataset original (desbalanceado)", fontweight="bold")
axes[0].set_ylabel("Quantidade")
for i, v in enumerate([n_benigno, n_maligno]):
    axes[0].text(i, v + 4, str(v), ha="center", fontweight="bold")

nb, nm = (y_train_bal == 0).sum(), (y_train_bal == 1).sum()
axes[1].bar(["Benigno (0)", "Maligno (1)"], [nb, nm],
            color=[COR_SVM, COR_RNA], alpha=0.85)
axes[1].set_title("Treino apos balanceamento", fontweight="bold")
axes[1].set_ylabel("Quantidade")
for i, v in enumerate([nb, nm]):
    axes[1].text(i, v + 4, str(v), ha="center", fontweight="bold")
plt.suptitle("Distribuicao da Variavel Alvo", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graficos/01_distribuicao_alvo.png", dpi=150)
plt.close()
print("   OK graficos/01_distribuicao_alvo.png")

# --- Grafico 2: comparacao de metricas ---
fig, ax = plt.subplots(figsize=(9, 5))
vals_s = [m_svm[c] for c in cats]
vals_r = [m_rna[c] for c in cats]
x = np.arange(len(cats)); w = 0.35
b1 = ax.bar(x - w/2, vals_s, w, label="SVM (Parte 1)", color=COR_SVM, alpha=0.9)
b2 = ax.bar(x + w/2, vals_r, w, label="RNA (Parte 2)", color=COR_RNA, alpha=0.9)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("Comparacao de Metricas - SVM vs RNA", fontsize=13, fontweight="bold")
ax.set_ylabel("Valor (%)"); ax.set_ylim(0, 110)
ax.set_xticks(x); ax.set_xticklabels(rot_cats)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/02_comparacao_metricas.png", dpi=150)
plt.close()
print("   OK graficos/02_comparacao_metricas.png")

# --- Grafico 3: matrizes de confusao ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y_pred, titulo, cor in [
    (axes[0], y_pred_svm, "SVM (Parte 1)", COR_SVM),
    (axes[1], y_pred_rna, "RNA (Parte 2)", COR_RNA),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False,
                cmap=sns.light_palette(cor, as_cmap=True),
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"], linewidths=0.5)
    ax.set_title(f"Matriz de Confusao - {titulo}", fontweight="bold")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("graficos/03_matrizes_confusao.png", dpi=150)
plt.close()
print("   OK graficos/03_matrizes_confusao.png")

# --- Grafico 4: curva de perda da RNA ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(perda_treino, label="Perda Treino", color=COR_RNA, linewidth=2)
ax.plot(perda_teste, label="Perda Teste", color=COR_SVM, linewidth=2, linestyle="--")
ax.set_title("Curva de Perda - RNA (Parte 2)", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoca"); ax.set_ylabel("Loss (Log Loss)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/04_curva_perda_rna.png", dpi=150)
plt.close()
print("   OK graficos/04_curva_perda_rna.png")

print("\n" + "=" * 60)
melhor = "RNA (Parte 2)" if m_rna["F1"] > m_svm["F1"] else "SVM (Parte 1)"
print(f"   Melhor modelo pelo F1-Score: {melhor}")
print("   Graficos em graficos/  |  Modelos em modelos/")
print("=" * 60)
