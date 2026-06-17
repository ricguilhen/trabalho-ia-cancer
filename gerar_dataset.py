"""
gerar_dataset.py
----------------
Exporta o dataset "Breast Cancer Wisconsin" (do scikit-learn) para um CSV,
deixando-o disponivel no repositorio. Assim quem clonar o projeto consegue
reproduzir o trabalho sem precisar baixar nada da internet.

O dataset reune medidas calculadas a partir de imagens digitalizadas de
exames de nucleos celulares de tumores de mama. Cada linha e um exame e o
objetivo e classificar o tumor como MALIGNO (1) ou BENIGNO (0).
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

dados = load_breast_cancer(as_frame=True)
df = dados.frame.copy()

# No scikit-learn o alvo vem como 0=maligno e 1=benigno. Aqui invertemos para
# 1=MALIGNO e 0=BENIGNO, porque o caso de interesse (que queremos detectar) e o
# tumor maligno -- que tambem e a classe MINORITARIA. Isso deixa as metricas
# de precisao/revocacao focadas em "acertar os casos de cancer".
df["diagnostico"] = df["target"].map({0: 1, 1: 0})  # 0(maligno)->1 ; 1(benigno)->0
df = df.drop(columns=["target"])

df.to_csv("cancer_mama.csv", index=False)

print(f"Dataset salvo: cancer_mama.csv  ({df.shape[0]} linhas, {df.shape[1]} colunas)")
print("Distribuicao do alvo (diagnostico):")
print(f"  Benigno (0): {(df['diagnostico']==0).sum()}")
print(f"  Maligno (1): {(df['diagnostico']==1).sum()}")
