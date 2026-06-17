# GUIA DE DEFESA — Trabalho Final de IA

> **Só para você estudar.** O feedback anterior foi: (A) o método não era da matéria e (B) você não soube explicar. Este trabalho resolve os dois: usa **SVM + RNA** (os métodos da sala) e este guia te prepara para **explicar tudo**. Leia até entender — não decore só.

---

## RESUMO EM 30 SEGUNDOS (decore a ideia)

> "Classifiquei tumores de mama em malignos ou benignos a partir de 30 medidas de exame. Comparei um SVM (Parte 1) com uma Rede Neural (Parte 2). O dataset era desbalanceado, então balanceei as classes do treino por oversampling. Avaliei com matriz de confusão e métricas. A Rede Neural venceu — F1 de 96,3% contra 95,0% do SVM — porque deixou de detectar menos casos malignos."

---

## AS PERGUNTAS QUE VOCÊ JÁ SABE QUE VÊM

### 1. "Qual é a base de dados (dataset)?"
**Breast Cancer Wisconsin**, do scikit-learn. São **569 exames** de tumores de mama, cada um com **30 atributos numéricos** (medidas dos núcleos celulares: raio, textura, perímetro, área, concavidade etc.). O alvo é **maligno (1)** ou **benigno (0)**. Eu salvei tudo no arquivo `cancer_mama.csv`.

### 2. "Está balanceado?"
**Não.** É desbalanceado: **357 benignos (62,7%)** e **212 malignos (37,3%)**. Tem mais benignos que malignos.

### 3. ⭐ "Onde (em que função) você balanceou? O código balanceia no treino e no teste?"
Esta é a pergunta-chave. Resposta completa:

O balanceamento acontece em **dois momentos**, os dois visíveis no `main.py`:

**(a) Na divisão treino/teste — função `train_test_split` com o parâmetro `stratify=y`.**
Isso mantém a **mesma proporção de classes no treino e no teste**. Ou seja, os dois conjuntos ficam balanceados na mesma proporção um em relação ao outro. *(É isso que garante "balanceamento na hora de treinar e na hora de testar".)*

**(b) No conjunto de treino — função `balancear_classes` (que eu escrevi).**
Ela faz **oversampling**: sorteia cópias das amostras da classe minoritária (maligno) até as duas classes ficarem com o mesmo número (285 × 285). Assim o modelo aprende as duas classes com igual peso e não fica "viciado" em chutar benigno.

**Por que NÃO balanceei o teste com oversampling?** Porque o teste deve refletir a realidade. Se eu duplicasse amostras no teste, estaria "trapaceando" na avaliação. Para o desbalanceamento do teste não enganar, eu também reporto **acurácia balanceada** e **revocação**, que olham cada classe separadamente.

> Onde apontar no código: a função `balancear_classes(...)` está logo após a divisão e a normalização. Você pode mostrar as linhas que imprimem "Antes -> ... | Depois -> ..." rodando.

### 4. ⭐ "Como foi feita a RNA? Explique as camadas de neurônios."
A RNA é um **Perceptron Multicamadas (MLP)**. A arquitetura é:

```
Entrada (30 neurônios) → Oculta 1 (32, ReLU) → Oculta 2 (16, ReLU) → Saída (1, probabilidade)
```

- **Camada de entrada: 30 neurônios** — um para cada atributo do exame.
- **1ª camada oculta: 32 neurônios**, ativação **ReLU**.
- **2ª camada oculta: 16 neurônios**, ativação **ReLU**.
- **Camada de saída: 1 neurônio** — dá a probabilidade de o tumor ser maligno.
- **ReLU** = `max(0, x)`: zera valores negativos. Serve para a rede aprender padrões **não-lineares**.
- **Otimizador Adam**, taxa de aprendizado 0,001.
- Treinei por **200 épocas** (200 passagens pelos dados). A cada época guardei a perda para desenhar a curva.

### 5. "Como foram feitos os cálculos? Você fez matriz de confusão e explicitou os falsos positivos e negativos?"
Sim. A **matriz de confusão** cruza o real com o previsto e dá 4 números (classe positiva = maligno):

| | Previu Benigno | Previu Maligno |
|---|---|---|
| **É Benigno** | VN | FP |
| **É Maligno** | FN | VP |

- **VP** (Verdadeiro Positivo): maligno previsto como maligno. ✔
- **VN** (Verdadeiro Negativo): benigno previsto como benigno. ✔
- **FP** (Falso Positivo): benigno previsto como maligno (alarme falso).
- **FN** (Falso Negativo): maligno previsto como benigno → **o pior erro**, deixa passar um câncer.

Todas as métricas saem desses 4 números:
- **Acurácia** = (VP+VN) / total
- **Precisão** = VP / (VP+FP)
- **Revocação** = VP / (VP+FN)
- **F1** = média harmônica de precisão e revocação

> O código imprime esses 4 valores na tela para cada modelo (função `explicar_matriz`).

### 6. "Mostre os resultados (acurácia e outros indicadores)."

**Números reais (no conjunto de teste, 114 amostras):**

| Métrica | SVM (P1) | RNA (P2) |
|---|---|---|
| Acurácia | 96,49% | **97,37%** |
| Acurácia Balanceada | 95,24% | **96,43%** |
| Precisão | 100% | 100% |
| Revocação | 90,48% | **92,86%** |
| F1-Score | 95,00% | **96,30%** |

Matrizes: **SVM** → VN=72, FP=0, FN=4, VP=38. **RNA** → VN=72, FP=0, FN=3, VP=39.

**Leitura:** os dois tiveram precisão 100% (zero falso positivo). A RNA ganhou na revocação — deixou passar só 3 malignos, contra 4 do SVM. Por isso a RNA é o melhor modelo aqui.

---

## OUTRAS PERGUNTAS PROVÁVEIS

### "O que é o SVM?"
Método que acha a **fronteira (hiperplano) que separa as duas classes com a maior margem possível** — a linha que fica o mais longe possível dos pontos de cada lado. Usei o **kernel RBF** para separar classes que não são separáveis por uma reta.

### "Por que normalizou os dados (StandardScaler)?"
Porque os atributos têm escalas muito diferentes (área na casa dos milhares, suavidade perto de zero). Normalizar (média 0, desvio 1) impede que os atributos de valor alto dominem. Ajustei o scaler **só no treino** e apliquei no teste — para não haver **data leakage** (vazamento).

### "O que é a curva de perda? O que ela mostra?"
É a perda (erro) da rede a cada época. Cai rápido no começo. A partir da época ~50, a perda de **treino** continua caindo mas a de **teste** começa a subir → **overfitting** (a rede começa a decorar o treino em vez de generalizar). Mesmo assim a acurácia final ficou alta. Para melhorar, usaria *early stopping*.

### "Por que a RNA foi melhor que o SVM?"
A rede neural tem mais camadas e mais parâmetros, então captura padrões **não-lineares mais complexos**. Neste caso isso se traduziu em detectar um tumor maligno a mais que o SVM.

### "Qual métrica é a mais importante aqui e por quê?"
A **revocação** da classe maligna. Em medicina, o pior erro é o **falso negativo** (dizer que está tudo bem quando há câncer). Revocação alta = poucos malignos passando despercebidos.

### "O que é a SEED / random_state=42?"
Fixa a aleatoriedade (divisão dos dados, sorteio do oversampling, pesos iniciais da rede) para os resultados serem **reproduzíveis** — qualquer um que rodar obtém os mesmos números.

### "O que tem dentro dos .pkl?"
São os modelos já treinados, salvos com `pickle`. Dá para carregar e prever sem treinar de novo. Salvei também o `scaler.pkl`, porque dados novos precisam ser normalizados com a mesma escala antes da predição.

### "Qual a estrutura do código?" (mostrar rápido)
`main.py` segue 7 blocos numerados: (1) carrega o dataset, (2) divide treino/teste e normaliza, (3) **balanceia o treino**, (4) treina o SVM, (5) treina a RNA, (6) avalia com matriz de confusão e métricas, (7) gera os gráficos. É só rolar o arquivo mostrando os comentários de cada bloco.

---

## NA HORA DE RODAR AO VIVO
1. Abra o terminal já na pasta do projeto.
2. Rode `python main.py`.
3. Vá narrando: "carregou o dataset... aqui vê que é desbalanceado, 357 contra 212... aqui ele balanceia o treino, ficou 285 e 285... treina o SVM... treina a rede por 200 épocas... e aqui a avaliação, com VP, VN, FP, FN de cada modelo... a RNA ganhou no F1."
4. Abra os gráficos da pasta `graficos/` para mostrar.

**Plano B:** se algo travar, abra o `relatorio.pdf` — tem todos os resultados e gráficos salvos.
