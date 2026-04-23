import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------- CARREGAR DADOS ---------------- #
dados = pd.read_csv('dados_libras.csv')

# Separar atributos e rótulos
X = dados.drop('label', axis=1)
y = dados['label']

# ---------------- DIVIDIR BASE ---------------- #
# 80% treino | 20% teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42, 
    stratify=y
)

# ---------------- TREINAR MODELO ---------------- #
modelo = RandomForestClassifier(
    n_estimators=200, # Número de árvores na floresta. Mais árvores = maior estabilidade (até certo ponto).
    random_state=42 # Garante reprodutibilidade. Sem isso, cada treino pode gerar resultados ligeiramente diferentes.
)

modelo.fit(X_treino, y_treino)

# ---------------- AVALIAR ---------------- #
predicoes = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, predicoes)
print(f'Acurácia: {acuracia * 100:.2f}%')

print('\nRelatório de Classificação:')
print(classification_report(y_teste, predicoes))

print('Matriz de Confusão:')
print(confusion_matrix(y_teste, predicoes))

# ---------------- SALVAR MODELO ---------------- #
joblib.dump(modelo, 'modelo_libras.pkl')
print('\nModelo salvo como modelo_libras.pkl')
