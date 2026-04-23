import cv2
import joblib
import numpy as np
from main import DetectorMaos
import pandas as pd

# ---------------- CONFIGURAÇÕES ---------------- #
LIMIAR_CONFIANCA = 0.80

# ---------------- VERIFICAR DATASET ---------------- #
dados = pd.read_csv('dados_libras.csv')
print(dados['label'].value_counts())

# ---------------- CARREGAR MODELO ---------------- #
modelo = joblib.load('modelo_libras.pkl')

# ---------------- WEBCAM ---------------- #
cap = cv2.VideoCapture(0)

# ---------------- DETECTOR ---------------- #
detector = DetectorMaos(max_maos=1)

while True:
    ret, imagem = cap.read()
    if not ret:
        continue

    imagem = cv2.flip(imagem, 1)

    # Detectar mão
    imagem = detector.encontrar_maos(imagem)
    pontos = detector.encontrar_pontos(imagem, desenho=False)

    if len(pontos) == 21:
        # ---------------- NORMALIZAÇÃO ---------------- #
        base_x = pontos[0][1]
        base_y = pontos[0][2]

        entrada = []

        for _, x, y in pontos:
            entrada.append(x - base_x)
            entrada.append(y - base_y)

        entrada = np.array(entrada).reshape(1, -1)

        # ---------------- PREDIÇÃO COM CONFIANÇA ---------------- #
        probabilidades = modelo.predict_proba(entrada)[0]

        indice = np.argmax(probabilidades)
        previsao = modelo.classes_[indice]
        confianca = probabilidades[indice]

        # ---------------- FILTRO DE CONFIANÇA ---------------- #
        if confianca >= LIMIAR_CONFIANCA:
            texto = f'Sinal: {previsao} ({confianca * 100:.1f}%)'
            cor = (0, 255, 0)
        else:
            texto = f'Incerto ({confianca * 100:.1f}%)'
            cor = (0, 0, 255)

        cv2.putText(
            imagem,
            texto,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            cor,
            3
        )

    cv2.imshow('Reconhecimento de Libras', imagem)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()