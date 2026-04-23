import cv2
import csv
import mediapipe as mp
import sys
import os
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ---------------- MEDIA PIPE ---------------- #
mp_maos = mp.solutions.hands
maos = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

desenho = mp.solutions.drawing_utils

# ---------------- CSV ---------------- #
ARQUIVO_CSV = 'dados_libras.csv'

# Criar arquivo e cabeçalho apenas se não existir
arquivo_existe = os.path.exists(ARQUIVO_CSV)
arquivo = open(ARQUIVO_CSV, 'a', newline='', encoding='utf-8')
writer = csv.writer(arquivo)

if not arquivo_existe:
    cabecalho = ['label']
    for i in range(21):
        cabecalho.append(f'x{i}')
        cabecalho.append(f'y{i}')
    writer.writerow(cabecalho)

# ---------------- CONTADOR DE AMOSTRAS ---------------- #
contagem_labels = defaultdict(int)

if arquivo_existe:
    with open(ARQUIVO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # pular cabeçalho
        for linha in reader:
            if linha:
                contagem_labels[linha[0]] += 1

# ---------------- LABEL ---------------- #
letra = input('Digite a letra/sinal que quer gravar: ').upper()

print(f'\nAmostras já existentes para {letra}: {contagem_labels[letra]}')

# ---------------- WEBCAM ---------------- #
cap = cv2.VideoCapture(0)

print('\nPressione "S" para salvar um exemplo')
print('Pressione ESC para sair\n')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    lista_pontos = []

    # ---------------- DETECÇÃO ---------------- #
    if resultado.multi_hand_landmarks:
        mao = resultado.multi_hand_landmarks[0]

        desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

        altura, largura, _ = frame.shape

        for ponto in mao.landmark:
            cx = int(ponto.x * largura)
            cy = int(ponto.y * altura)
            lista_pontos.append([cx, cy])

    # ---------------- EXIBIR CONTAGEM NA TELA ---------------- #
    cv2.putText(
        frame,
        f'{letra}: {contagem_labels[letra]} amostras',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # ---------------- TECLAS ---------------- #
    tecla = cv2.waitKey(1)

    # ---------------- SALVAR DADOS ---------------- #
    if tecla == ord('s'):
        if resultado.multi_hand_landmarks and len(lista_pontos) == 21:

            # Normalização
            base_x = lista_pontos[0][0]
            base_y = lista_pontos[0][1]

            linha = [letra]

            for x, y in lista_pontos:
                linha.append(x - base_x)
                linha.append(y - base_y)

            writer.writerow(linha)
            arquivo.flush()

            # Atualizar contador
            contagem_labels[letra] += 1

            print(f'✔ Amostra salva | {letra}: {contagem_labels[letra]} amostras')

        else:
            print('⚠ Mão não detectada corretamente')

    # ---------------- SAIR ---------------- #
    if tecla == 27:
        break

    cv2.imshow('Coletando Dados - Libras', frame)

# ---------------- FINAL ---------------- #
cap.release()
arquivo.close()
cv2.destroyAllWindows()

print('\nResumo das amostras coletadas:')
for label, quantidade in sorted(contagem_labels.items()):
    print(f'{label}: {quantidade}')