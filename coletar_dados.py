import cv2
import csv
import mediapipe as mp
import sys
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
arquivo = open('dados_libras.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(arquivo)

# Cabeçalho (opcional mas recomendado)
# writer.writerow(["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

# ---------------- LABEL ---------------- #
letra = input("Digite a letra/sinal que quer gravar: ").upper()

# ---------------- WEBCAM ---------------- #
cap = cv2.VideoCapture(0)

print("\nPressione 'S' para salvar um exemplo")
print("Pressione ESC para sair\n")

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

        # garante ordem fixa dos 21 pontos
        for ponto in mao.landmark:
            cx = int(ponto.x * largura)
            cy = int(ponto.y * altura)
            lista_pontos.append([cx, cy])

    # ---------------- TECLAS ---------------- #
    tecla = cv2.waitKey(1)

    # SALVAR DADOS
    if tecla == ord('s'):

        if resultado.multi_hand_landmarks and len(lista_pontos) == 21:

            # ---------------- NORMALIZAÇÃO ---------------- #
            base_x = lista_pontos[0][0]
            base_y = lista_pontos[0][1]

            linha = [letra]

            for x, y in lista_pontos:
                linha.append(x - base_x)
                linha.append(y - base_y)

            writer.writerow(linha)
            arquivo.flush()

            print("✔ Amostra salva (normalizada)")

        else:
            print("⚠ Mão não detectada corretamente")

    # SAIR
    if tecla == 27:
        break

    cv2.imshow("Coletando Dados - Libras", frame)

# ---------------- FINAL ---------------- #
cap.release()
arquivo.close()
cv2.destroyAllWindows()