# ============================================================
# BIBLIOTECAS
# ============================================================

import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp

# Componentes do PySide6 utilizados na interface gráfica
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import QApplication, QStackedWidget

# Interface criada no Qt Designer e convertida para Python
from ui_pages import Ui_StackedWidget


# ============================================================
# CLASSE PARA DETECÇÃO DA MÃO
# ============================================================

class DetectorMaos:

    def __init__(self,modo=False,max_maos=2,deteccao_confianca=0.5,rastreio_confianca=0.5,cor_pontos=(0, 0, 255),cor_conexoes=(255, 255, 255)):

        # Acessa o módulo de detecção de mãos do MediaPipe
        self.maos_mp = mp.solutions.hands

        # Configura o detector de mãos
        #
        # static_image_mode:
        #   Define se cada imagem será tratada individualmente.
        #
        # max_num_hands:
        #   Número máximo de mãos que podem ser detectadas.
        #
        # min_detection_confidence:
        #   Confiança mínima para considerar uma mão detectada.
        #
        # min_tracking_confidence:
        #   Confiança mínima para continuar rastreando a mão.
        self.maos = self.maos_mp.Hands( static_image_mode=modo, max_num_hands=max_maos, min_detection_confidence=deteccao_confianca, min_tracking_confidence=rastreio_confianca )

        # Utilizado para desenhar os pontos e conexões da mão
        self.desenho_mp = mp.solutions.drawing_utils

        # Configuração visual dos pontos (landmarks)
        self.desenho_config_pontos = self.desenho_mp.DrawingSpec( color=cor_pontos, thickness=2, circle_radius=4 )

        # Configuração visual das conexões entre os landmarks
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec( color=cor_conexoes, thickness=2 )

        # Armazena o resultado da última detecção
        self.resultado = None

    # ========================================================
    # DETECTA AS MÃOS NA IMAGEM
    # ========================================================

    def encontrar_maos(self, imagem, desenho=True):

        # O OpenCV utiliza BGR, enquanto o MediaPipe utiliza RGB.
        # Por isso, a imagem é convertida antes de ser processada.
        imagem_rgb = cv2.cvtColor( imagem, cv2.COLOR_BGR2RGB)

        # Executa a detecção das mãos
        self.resultado = self.maos.process(imagem_rgb)

        # Verifica se alguma mão foi encontrada
        if self.resultado.multi_hand_landmarks:

            # Percorre todas as mãos detectadas
            for pontos in self.resultado.multi_hand_landmarks:

                # Desenha os landmarks e suas conexões na imagem
                if desenho:
                    self.desenho_mp.draw_landmarks( imagem, pontos, self.maos_mp.HAND_CONNECTIONS, self.desenho_config_pontos, self.desenho_config_conexoes)

        # Retorna a imagem, podendo conter os landmarks desenhados
        return imagem

    # ========================================================
    # OBTÉM OS LANDMARKS DA MÃO
    # ========================================================

    def encontrar_pontos(self, imagem, mao_num=0, desenho=False):

        # Lista que armazenará os pontos encontrados
        lista_pontos = []

        # Verifica se existe alguma mão detectada
        if self.resultado and self.resultado.multi_hand_landmarks:

            # Seleciona a mão desejada
            minha_mao = self.resultado.multi_hand_landmarks[mao_num]

            # Obtém altura e largura da imagem
            h, w, _ = imagem.shape

            # O MediaPipe fornece 21 landmarks por mão
            for id_ponto, landmark in enumerate(minha_mao.landmark):

                # Converte as coordenadas normalizadas do MediaPipe
                # para coordenadas em pixels da imagem
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)

                # Armazena:
                # [identificador do ponto, coordenada X, coordenada Y]
                lista_pontos.append([ id_ponto, cx, cy ])

                # Opcionalmente desenha um círculo em cada ponto
                if desenho:
                    cv2.circle(imagem,(cx, cy),6,(255, 0, 0),cv2.FILLED)

        # Retorna a lista contendo os 21 landmarks
        return lista_pontos


# ============================================================
# JANELA PRINCIPAL DA APLICAÇÃO
# ============================================================

class MainWindow(QStackedWidget):

    # Confiança mínima necessária para aceitar uma previsão
    LIMIAR_CONFIANCA = 0.80


    def __init__(self):

        # Inicializa a classe QStackedWidget
        super().__init__()

        # Cria a interface gerada pelo Qt Designer
        self.ui = Ui_StackedWidget()
        self.ui.setupUi(self)

        # Configura os botões de navegação entre as páginas
        self.configurar_navegacao()

        # Cria o detector de mãos.
        # Neste projeto será utilizada apenas uma mão.
        self.detector = DetectorMaos(max_maos=1)

        # Carrega o modelo de Machine Learning previamente treinado
        self.modelo = joblib.load('modelo_libras.pkl')

        # Inicialmente a câmera não está aberta
        self.cap = None

        # Timer utilizado para atualizar continuamente os frames
        self.timer = QTimer()

        # Carrega as imagens de referência da letra/sinal
        self.load_reference_image_letra()
        self.load_reference_image_sinal()
        self.load_reference_mao_sinal()

        # Inicializa a câmera
        self.init_camera()


    # ========================================================
    # CARREGA IMAGEM DE REFERÊNCIA DA LETRA
    # ========================================================

    def load_reference_image_letra(self):

        # Carrega a imagem que representa a letra A
        pixmap = QPixmap('imagens/modelo_letra_A.png')

        # Verifica se a imagem foi carregada corretamente
        if not pixmap.isNull():
            # Ajusta a imagem ao tamanho do QLabel,
            # mantendo sua proporção.
            self.ui.label_signal.setPixmap(pixmap.scaled(self.ui.label_signal.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))


    # ========================================================
    # CARREGA IMAGEM DE REFERÊNCIA DO SINAL
    # ========================================================

    def load_reference_image_sinal(self):

        # Carrega a imagem do sinal correspondente à letra A
        pixmap = QPixmap('imagens/sinal_letra_A.png')

        # Verifica se a imagem foi carregada corretamente
        if not pixmap.isNull():
            # Redimensiona a imagem mantendo a proporção
            self.ui.label_signal_2.setPixmap(pixmap.scaled(self.ui.label_signal_2.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # ========================================================
    # CARREGA IMAGEM DE REFERÊNCIA DA MÃO
    # ========================================================

    def load_reference_mao_sinal(self):

        # Carrega uma imagem mostrando o posicionamento da mão
        pixmap = QPixmap('imagens/mao_letra_A.jpeg')

        # Verifica se a imagem foi carregada corretamente
        if not pixmap.isNull():

            # Redimensiona a imagem mantendo sua proporção
            self.ui.label_signal_3.setPixmap(pixmap.scaled(self.ui.label_signal_3.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # ========================================================
    # INICIALIZA A CÂMERA
    # ========================================================

    def init_camera(self):

        # Abre a câmera padrão do computador
        self.cap = cv2.VideoCapture(0)

        # Faz com que update_frame() seja executado
        # sempre que o timer disparar
        self.timer.timeout.connect(self.update_frame)

        # Define o intervalo de atualização em 30 ms
        # aproximadamente 33 frames por segundo
        self.timer.start(30)


    # ========================================================
    # RECONHECIMENTO DO SINAL
    # ========================================================

    def reconhecer_sinal(self, frame):

        # Obtém os landmarks da mão detectada
        pontos = self.detector.encontrar_pontos(frame,desenho=False)

        # Uma mão detectada pelo MediaPipe possui 21 landmarks.
        # Caso não sejam encontrados exatamente 21 pontos,
        # não é possível realizar a classificação.
        if len(pontos) != 21:
            return None, None

        # ====================================================
        # NORMALIZAÇÃO DAS COORDENADAS
        # ====================================================
        # O landmark 0 corresponde à base/pulso da mão.
        # Ele será utilizado como ponto de referência para
        # calcular a posição relativa dos demais pontos.
        base_x = pontos[0][1]
        base_y = pontos[0][2]

        # Lista que armazenará os dados enviados ao modelo
        entrada = []

        # Para cada landmark:
        # X relativo = X do ponto - X da base
        # Y relativo = Y do ponto - Y da base
        # Isso faz com que a posição absoluta da mão na imagem
        # tenha menor influência sobre a classificação.
        for _, x, y in pontos:

            entrada.append(x - base_x)
            entrada.append(y - base_y)

        # Converte os dados para um array NumPy
        # e organiza no formato esperado pelo modelo.
        entrada = np.array(entrada).reshape(1, -1)

        # ====================================================
        # CLASSIFICAÇÃO PELO MODELO
        # ====================================================
        # Obtém a probabilidade de cada classe conhecida
        # pelo modelo de Machine Learning.
        probabilidades = self.modelo.predict_proba(entrada)[0]

        # Obtém o índice da classe com maior probabilidade
        indice = np.argmax(probabilidades)

        # Obtém o nome da classe correspondente ao índice
        previsao = self.modelo.classes_[indice]

        # Obtém a confiança da previsão
        confianca = probabilidades[indice]

        # Retorna a classe prevista e sua confiança
        return previsao, confianca


    # ========================================================
    # CONFIGURAÇÃO DA NAVEGAÇÃO
    # ========================================================

    def configurar_navegacao(self):

        # Botão "Aprender A":
        # direciona o usuário para a página de aprendizagem.
        self.ui.pushButton_Aprender_A.clicked.connect(lambda: self.setCurrentWidget(self.ui.page_2))

        # Botão "Testar":
        # direciona o usuário para a página de teste.
        self.ui.pushButton_Testar.clicked.connect(lambda: self.setCurrentWidget(self.ui.page_1))

    # ========================================================
    # ATUALIZAÇÃO DA CÂMERA E RECONHECIMENTO
    # ========================================================

    def update_frame(self):

        # Verifica se a câmera está disponível e aberta
        if not self.cap or not self.cap.isOpened():
            return
        
        # Captura um frame da câmera
        ret, frame = self.cap.read()

        # Caso o frame não tenha sido capturado,
        # encerra esta atualização.
        if not ret:
            return

        # Inverte horizontalmente a imagem.
        # Isso cria um efeito semelhante a um espelho,
        # facilitando a interação do usuário.
        frame = cv2.flip(frame,1)

        # Detecta a mão e desenha os landmarks na imagem
        frame = self.detector.encontrar_maos(frame)

        # Tenta reconhecer o sinal apresentado pelo usuário
        previsao, confianca = self.reconhecer_sinal(frame)

        # ====================================================
        # CASO UMA PREVISÃO TENHA SIDO OBTIDA
        # ====================================================

        if previsao is not None:

            # Verifica se a confiança da previsão
            # é maior ou igual ao limite definido.
            if confianca >= self.LIMIAR_CONFIANCA:

                # Texto exibido na interface e na câmera
                texto = (f'Sinal: {previsao} 'f'({confianca * 100:.1f}%)')

                # Verde indica uma previsão aceita
                cor = (0, 255, 0)

                # ============================================
                # VERIFICAÇÃO DA RESPOSTA DO USUÁRIO
                # ============================================

                # Neste momento o exercício está configurado
                # para verificar especificamente a letra A.
                if previsao == "A":

                    # Mensagem de sucesso
                    self.ui.label_return.setText("Parabéns, Você conseguiu!")

                    # Estilo visual da mensagem de sucesso
                    self.ui.label_return.setStyleSheet("""
                        QLabel {
                            background-color: #28a745;
                            color: white;
                            font-size: 24px;
                            font-weight: bold;
                            border-radius: 10px;
                        }
                    """)


                # Atualiza o resultado na interface,
                # caso o QLabel exista.
                if hasattr(self.ui,'label_resultado'):
                    self.ui.label_resultado.setText(texto)

            # =================================================
            # PREVISÃO ABAIXO DO LIMIAR
            # =================================================

            else:

                # Informa que o modelo não possui confiança
                # suficiente para aceitar a previsão.
                texto = (f'Incerto ' f'({confianca * 100:.1f}%)')

                # Vermelho indica baixa confiança
                cor = (0, 0, 255)

                # Atualiza o resultado na interface
                if hasattr(self.ui,'label_resultado'):
                    self.ui.label_resultado.setText(texto)

                # Mensagem de tentativa novamente
                self.ui.label_return.setText("Não foi dessa vez, continue tentando!")

                # Estilo visual da mensagem de erro
                self.ui.label_return.setStyleSheet("""
                QLabel {
                        background-color: #dc3545;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        border-radius: 10px;
                    }
                """)

            # =================================================
            # ESCREVE O RESULTADO SOBRE O VÍDEO
            # =================================================

            cv2.putText(frame,texto,(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,cor,3)

        # ====================================================
        # NENHUMA MÃO DETECTADA
        # ====================================================

        else:

            # Exibe mensagem informando que o usuário
            # deve continuar tentando.
            self.ui.label_return.setText("Não foi dessa vez, continue tentando!")

            # Estilo visual da mensagem
            self.ui.label_return.setStyleSheet("""
            QLabel {
                background-color: #dc3545;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 10px;
                }
            """)


        # ====================================================
        # CONVERSÃO PARA EXIBIÇÃO NO QT
        # ====================================================
        # OpenCV trabalha originalmente com BGR.
        # O QImage utiliza RGB.
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Obtém as dimensões do frame
        h, w, ch = frame_rgb.shape

        # Calcula o número de bytes de cada linha da imagem
        bytes_per_line = ch * w

        # Cria uma imagem compatível com o Qt
        qt_image = QImage(frame_rgb.data,w,h,bytes_per_line,QImage.Format_RGB888)

        # Converte QImage para QPixmap
        pixmap = QPixmap.fromImage(qt_image)

        # Exibe o frame no QLabel da interface
        self.ui.label_cam.setPixmap(pixmap.scaled(self.ui.label_cam.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # ========================================================
    # ENCERRAMENTO DA APLICAÇÃO
    # ========================================================

    def closeEvent(self, event):

        # Verifica se a câmera está aberta
        if self.cap and self.cap.isOpened():

            # Libera a câmera
            self.cap.release()

        # Permite que a janela seja encerrada
        event.accept()


# ============================================================
# CONFIGURAÇÃO DO TEMA DA APLICAÇÃO
# ============================================================

def aplicar_tema(app,cor_fundo="#ffffff",cor_texto="#000000"):

    # Define o estilo Fusion do Qt
    app.setStyle("Fusion")

    # Aplica as cores para todos os widgets
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {cor_fundo};
            color: {cor_texto};
        }}
    """)


# ============================================================
# INÍCIO DA APLICAÇÃO
# ============================================================

if __name__ == '__main__':

    # Cria a aplicação Qt
    app = QApplication(sys.argv)

    # Define o ícone da aplicação
    app.setWindowIcon(QIcon("imagens/icon_alfa.png"))

    # Aplica o tema da aplicação
    aplicar_tema(app, "#6a6a6a", "#000000")

    # Cria a janela principal
    window = MainWindow()

    # Exibe a janela
    window.show()

    # Inicia o loop de eventos do Qt
    sys.exit(app.exec())