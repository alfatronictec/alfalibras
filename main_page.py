import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QStackedWidget

from ui_pages import Ui_StackedWidget


class DetectorMaos:
    def __init__(self, modo=False, max_maos=2,
                 deteccao_confianca=0.5,
                 rastreio_confianca=0.5,
                 cor_pontos=(0, 0, 255),
                 cor_conexoes=(255, 255, 255)):

        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            static_image_mode=modo,
            max_num_hands=max_maos,
            min_detection_confidence=deteccao_confianca,
            min_tracking_confidence=rastreio_confianca
        )

        self.desenho_mp = mp.solutions.drawing_utils
        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(
            color=cor_pontos,
            thickness=2,
            circle_radius=4
        )
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(
            color=cor_conexoes,
            thickness=2
        )

        self.resultado = None

    def encontrar_maos(self, imagem, desenho=True):
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        self.resultado = self.maos.process(imagem_rgb)

        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                if desenho:
                    self.desenho_mp.draw_landmarks(
                        imagem,
                        pontos,
                        self.maos_mp.HAND_CONNECTIONS,
                        self.desenho_config_pontos,
                        self.desenho_config_conexoes
                    )

        return imagem

    def encontrar_pontos(self, imagem, mao_num=0, desenho=False):
        lista_pontos = []

        if self.resultado and self.resultado.multi_hand_landmarks:
            minha_mao = self.resultado.multi_hand_landmarks[mao_num]
            h, w, _ = imagem.shape

            for id_ponto, landmark in enumerate(minha_mao.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                lista_pontos.append([id_ponto, cx, cy])

                if desenho:
                    cv2.circle(imagem, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

        return lista_pontos


class MainWindow(QStackedWidget):
    LIMIAR_CONFIANCA = 0.80

    def __init__(self):
        super().__init__()

        self.ui = Ui_StackedWidget()
        self.ui.setupUi(self)

        self.configurar_navegacao()
        
        self.detector = DetectorMaos(max_maos=1)
        self.modelo = joblib.load('modelo_libras.pkl')

        self.cap = None
        self.timer = QTimer()

        self.load_reference_image_letra()
        self.load_reference_image_sinal()
        self.init_camera()

    def load_reference_image_letra(self):
        pixmap = QPixmap('imagens/letra_A.png')

        if not pixmap.isNull():
            self.ui.label_signal.setPixmap(
                pixmap.scaled(
                    self.ui.label_signal.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def load_reference_image_sinal(self):
        pixmap = QPixmap('imagens/sinal_letra_A.png')

        if not pixmap.isNull():
            self.ui.label_signal_2.setPixmap(
                pixmap.scaled(
                    self.ui.label_signal_2.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def reconhecer_sinal(self, frame):
        pontos = self.detector.encontrar_pontos(frame, desenho=False)

        if len(pontos) != 21:
            return None, None

        base_x = pontos[0][1]
        base_y = pontos[0][2]

        entrada = []
        for _, x, y in pontos:
            entrada.append(x - base_x)
            entrada.append(y - base_y)

        entrada = np.array(entrada).reshape(1, -1)

        probabilidades = self.modelo.predict_proba(entrada)[0]
        indice = np.argmax(probabilidades)

        previsao = self.modelo.classes_[indice]
        confianca = probabilidades[indice]

        return previsao, confianca

    def configurar_navegacao(self):
        self.ui.pushButton_Aprender.clicked.connect(
            lambda: self.setCurrentWidget(self.ui.page_2)
        )

        self.ui.pushButton_Testar.clicked.connect(
            lambda: self.setCurrentWidget(self.ui.page_1)
        )

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = self.detector.encontrar_maos(frame)

        previsao, confianca = self.reconhecer_sinal(frame)

        if previsao is not None:
            if confianca >= self.LIMIAR_CONFIANCA:
                texto = f'Sinal: {previsao} ({confianca * 100:.1f}%)'
                cor = (0, 255, 0)

                if previsao == "A":

                    self.ui.label_return.setText("Acertou")
                    self.ui.label_return.setStyleSheet("""
                        QLabel {
                            background-color: #28a745;
                            color: white;
                            font-size: 24px;
                            font-weight: bold;
                            border-radius: 10px;
                        }
                    """)

                if hasattr(self.ui, 'label_resultado'):
                    self.ui.label_resultado.setText(texto)
            else:
                texto = f'Incerto ({confianca * 100:.1f}%)'
                cor = (0, 0, 255)

                if hasattr(self.ui, 'label_resultado'):
                    self.ui.label_resultado.setText(texto)
                
                self.ui.label_return.setText("Continue tentando")
                self.ui.label_return.setStyleSheet("""
                QLabel {
                        background-color: #dc3545;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        border-radius: 10px;
                    }
                """)

            cv2.putText(
                frame,
                texto,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                cor,
                3
            )

        else:
            self.ui.label_return.setText("Continue tentando")
            self.ui.label_return.setStyleSheet("""
            QLabel {
                background-color: #dc3545;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 10px;
                }
            """)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        self.ui.label_cam.setPixmap(
            pixmap.scaled(
                self.ui.label_cam.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
