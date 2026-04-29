import sys
import cv2
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


class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.ui = Ui_StackedWidget()
        self.ui.setupUi(self)

        self.detector = DetectorMaos()
        self.cap = None
        self.timer = QTimer()

        self.load_reference_image()
        self.init_camera()

    def load_reference_image(self):
        pixmap = QPixmap("imagens/sinal_A.jpeg")

        if not pixmap.isNull():
            self.ui.label_signal.setPixmap(
                pixmap.scaled(
                    self.ui.label_signal.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = self.detector.encontrar_maos(frame)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())