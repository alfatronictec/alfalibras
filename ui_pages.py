# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledfJqCwe.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QWidget)

class Ui_StackedWidget(object):
    def setupUi(self, StackedWidget):
        if not StackedWidget.objectName():
            StackedWidget.setObjectName(u"StackedWidget")
        StackedWidget.resize(800, 600)
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.label_signal = QLabel(self.page_1)
        self.label_signal.setObjectName(u"label_signal")
        self.label_signal.setGeometry(QRect(28, 85, 381, 381))
        self.label_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_cam = QLabel(self.page_1)
        self.label_cam.setObjectName(u"label_cam")
        self.label_cam.setGeometry(QRect(430, 85, 351, 351))
        self.label_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_return = QLabel(self.page_1)
        self.label_return.setObjectName(u"label_return")
        self.label_return.setGeometry(QRect(70, 480, 291, 111))
        self.label_return.setScaledContents(False)
        self.label_return.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pushButton_Aprender = QPushButton(self.page_1)
        self.pushButton_Aprender.setObjectName(u"pushButton_Aprender")
        self.pushButton_Aprender.setGeometry(QRect(30, 30, 81, 26))
        self.label_2 = QLabel(self.page_1)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(320, 40, 251, 41))
        font = QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        StackedWidget.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.pushButton_Testar = QPushButton(self.page_2)
        self.pushButton_Testar.setObjectName(u"pushButton_Testar")
        self.pushButton_Testar.setGeometry(QRect(30, 30, 81, 26))
        self.scrollArea = QScrollArea(self.page_2)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(30, 70, 691, 711))
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 675, 707))
        self.label = QLabel(self.scrollAreaWidgetContents)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(260, 10, 111, 31))
        self.label.setFont(font)
        self.label_signal_2 = QLabel(self.scrollAreaWidgetContents)
        self.label_signal_2.setObjectName(u"label_signal_2")
        self.label_signal_2.setGeometry(QRect(140, 60, 381, 401))
        self.label_signal_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_signal_3 = QLabel(self.scrollAreaWidgetContents)
        self.label_signal_3.setObjectName(u"label_signal_3")
        self.label_signal_3.setGeometry(QRect(150, 430, 381, 401))
        self.label_signal_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        StackedWidget.addWidget(self.page_2)

        self.retranslateUi(StackedWidget)

        QMetaObject.connectSlotsByName(StackedWidget)
    # setupUi

    def retranslateUi(self, StackedWidget):
        StackedWidget.setWindowTitle(QCoreApplication.translate("StackedWidget", u"Alfalibras", None))
        self.label_signal.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_cam.setText(QCoreApplication.translate("StackedWidget", u"Imagem Camera", None))
        self.label_return.setText(QCoreApplication.translate("StackedWidget", u"Continue Tentando", None))
        self.pushButton_Aprender.setText(QCoreApplication.translate("StackedWidget", u"Aprender", None))
        self.label_2.setText(QCoreApplication.translate("StackedWidget", u"Fa\u00e7a o Sinal da Letra na Imagem", None))
        self.pushButton_Testar.setText(QCoreApplication.translate("StackedWidget", u"Testar", None))
        self.label.setText(QCoreApplication.translate("StackedWidget", u"Sinal da Letra A", None))
        self.label_signal_2.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_signal_3.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
    # retranslateUi

