# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledtdqpLi.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QSizePolicy,
    QStackedWidget, QWidget)

class Ui_StackedWidget(object):
    def setupUi(self, StackedWidget):
        if not StackedWidget.objectName():
            StackedWidget.setObjectName(u"StackedWidget")
        StackedWidget.resize(1366, 768)
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.label_signal = QLabel(self.page_1)
        self.label_signal.setObjectName(u"label_signal")
        self.label_signal.setGeometry(QRect(28, 85, 600, 500))
        self.label_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_cam = QLabel(self.page_1)
        self.label_cam.setObjectName(u"label_cam")
        self.label_cam.setGeometry(QRect(700, 250, 500, 400))
        self.label_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_return = QLabel(self.page_1)
        self.label_return.setObjectName(u"label_return")
        self.label_return.setGeometry(QRect(80, 590, 500, 150))
        self.label_return.setScaledContents(False)
        self.label_return.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pushButton_Aprender = QPushButton(self.page_1)
        self.pushButton_Aprender.setObjectName(u"pushButton_Aprender")
        self.pushButton_Aprender.setGeometry(QRect(30, 30, 81, 26))
        self.label_2 = QLabel(self.page_1)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(770, 150, 381, 51))
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.label_2.setFont(font)
        StackedWidget.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.pushButton_Testar = QPushButton(self.page_2)
        self.pushButton_Testar.setObjectName(u"pushButton_Testar")
        self.pushButton_Testar.setGeometry(QRect(30, 30, 81, 26))
        self.label = QLabel(self.page_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(390, 70, 531, 51))
        font1 = QFont()
        font1.setPointSize(26)
        font1.setBold(True)
        self.label.setFont(font1)
        self.label_signal_2 = QLabel(self.page_2)
        self.label_signal_2.setObjectName(u"label_signal_2")
        self.label_signal_2.setGeometry(QRect(700, 160, 600, 600))
        self.label_signal_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_signal_3 = QLabel(self.page_2)
        self.label_signal_3.setObjectName(u"label_signal_3")
        self.label_signal_3.setGeometry(QRect(50, 160, 600, 600))
        self.label_signal_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.label_2.setText(QCoreApplication.translate("StackedWidget", u"FA\u00c7A O SINAL DA LETRA", None))
        self.pushButton_Testar.setText(QCoreApplication.translate("StackedWidget", u"Testar", None))
        self.label.setText(QCoreApplication.translate("StackedWidget", u"APRENDA O SINAL DA LETRA A", None))
        self.label_signal_2.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_signal_3.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
    # retranslateUi

