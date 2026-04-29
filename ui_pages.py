# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledMKJNCj.ui'
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
        StackedWidget.resize(800, 600)
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.label_signal = QLabel(self.page_1)
        self.label_signal.setObjectName(u"label_signal")
        self.label_signal.setGeometry(QRect(28, 25, 381, 441))
        self.label_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_cam = QLabel(self.page_1)
        self.label_cam.setObjectName(u"label_cam")
        self.label_cam.setGeometry(QRect(438, 35, 291, 231))
        self.label_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pushButton = QPushButton(self.page_1)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QRect(540, 290, 81, 31))
        self.label_return = QLabel(self.page_1)
        self.label_return.setObjectName(u"label_return")
        self.label_return.setGeometry(QRect(70, 480, 291, 111))
        self.label_return.setScaledContents(False)
        self.label_return.setAlignment(Qt.AlignmentFlag.AlignCenter)
        StackedWidget.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        StackedWidget.addWidget(self.page_2)

        self.retranslateUi(StackedWidget)

        QMetaObject.connectSlotsByName(StackedWidget)
    # setupUi

    def retranslateUi(self, StackedWidget):
        StackedWidget.setWindowTitle(QCoreApplication.translate("StackedWidget", u"Alfalibras", None))
        self.label_signal.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_cam.setText(QCoreApplication.translate("StackedWidget", u"Imagem Camera", None))
        self.pushButton.setText(QCoreApplication.translate("StackedWidget", u"PushButton", None))
        self.label_return.setText(QCoreApplication.translate("StackedWidget", u"Continue Tentando", None))
    # retranslateUi

