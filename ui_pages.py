# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledBndLqt.ui'
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
        StackedWidget.resize(1280, 720)
        StackedWidget.setMaximumSize(QSize(1280, 720))
        StackedWidget.setBaseSize(QSize(1280, 720))
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")

        # Plano de Fundo
        self.page_1.setStyleSheet("""
        QWidget#page_1 {
            border-image: url(imagens/alfa.jpeg) 0 0 0 0 stretch stretch;
        }
        """)

        self.page_1.setObjectName("page_1")
        self.label_signal = QLabel(self.page_1)
        self.label_signal.setObjectName(u"label_signal")
        self.label_signal.setGeometry(QRect(750, 200, 300, 300))
        self.label_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_cam = QLabel(self.page_1)
        self.label_cam.setObjectName(u"label_cam")
        self.label_cam.setGeometry(QRect(120, 150, 500, 400))
        self.label_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_return = QLabel(self.page_1)
        self.label_return.setObjectName(u"label_return")
        self.label_return.setGeometry(QRect(120, 550, 500, 100))
        self.label_return.setScaledContents(False)
        self.label_return.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pushButton_Aprender_A = QPushButton(self.page_1)
        self.pushButton_Aprender_A.setObjectName(u"pushButton_Aprender_A")
        self.pushButton_Aprender_A.setGeometry(QRect(30, 30, 100, 100))
        self.pushButton_Aprender_A.setIconSize(QSize(20, 20))
        self.label_2 = QLabel(self.page_1)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(440, 30, 330, 50))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_4 = QLabel(self.page_1)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(440, 80, 330, 50))
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(36)
        font1.setBold(True)
        self.label_4.setFont(font1)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        StackedWidget.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")

        # Plano de Fundo 
        self.page_2.setStyleSheet("""
        QWidget#page_2 {
            border-image: url(imagens/alfa.jpeg) 0 0 0 0 stretch stretch;
        }
        """)
        self.page_2.setObjectName("page_2")
        font2 = QFont()
        font2.setPointSize(10)
        self.page_2.setFont(font2)
        self.pushButton_Testar = QPushButton(self.page_2)
        self.pushButton_Testar.setObjectName(u"pushButton_Testar")
        self.pushButton_Testar.setGeometry(QRect(30, 30, 100, 100))
        self.label = QLabel(self.page_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(430, 30, 380, 50))
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_signal_2 = QLabel(self.page_2)
        self.label_signal_2.setObjectName(u"label_signal_2")
        self.label_signal_2.setGeometry(QRect(650, 160, 410, 500))
        self.label_signal_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_signal_3 = QLabel(self.page_2)
        self.label_signal_3.setObjectName(u"label_signal_3")
        self.label_signal_3.setGeometry(QRect(150, 160, 450, 390))
        self.label_signal_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pushButton_Aprender_E = QPushButton(self.page_2)
        self.pushButton_Aprender_E.setObjectName(u"pushButton_Aprender_E")
        self.pushButton_Aprender_E.setGeometry(QRect(1140, 30, 100, 100))
        self.label_3 = QLabel(self.page_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(430, 80, 380, 50))
        self.label_3.setFont(font1)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        StackedWidget.addWidget(self.page_2)

        self.retranslateUi(StackedWidget)

        QMetaObject.connectSlotsByName(StackedWidget)
    # setupUi

    def retranslateUi(self, StackedWidget):
        StackedWidget.setWindowTitle(QCoreApplication.translate("StackedWidget", u"Alfalibras", None))
        self.label_signal.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_cam.setText(QCoreApplication.translate("StackedWidget", u"Imagem Camera", None))
        self.label_return.setText(QCoreApplication.translate("StackedWidget", u"N\u00e3o foi dessa vez, continue tentando!", None))

        self.pushButton_Aprender_A.setStyleSheet("""
        QPushButton{
            border: none;
            background-image: url(imagens/icone_aprender.png);
            background-repeat: no-repeat;
            background-position: center;}

            QPushButton:hover {
                background-image: url(imagens/icon_aprender_pressed.png);
            }
            
            QPushButton:pressed {
            background-image: url(imagens/icon_aprender_pressed.png);
        }    
        """)

        self.label_2.setText(QCoreApplication.translate("StackedWidget", u"Fa\u00e7a o Sinal", None))
        self.label_4.setText(QCoreApplication.translate("StackedWidget", u"Letra A", None))

        self.pushButton_Testar.setStyleSheet("""
        QPushButton{
            border: none;
            background-image: url(imagens/icone_cam.png);
            background-repeat: no-repeat;
            background-position: center;}
            
            QPushButton:hover {
                background-image: url(imagens/icom_cam_pressed.png);
            }
                        
            QPushButton:pressed {
            background-image: url(imagens/icom_cam_pressed.png);
        }    
        """)

        self.label.setText(QCoreApplication.translate("StackedWidget", u"Aprenda o Sinal", None))
        self.label_signal_2.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))
        self.label_signal_3.setText(QCoreApplication.translate("StackedWidget", u"Imagem do Sinal", None))

        self.pushButton_Aprender_E.setStyleSheet("""
        QPushButton{
            border: none;
            background-image: url(imagens/icon_passar.png);
            background-repeat: no-repeat;
            background-position: center;}
            QPushButton:hover {
                background-image: url(imagens/icon_passar_pressed.png);
            }
            QPushButton:pressed {
            background-image: url(imagens/icon_passar_pressed.png);
        }    
        """)


        self.label_3.setText(QCoreApplication.translate("StackedWidget", u"Letra A", None))
    # retranslateUi

