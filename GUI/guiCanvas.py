import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint


class PaintBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # 画板大小 (28x28 的 10倍)
        self.setStyleSheet("background-color: white; border: 1px solid black;")

        # 初始化画布
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)  # 白底

        self.last_point = QPoint()
        self.drawing = False

        # 画笔设置 (黑色，宽度20，圆头)
        self.pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def save_image(self, file_path):
        self.image.save(file_path)