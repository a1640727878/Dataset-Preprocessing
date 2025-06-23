import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel


app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Hello World")
window.setFixedSize(200, 200)
label = QLabel("Hello World!", window)

window.show()
app.exec()
