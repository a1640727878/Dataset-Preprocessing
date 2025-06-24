from PySide6.QtWidgets import QApplication, QWidget, QTabWidget

class Main_Gui:
    def __init__(self):
        self.title = "兔兔的妙妙小工具"
        self.app = QApplication()
        self.window = QWidget()
        self.tabWidget = QTabWidget(self.window)
        self.tabWidget.setGeometry(0, 0, 700, 500)
        self.window.setWindowTitle(self.title)
        self.window.setFixedSize(700, 500)

    def set_title(self, title):
        self.title = title
        self.window.setWindowTitle(self.title)

    def add_main_tab_widget(self, tags: list[tuple[QWidget, str]]):
        for widget, title in tags:
            self.tabWidget.addTab(widget, title)

    def get_app(self):
        return self.app

    def get_window(self):
        return self.window

    def show(self):
        self.window.show()
        self.app.exec()


if __name__ == "__main__":
    gui = Main_Gui()
    gui.show()
