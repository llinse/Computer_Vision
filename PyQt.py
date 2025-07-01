import sys
from PyQt5.QtWidgets import QWidget, QApplication,QPushButton,QLineEdit,QInputDialog


# class Example
class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.InitUI()

    def InitUI(self):
        self.btn = QPushButton("Dialog", self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.ShowDialog)

        self.le = QLineEdit(self)
        self.le.move(130, 22)

        self.setWindowTitle("Input Dialog")
        self.show()

    def ShowDialog(self):
        text, ok = QInputDialog.getText(self, "Input Dialog", "Enter your name:")
        if ok:
            self.le.setText(str(text))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())