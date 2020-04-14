from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtWidgets, QtCore, QtGui


def get_font(num):
    font = QtGui.QFont()
    font.setFamily("宋体")
    font.setPointSize(num)
    return font


class Printable:
    def __init__(self, func):
        self.write = func


class FormUI(QWidget):
    def __init__(self):
        super(FormUI, self).__init__()
        self.setWindowTitle("SVM识别XSS注入")
        self.setObjectName("Form")
        self.resize(400, 200)
        self.btn_check = QtWidgets.QPushButton()
        self.btn_check.setObjectName("btn_check")
        self.txt_input = QtWidgets.QLineEdit()
        self.txt_input.setObjectName("txt_input")

        self.tbr_show = QtWidgets.QTextBrowser()
        self.tbr_show.setObjectName("tbr_show")

        layout = QtWidgets.QVBoxLayout(self)

        _layout = QtWidgets.QHBoxLayout()
        _layout.addSpacerItem(QtWidgets.QSpacerItem(0, 0, hPolicy=QtWidgets.QSizePolicy.MinimumExpanding))
        lb_info = QtWidgets.QLabel("XSS注入检测")
        lb_info.resize(400, 200)
        lb_info.setFont(get_font(20))
        _layout.addWidget(lb_info)
        _layout.addSpacerItem(QtWidgets.QSpacerItem(0, 0, hPolicy=QtWidgets.QSizePolicy.MinimumExpanding))
        layout.addLayout(_layout)

        _layout = QtWidgets.QHBoxLayout()
        _layout.addWidget(self.txt_input)
        _layout.addWidget(self.btn_check)
        layout.addLayout(_layout)

        _layout = QtWidgets.QVBoxLayout(self)
        _layout.addWidget(self.tbr_show)

        layout.addLayout(_layout)
        self.setLayout(layout)
        self.translate()

        QtCore.QMetaObject.connectSlotsByName(self)

    def translate(self):
        _translate = QtCore.QCoreApplication.translate
        self.btn_check.setText(_translate("Form", "Check"))


class Form(FormUI):
    def __init__(self):
        super(Form, self).__init__()
        from train import Detect
        self.detect = Detect()

    @QtCore.pyqtSlot()
    def on_btn_check_clicked(self):
        data = self.txt_input.text()
        r = self.detect.predict(data)
        out = "识别语句为：{} \nthe checking result is {}".format(data, r)
        writer = Printable(self.tbr_show.append)
        print(out, file=writer)
        self.txt_input.clear()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec_())
