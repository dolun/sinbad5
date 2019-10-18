import sys
 
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QPushButton, QLineEdit
from PySide2.QtCore import QFile, QObject,QCoreApplication,Qt
 
class Form(QObject):
    """ """
    def __init__(self, ui_file, parent=None):
        super(Form, self).__init__(parent)
        # ui_file = QFile(ui_file)
        # ui_file.open(QFile.ReadOnly)
 
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        # ui_file.close()
 
        self.dock = self.window.dockView
        print(self.dock)
 
        btn = self.window.findChild(QPushButton, 'openButton')
        print(btn)
        btn.clicked.connect(self.ok_handler)
        self.window.show()
 
    def ok_handler(self):
        print('ok_handler')
 
if __name__ == '__main__':
    # https://stackoverflow.com/questions/56159475/qt-webengine-seems-to-be-initialized
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    form = Form('sinbad5.ui')
    sys.exit(app.exec_())
