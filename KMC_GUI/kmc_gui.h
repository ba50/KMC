#ifndef KMC_GUI_H
#define KMC_GUI_H

#include <QMainWindow>

namespace Ui {
class KMC_GUI;
}

class KMC_GUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit KMC_GUI(QWidget *parent = nullptr);
    ~KMC_GUI();

private:
    Ui::KMC_GUI *ui;
};

#endif // KMC_GUI_H
