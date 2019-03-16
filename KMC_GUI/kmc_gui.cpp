#include "kmc_gui.h"
#include "ui_kmc_gui.h"

KMC_GUI::KMC_GUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::KMC_GUI)
{
    ui->setupUi(this);
}

KMC_GUI::~KMC_GUI()
{
    delete ui;
}
