#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "model.h"
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include "qcustomplot.h"
#include <QRegularExpression>
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_LoadData_clicked();

    void on_Train_clicked();

    void on_Recall_clicked();

private:
    Ui::MainWindow *ui;
    QStringListModel *logModel;
    QStringList logList;

    std::unique_ptr<Model> model; // smart pointer
    std::vector<std::vector<double>> xData;
    std::vector<std::vector<double>> yData;

    std::vector<std::vector<double>> X_test;
    std::vector<int> y_test_true;
    double rrMin, rrMax, qrsMin, qrsMax; // For normalization reuse

    void logMessage(const QString &msg);
};
#endif // MAINWINDOW_H
