#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QFileDialog>
#include <QMainWindow>

#include "dataset.hpp"
#include "model.hpp"
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

  void labeling(Dataset &data, QCustomPlot *whichPlot = nullptr);
  void updateTrainingProgress(int epoch, double loss, double lr, double loss_change);
  void plotTrainingMetrics();

 private slots:

  void on_OpenFileButton_clicked();

  void on_trainButton_clicked();

  void on_labelButton_clicked();

  void on_recallButton_clicked();

 private:
  Ui::MainWindow *ui;

  int    maxEpochs    = 100000;
  double learningRate = 1e-2;
  double decay        = 0.95;
};
#endif  // MAINWINDOW_H
