#include "mainwindow.h"

#include <QFile>
#include <QMessageBox>
#include <QRegularExpression>
#include <QStringListModel>
#include <QVector>

#include "./ui_mainwindow.h"

QVector<double> xData1, xData2, y1Data1, y2Data1, y1Data2, y2Data2;

Dataset traindata, testCase;
MLP     MLGait;

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  ui->rawPlot->setInteractions(QCP::iRangeDrag | QCP::iSelectPlottables | QCP::iRangeZoom);
  ui->trainPlot->setInteractions(QCP::iRangeDrag | QCP::iSelectPlottables | QCP::iRangeZoom);

  QStringListModel *logModel = new QStringListModel(this);
  logModel->setStringList(QStringList());  // Start with an empty list
  ui->logInfo->setModel(logModel);
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::on_OpenFileButton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Select File", "", "All Files (*)");
  if (filename.isEmpty())
    return;

  QFile file(filename);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::critical(this, "Error", "Gagal membuka file!");
    return;
  }

  QTextStream in(&file);
  QStringList lines;

  while (!in.atEnd()) {
    QString line = in.readLine();
    lines << line;
  }

  file.close();

  QStringListModel *model = new QStringListModel(this);
  model->setStringList(lines);

  ui->RawData->setModel(model);

  int index = 0;
  for (const QString &line : lines) {
    QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    if (parts.size() >= 2) {
      bool   ok1, ok2;
      double y1 = parts[0].toDouble(&ok1);
      double y2 = parts[1].toDouble(&ok2);
      if (ok1 && ok2) {
        xData1.append(index++);
        y1Data1.append(y1);
        y2Data1.append(y2);
        int label = traindata.classifyPhase(y1, y2);
        traindata.addDataPoint(y1, y2, label);
      }
    }
  }

  ui->rawPlot->clearGraphs();

  // Graph 1
  ui->rawPlot->addGraph();
  ui->rawPlot->graph(0)->setName("FSR Heel");
  ui->rawPlot->graph(0)->setData(xData1, y1Data1);
  ui->rawPlot->graph(0)->setPen(QPen(Qt::blue));

  // Graph 2
  ui->rawPlot->addGraph();
  ui->rawPlot->graph(1)->setName("FSR Toe");
  ui->rawPlot->graph(1)->setData(xData1, y2Data1);
  ui->rawPlot->graph(1)->setPen(QPen(Qt::red));

  ui->rawPlot->xAxis->setLabel("sample");
  ui->rawPlot->yAxis->setLabel("FSR Value");
  ui->rawPlot->legend->setVisible(true);
  ui->rawPlot->legend->setBrush(QColor(255, 255, 255, 150));
  ui->rawPlot->rescaleAxes();
  ui->rawPlot->replot();
}

void MainWindow::on_trainButton_clicked() {
  Dataset augmented = traindata;

  MLGait = MLP(learningRate);

  // Count the occurrences of each class
  int min_count = 999;
  int max_count = 0;

  std::vector<std::pair<std::vector<double>, int>> augmented_data;

  int class_distribution[6] = {0};

  for (const auto &point : traindata.data) {
    if (point.label >= 0 && point.label < 6) {
      class_distribution[point.label]++;
      min_count = std::min(min_count, class_distribution[point.label]);
      max_count = std::max(max_count, class_distribution[point.label]);
    }
  }

  augmented.balanceDataset(augmented);

  std::cout << "Class distribution before augmentation:" << std::endl;
  for (int i = 0; i < 6; ++i) {
    std::cout << "Class " << i << ": " << class_distribution[i] << " samples" << std::endl;
  }

  // Augment the dataset to balance classes
  augmented.duplicateAndAugment(3);  // Use a factor of x for all data

  // Convert the augmented Dataset to the format expected by the model
  for (auto &point : augmented.data) {
    augmented_data.push_back({{point.x1, point.x2}, point.label});
  }

  MLGait.train(augmented_data, maxEpochs);
  // After training:
  int correct = 0;
  for (auto &point : augmented.data) {
    int pred = MLGait.predictClass({point.x1, point.x2});
    if (pred == point.label)
      correct++;
  }
  double testAcc = 100.0 * correct / augmented.data.size();
  std::cout << "Test accuracy: " << testAcc << "%" << std::endl;

  QString trainAccMsg = QString("Train Accuracy: %1% (%2/%3)").arg(testAcc, 0, 'f', 2).arg(correct).arg(augmented.data.size());

  QStringListModel *accModel = new QStringListModel(this);
  accModel->setStringList(QStringList() << trainAccMsg);
  ui->AccuracyLog->setModel(accModel);

  plotTrainingMetrics();
}

void MainWindow::updateTrainingProgress(int epoch, double loss, double lr, double loss_change) {
  // Update progress text
  QString progressText =
      QString("Epoch %1: MSE = %2 (lr = %3, change: %4)").arg(epoch).arg(loss, 0, 'f', 6).arg(lr, 0, 'f', 6).arg(loss_change, 0, 'f', 6);

  // Get the model and add to it
  QStringListModel *model = qobject_cast<QStringListModel *>(ui->logInfo->model());
  if (model) {
    QStringList entries = model->stringList();
    entries.append(progressText);
    model->setStringList(entries);

    // Scroll to show the new entry
    ui->logInfo->scrollToBottom();
  }

  // Process events to keep UI responsive
  QApplication::processEvents();
}

void MainWindow::labeling(Dataset &data, QCustomPlot *whichPlot) {
  // Process data to classify it into phases
  data.dataToClassify();

  // Count data points in each class
  int class_counts[6] = {0};
  for (const auto &d : data.data) {
    if (d.label >= 0 && d.label < 6)
      class_counts[d.label]++;
  }

  // Log the counts
  for (int i = 0; i < 6; ++i) {
    std::cout << "Label " << i << ": " << class_counts[i] << " data\n";
  }

  // Define colors for the different gait phases
  QVector<QColor> colors = {
      QColor(255, 0, 0),    // Label 0 (ic) - Red
      QColor(0, 255, 0),    // Label 1 (ff) - Green
      QColor(0, 0, 255),    // Label 2 (ho) - Blue
      QColor(255, 255, 0),  // Label 3 (mst) - Yellow
      QColor(255, 0, 255),  // Label 4 (to) - Magenta
      QColor(0, 255, 255)   // Label 5 (sw) - Cyan
  };

  // Add markers for each phase without clearing the existing graphs
  for (int i = 0; i < xData1.size(); i++) {
    int label = data.classifyPhase(y1Data1[i], y2Data1[i]);

    if (label >= 0 && label < 6) {
      // Create markers at each actual data point

      // Marker for FSR Heel (y1Data1)
      QCPItemEllipse *markerHeel = new QCPItemEllipse(whichPlot);
      markerHeel->topLeft->setCoords(xData1[i] - 0.15, y1Data1[i] + 0.03);
      markerHeel->bottomRight->setCoords(xData1[i] + 0.15, y1Data1[i] - 0.03);
      markerHeel->setPen(QPen(colors[label]));
      markerHeel->setBrush(QBrush(colors[label], Qt::SolidPattern));

      // Marker for FSR Toe (y2Data1)
      QCPItemEllipse *markerToe = new QCPItemEllipse(whichPlot);
      markerToe->topLeft->setCoords(xData1[i] - 0.15, y2Data1[i] + 0.03);
      markerToe->bottomRight->setCoords(xData1[i] + 0.15, y2Data1[i] - 0.03);
      markerToe->setPen(QPen(colors[label]));
      markerToe->setBrush(QBrush(colors[label], Qt::SolidPattern));
    }
  }

  // Add a legend explaining the phase colors
  QString phaseNames[] = {"Initial Contact", "Foot Flat", "Heel Off", "Mid Stance", "Toe Off", "Swing"};

  // Add text labels for the phases at the top of the plot
  for (int i = 0; i < 6; i++) {
    QCPItemText *phaseText = new QCPItemText(whichPlot);
    phaseText->setPositionAlignment(Qt::AlignTop | Qt::AlignHCenter);
    phaseText->position->setType(QCPItemPosition::ptAxisRectRatio);
    phaseText->position->setCoords(0.1 + i * 0.15, 0.05);  // Spaced evenly across the top

    // Style the text
    phaseText->setText(phaseNames[i]);
    phaseText->setColor(colors[i]);
    phaseText->setPen(QPen(colors[i]));
    phaseText->setBrush(QBrush(QColor(255, 255, 255, 100)));
    phaseText->setPadding(QMargins(2, 2, 2, 2));
  }

  // Rescale and display
  whichPlot->replot();
}

void MainWindow::on_labelButton_clicked() {
  labeling(traindata, ui->rawPlot);
}

void MainWindow::on_recallButton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Select File", "", "All Files (*)");
  if (filename.isEmpty())
    return;

  QFile file(filename);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::critical(this, "Error", "Gagal membuka file!");
    return;
  }

  QTextStream in(&file);
  QStringList lines;

  while (!in.atEnd()) {
    QString line = in.readLine();
    lines << line;
  }

  file.close();

  int             index = 0;
  QVector<double> xData2, y1Data2, y2Data2;
  testCase.data.clear();  // Clear previous test case data

  for (const QString &line : lines) {
    QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    if (parts.size() >= 2) {
      bool   ok1, ok2;
      double y1 = parts[0].toDouble(&ok1);
      double y2 = parts[1].toDouble(&ok2);
      if (ok1 && ok2) {
        xData2.append(index++);
        y1Data2.append(y1);
        y2Data2.append(y2);
        int label = testCase.classifyPhase(y1, y2);  // Classify phase
        testCase.addDataPoint(y1, y2, label);
      }
    }
  }

  testCase.dataToClassify();  // Prepare the test case data for classification

  // Create data for the model
  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto &point : testCase.data) {
    testData.push_back({{point.x1, point.x2}, point.label});
  }
  // Classify the test data
  int correct = 0;
  int count   = 1;

  for (const auto &sample : testData) {
    std::vector<double> output_vector = MLGait.predict(sample.first[0], sample.first[1]);
    int                 predicted     = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));

    // Convert predicted label to string
    QString predictLabel;
    switch (predicted) {
    case 0:
      predictLabel = "IC";
      break;
    case 1:
      predictLabel = "FF";
      break;
    case 2:
      predictLabel = "HO";
      break;
    case 3:
      predictLabel = "MSt";
      break;
    case 4:
      predictLabel = "TO";
      break;
    case 5:
      predictLabel = "Sw";
      break;
    default:
      predictLabel = "Unknown";
      break;
    }

    // Convert target label to string
    QString targetLabel;
    switch (sample.second) {
    case 0:
      targetLabel = "IC";
      break;
    case 1:
      targetLabel = "FF";
      break;
    case 2:
      targetLabel = "HO";
      break;
    case 3:
      targetLabel = "MSt";
      break;
    case 4:
      targetLabel = "TO";
      break;
    case 5:
      targetLabel = "Sw";
      break;
    default:
      targetLabel = "Unknown";
      break;
    }

    QString     outputStr;
    QTextStream stream(&outputStr);
    stream << "Output: ";
    QStringList labels = {"IC", "FF", "HO", "MSt", "TO", "Sw"};

    for (size_t i = 0; i < output_vector.size(); ++i) {
      if (i < labels.size()) {
        stream << labels[i] << " (" << qRound(output_vector[i] * 100) << "%) ";
      }
    }

    // Add entry to log
    QString log_entry = QString("Sample %1: Input: (%2, %3) | Target: %4 | Predicted: %5")
                            .arg(count)
                            .arg(sample.first[0], 0, 'f', 2)
                            .arg(sample.first[1], 0, 'f', 2)
                            .arg(targetLabel)
                            .arg(predictLabel);

    ui->logInfo->model()->insertRow(ui->logInfo->model()->rowCount());
    ui->logInfo->model()->setData(ui->logInfo->model()->index(ui->logInfo->model()->rowCount() - 1, 0), log_entry);

    // Add output probabilities on the next line
    ui->logInfo->model()->insertRow(ui->logInfo->model()->rowCount());
    ui->logInfo->model()->setData(ui->logInfo->model()->index(ui->logInfo->model()->rowCount() - 1, 0), "    " + outputStr);

    if (predicted == sample.second)
      correct++;

    count++;
  }

  double  accuracy        = 100.0 * correct / testData.size();
  QString accuracyMessage = QString("Accuracy: %1% (%2/%3 correct)").arg(accuracy, 0, 'f', 2).arg(correct).arg(testData.size());

  QString recallAccMsg = QString("Recall Accuracy: %1% (%2/%3)").arg(accuracy, 0, 'f', 2).arg(correct).arg(testData.size());

  QStringListModel *recAccModel = new QStringListModel(this);
  recAccModel->setStringList(QStringList() << recallAccMsg);
  ui->RecAccLog->setModel(recAccModel);

  // Add empty line and accuracy to log
  ui->logInfo->model()->insertRow(ui->logInfo->model()->rowCount());
  ui->logInfo->model()->setData(ui->logInfo->model()->index(ui->logInfo->model()->rowCount() - 1, 0), "");

  ui->logInfo->model()->insertRow(ui->logInfo->model()->rowCount());
  ui->logInfo->model()->setData(ui->logInfo->model()->index(ui->logInfo->model()->rowCount() - 1, 0), accuracyMessage);

  // Clear the plot before adding new data
  ui->trainPlot->clearGraphs();

  // Graph 1 - FSR Heel
  ui->trainPlot->addGraph();
  ui->trainPlot->graph(0)->setName("FSR Heel");
  ui->trainPlot->graph(0)->setData(xData2, y1Data2);
  ui->trainPlot->graph(0)->setPen(QPen(Qt::blue));

  // Graph 2 - FSR Toe
  ui->trainPlot->addGraph();
  ui->trainPlot->graph(1)->setName("FSR Toe");
  ui->trainPlot->graph(1)->setData(xData2, y2Data2);
  ui->trainPlot->graph(1)->setPen(QPen(Qt::red));

  // Set up the plot
  ui->trainPlot->xAxis->setLabel("sample");
  ui->trainPlot->yAxis->setLabel("FSR Value");
  ui->trainPlot->legend->setVisible(true);
  ui->trainPlot->legend->setBrush(QColor(255, 255, 255, 150));
  ui->trainPlot->rescaleAxes();
  ui->trainPlot->replot();

  // Add markers for labels
  QVector<QColor> colors = {
      QColor(255, 0, 0),    // Label 0 (ic) - Red
      QColor(0, 255, 0),    // Label 1 (ff) - Green
      QColor(0, 0, 255),    // Label 2 (ho) - Blue
      QColor(255, 255, 0),  // Label 3 (mst) - Yellow
      QColor(255, 0, 255),  // Label 4 (to) - Magenta
      QColor(0, 255, 255)   // Label 5 (sw) - Cyan
  };

  for (int i = 0; i < xData2.size(); i++) {
    int label = testCase.classifyPhase(y1Data2[i], y2Data2[i]);

    if (label >= 0 && label < 6) {
      // Marker for FSR Heel (y1Data2)
      QCPItemEllipse *markerHeel = new QCPItemEllipse(ui->trainPlot);
      markerHeel->topLeft->setCoords(xData2[i] - 0.15, y1Data2[i] + 0.03);
      markerHeel->bottomRight->setCoords(xData2[i] + 0.15, y1Data2[i] - 0.03);
      markerHeel->setPen(QPen(colors[label]));
      markerHeel->setBrush(QBrush(colors[label], Qt::SolidPattern));

      // Marker for FSR Toe (y2Data2)
      QCPItemEllipse *markerToe = new QCPItemEllipse(ui->trainPlot);
      markerToe->topLeft->setCoords(xData2[i] - 0.15, y2Data2[i] + 0.03);
      markerToe->bottomRight->setCoords(xData2[i] + 0.15, y2Data2[i] - 0.03);
      markerToe->setPen(QPen(colors[label]));
      markerToe->setBrush(QBrush(colors[label], Qt::SolidPattern));
    }
  }

  // Add phase legend
  QString phaseNames[] = {"Initial Contact", "Foot Flat", "Heel Off", "Mid Stance", "Toe Off", "Swing"};
  for (int i = 0; i < 6; i++) {
    QCPItemText *phaseText = new QCPItemText(ui->trainPlot);
    phaseText->setPositionAlignment(Qt::AlignTop | Qt::AlignHCenter);
    phaseText->position->setType(QCPItemPosition::ptAxisRectRatio);
    phaseText->position->setCoords(0.1 + i * 0.15, 0.05);
    phaseText->setText(phaseNames[i]);
    phaseText->setColor(colors[i]);
    phaseText->setPen(QPen(colors[i]));
    phaseText->setBrush(QBrush(QColor(255, 255, 255, 100)));
    phaseText->setPadding(QMargins(2, 2, 2, 2));
  }

  ui->trainPlot->replot();
  // Apply markers to the plot
  labeling(testCase, ui->trainPlot);
}

void MainWindow::plotTrainingMetrics() {
  // Get the history data
  const std::vector<double> &loss_history     = MLGait.getLossHistory();
  const std::vector<double> &accuracy_history = MLGait.getAccuracyHistory();

  // Make sure we have data to plot
  if (loss_history.empty()) {
    QMessageBox::warning(this, "Warning", "No training history available to plot.");
    return;
  }

  // Create vectors for plotting
  QVector<double> epochs;
  QVector<double> loss;
  QVector<double> accuracy;

  // Fill the vectors with data
  for (size_t i = 0; i < loss_history.size(); ++i) {
    epochs.append(i + 1);  // Epoch numbers starting from 1
    loss.append(loss_history[i]);

    if (i < accuracy_history.size()) {
      accuracy.append(accuracy_history[i]);
    }
  }

  // Create a new dialog for the plot
  QDialog *plotDialog = new QDialog(this);
  plotDialog->setWindowTitle("Training Metrics");
  plotDialog->resize(800, 600);

  // Create layout
  QVBoxLayout *layout = new QVBoxLayout(plotDialog);

  // Create the plot widget
  QCustomPlot *metricsPlot = new QCustomPlot(plotDialog);
  layout->addWidget(metricsPlot);

  // Set up the left y-axis for MSE
  metricsPlot->yAxis->setLabel("Mean Squared Error");

  // Set up right y-axis for accuracy
  metricsPlot->yAxis2->setVisible(true);
  metricsPlot->yAxis2->setLabel("Accuracy (%)");
  metricsPlot->yAxis2->setRange(0, 100);  // Accuracy from 0-100%

  // Set up x axis
  metricsPlot->xAxis->setLabel("Epoch");

  // Add the loss graph (left y-axis)
  metricsPlot->addGraph();
  metricsPlot->graph(0)->setName("MSE Loss");
  metricsPlot->graph(0)->setData(epochs, loss);
  metricsPlot->graph(0)->setPen(QPen(Qt::blue, 2));

  // Add the accuracy graph using the right y-axis
  metricsPlot->addGraph(metricsPlot->xAxis, metricsPlot->yAxis2);
  metricsPlot->graph(1)->setName("Training Accuracy");
  metricsPlot->graph(1)->setData(epochs, accuracy);
  metricsPlot->graph(1)->setPen(QPen(Qt::red, 2));

  // Set axes ranges
  metricsPlot->xAxis->rescale();
  metricsPlot->yAxis->rescale();

  metricsPlot->legend->setVisible(true);
  metricsPlot->legend->setBrush(QColor(255, 255, 255, 200));
  metricsPlot->legend->setSelectableParts(QCPLegend::spNone);

  metricsPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

  metricsPlot->replot();

  QPushButton *closeButton = new QPushButton("Close", plotDialog);
  layout->addWidget(closeButton);
  connect(closeButton, &QPushButton::clicked, plotDialog, &QDialog::accept);

  plotDialog->setLayout(layout);
  plotDialog->exec();
}
