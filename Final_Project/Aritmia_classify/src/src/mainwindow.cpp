#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    logModel = new QStringListModel(this);
    ui->DataLOG->setModel(logModel);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::logMessage(const QString &msg)
{
    logList.prepend(msg); // latest on top
    logModel->setStringList(logList);
}

void MainWindow::on_LoadData_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Dataset", "", "Text Files (*.txt)");
    if (fileName.isEmpty())
        return;

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qWarning() << "File open failed.";
        return;
    }

    QTextStream in(&file);
    QString header = in.readLine(); // skip header

    xData.clear();
    yData.clear();

    while (!in.atEnd())
    {
        QString line = in.readLine();
        QStringList tokens = line.split(QRegularExpression("[\\t,]"));
        if (tokens.size() < 2)
            continue;

        double rr = tokens[0].toDouble();
        double qrsd = tokens[1].toDouble();

        std::vector<double> input = {rr, qrsd};
        // Fix: use -> operator for unique_ptr
        int label = model->classifyArrhythmia(rr, qrsd);

        std::vector<double> onehot(7, 0.0);
        onehot[label] = 1.0;

        xData.push_back(input);
        yData.push_back(onehot);
    }

    file.close();
    logMessage("Loaded " + QString::number(xData.size()) + " samples.");
    qDebug() << "Loaded" << xData.size() << "samples.";

    // --- Plot raw data (RR vs QRSd) ---
    ui->plotterRaw->clearGraphs();

    QVector<double> rrVals, qrsdVals;
    for (const auto &input : xData)
    {
        rrVals.append(input[0]);
        qrsdVals.append(input[1]);
    }

    QVector<QColor> colors = {
        Qt::green, Qt::red, Qt::blue,
        Qt::magenta, Qt::cyan, Qt::yellow,
        Qt::darkRed};
    QStringList labels = {
        "Normal", "Dropped", "R-on-T", "Tachycardia", "Bradycardia", "Fusion", "PVC"};

    ui->plotterRaw->addGraph();
    ui->plotterRaw->graph(0)->setData(rrVals, qrsdVals);
    ui->plotterRaw->graph(0)->setLineStyle(QCPGraph::lsNone);
    ui->plotterRaw->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Qt::gray, 5));
    ui->plotterRaw->xAxis->setLabel("RR (s)");
    ui->plotterRaw->yAxis->setLabel("QRSd (ms)");
    ui->plotterRaw->rescaleAxes();
    ui->plotterRaw->replot();

    logMessage("Plotted raw RR vs QRSd data.");
}

void MainWindow::on_Train_clicked()
{
    if (xData.empty())
    {
        qWarning() << "No data loaded.";
        return;
    }

    // --- Get hidden neuron count from lineEdit ---
    bool ok;
    int hiddenNeurons = ui->lineEdit->text().toInt(&ok);

    if (!ok || hiddenNeurons <= 0)
    {
        logMessage("Invalid hidden neuron count.");
        qWarning() << "Invalid hidden neuron count.";
        return;
    }

    model = std::make_unique<Model>(2, hiddenNeurons, 7, 0.1);

    logMessage("Training started with " + QString::number(hiddenNeurons) + " hidden neurons.");

    qDebug() << "Training with" << hiddenNeurons << "hidden neurons.";
    qDebug() << "Training data size:" << xData.size();

    // --- Train model ---
    // Normalize data
    model->normalizeTrain(xData, rrMin, rrMax, qrsMin, qrsMax);
    model->train(xData, yData, 500);

    // --- Plot MSE ---
    std::vector<double> mse = model->getMSEHistory();
    logMessage("Training finished. Final MSE: " + QString::number(mse.back()));
    ui->plotMSE->clearGraphs();
    ui->plotMSE->addGraph();

    QVector<double> x(mse.size()), y(mse.size());
    for (int i = 0; i < mse.size(); ++i)
    {
        x[i] = i;
        y[i] = mse[i];
    }

    ui->plotMSE->graph(0)->setData(x, y);
    ui->plotMSE->xAxis->setLabel("Epoch");
    ui->plotMSE->yAxis->setLabel("MSE");
    ui->plotMSE->rescaleAxes();
    ui->plotMSE->replot();

    // --- Plot Accuracy ---
    std::vector<double> acc = model->getAccuracyHistory();
    logMessage("Training accuracy: " + QString::number(acc.back(), 'f', 2) + "%");
    ui->plotAccuracy->clearGraphs();
    ui->plotAccuracy->addGraph();

    QVector<double> xAcc(acc.size()), yAcc(acc.size());
    for (int i = 0; i < acc.size(); ++i)
    {
        xAcc[i] = i;
        yAcc[i] = acc[i];
    }

    ui->plotAccuracy->graph(0)->setData(xAcc, yAcc);
    ui->plotAccuracy->xAxis->setLabel("Epoch");
    ui->plotAccuracy->yAxis->setLabel("Accuracy (%)");
    ui->plotAccuracy->yAxis->setRange(0, 100);
    ui->plotAccuracy->rescaleAxes();
    ui->plotAccuracy->replot();

    // --- Plot Prediction Clusters ---
    logMessage("Plotted training results.");
    qDebug() << "Plotting training results...";
}

void MainWindow::on_Recall_clicked()
{
    QString path = QFileDialog::getOpenFileName(this, "Open Test Data", "", "Text Files (*.txt)");
    if (path.isEmpty())
        return;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream in(&file);
    QString header = in.readLine();

    X_test.clear();
    y_test_true.clear();

    while (!in.atEnd())
    {
        QString line = in.readLine();
        QStringList tokens = line.split(QRegularExpression("[\t,]"));
        if (tokens.size() < 2)
            continue;

        double rr = tokens[0].toDouble();
        double qrs = tokens[1].toDouble();
        int trueLabel = Model::classifyArrhythmia(rr, qrs);

        std::vector<double> input = {
            (rr - rrMin) / (rrMax - rrMin + 1e-9),
            (qrs - qrsMin) / (qrsMax - qrsMin + 1e-9)};

        X_test.push_back(input);
        y_test_true.push_back(trueLabel);
    }

    file.close();

    // Evaluate
    int correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i)
    {
        int pred = model->predict(X_test[i]);
        if (pred == y_test_true[i])
            correct++;
    }

    double acc = 100.0 * correct / X_test.size();
    logMessage("External test accuracy: " + QString::number(acc, 'f', 2) + "%");
      ui->plotterTrainned->clearGraphs();

    QVector<QColor> colors = {
        Qt::green, Qt::red, Qt::blue,
        Qt::magenta, Qt::cyan, Qt::yellow,
        Qt::darkRed};

    QStringList labels = {
        "Normal", "Dropped", "R-on-T", "Tachycardia", "Bradycardia", "Fusion", "PVC"};

    std::vector<QVector<double>> rr_by_class(7), qrsd_by_class(7);

    for (size_t i = 0; i < xData.size(); ++i)
    {
        int label = model->predict(xData[i]);
        rr_by_class[label].append(xData[i][0]);
        qrsd_by_class[label].append(xData[i][1]);
    }

    for (int i = 0; i < 7; ++i)
    {
        ui->plotterTrainned->addGraph();
        ui->plotterTrainned->graph(i)->setData(rr_by_class[i], qrsd_by_class[i]);
        ui->plotterTrainned->graph(i)->setLineStyle(QCPGraph::lsNone);
        ui->plotterTrainned->graph(i)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, colors[i], 6));
        ui->plotterTrainned->graph(i)->setName(labels[i]);
    }

    ui->plotterTrainned->xAxis->setLabel("RR (s)");
    ui->plotterTrainned->yAxis->setLabel("QRSd (ms)");
    ui->plotterTrainned->legend->setVisible(true);
    ui->plotterTrainned->rescaleAxes();
    ui->plotterTrainned->replot();
}
