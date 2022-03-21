import os
import numpy as np
import matplotlib.pyplot as plt
from model import SSResNet
from HSIDataset import HSIDataset, DataInfo
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import spectral

RUN = 10
EPOCHS = 200
BATCHSZ = 16
LR = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
NUM_FILTERS = list(range(8, 33, 8))
PATCHSZ = [5, 7, 9, 11]
DATA_NAME = ['IN', 'KSC', 'UP']

def report(model, x_test, y_test, target_names):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    class_acc = classification_report(y_test, pred, target_names=target_names)
    confusion_mat = confusion_matrix(y_test, pred)
    score = model.evaluate(x_test, y_test, batch_size=32)
    test_loss = score[0]
    test_acc = score[1] * 100
    return class_acc, confusion_mat, test_loss, test_acc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = Normalized
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(run, data_name, patchsz, lr, num_filters, is_bn, is_dropout):
    HSI = HSIDataset(dataset_name=data_name, patchsz=patchsz, random_state=run)
    x_test, y_test = HSI.x_test, HSI.y_test
    nc = np.max(y_test) + 1
    model = SSResNet(nc=nc, filters=num_filters, is_bn=is_bn, is_dropout=is_dropout)
    rms_prop = RMSprop(learning_rate=lr)
    model.compile(optimizer=rms_prop, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    model_path = './model_ckp/{}/{}/patchsz-{}-lr-{}-num_filters-{}-is_bn-{}-is_dropout-{}'.format(
        run, data_name, patchsz, lr, num_filters, is_bn, is_dropout)
    assert os.path.exists(model_path + '.index')
    print('*'*5, 'loading trained model', '*'*5)
    model.load_weights(model_path)
    info = DataInfo.info[data_name]
    target_names = info['target_names']
    class_acc, cm, test_loss, test_acc = report(model, x_test, y_test, target_names)
    class_acc = str(class_acc)
    cm_str = str(cm)
    print('Test loss:{}'.format(test_loss))
    print('Test acc:{}%'.format(test_acc))
    print('Classification result:')
    print(class_acc)
    print('Confusion matrix:')
    print(cm_str)

    report_save_path = './test result/{}/{}/patchsz-{}-lr-{}-num_filters-{}-is_bn-{}-is_dropout-{}'.format(
        run, data_name, patchsz, lr, num_filters, is_bn, is_dropout)
    if not os.path.exists(report_save_path):
        os.makedirs(report_save_path)
    file_name = os.path.join(report_save_path, 'report.txt')
    with open(file_name, 'w') as f:
        f.write('Test loss:{}'.format(test_loss))
        f.write('\n')
        f.write('Test acc:{}%'.format(test_acc))
        f.write('\n')
        f.write('\n')
        f.write('Classification result:\n')
        f.write('{}'.format(class_acc))
        f.write('\n')
        f.write('Confusion matrix:\n')
        f.write('{}'.format(cm_str))
        f.write('\n')
    print('-------------successfully create report.txt!-------------------')

    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cm, classes=target_names, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(report_save_path, 'confusion_mat_without_norm.png'))
    print('------------succesfully generate confusion matrix pic!-----------')

    x_all, y_all = HSI.x_patch, HSI.y_patch
    label = HSI.y
    nonzero = np.nonzero(label)
    sample_ind = list(zip(*nonzero))
    num_sample = len(sample_ind)
    pred_map = np.zeros_like(label)
    pred = model.predict(x_all, batch_size=32)
    pred = np.argmax(pred, axis=1)
    for i, (x, y) in enumerate(sample_ind):
        pred_map[x, y] = pred[i] + 1
    predict_image = spectral.imshow(classes=pred_map.astype(int), figsize=(10, 10))
    plt.savefig(os.path.join(report_save_path, 'pred_map.jpg'))
    print('------------------sucessfully saved pred map!----------------------')
    return test_acc


# first test SSRN in different LR condition
for name in DATA_NAME:
    for lr in LR:
        acc_list = []
        for run in range(RUN):
            test_acc = main(run=run, data_name=name, patchsz=7, lr=lr, num_filters=24, is_bn=True, is_dropout=True)
            acc_list.append(test_acc)
        mean = np.mean(acc_list)
        std = np.std(acc_list)
        print('data=', name)
        print('lr={}'.format(lr))
        print('test acc mean:{}%'.format(mean))
        print('test acc std:{}'.format(std))

# test SSRN with different num filters
for name in DATA_NAME:
    for n in NUM_FILTERS:
        acc_list = []
        for run in range(RUN):
            test_acc = main(run=run, data_name=name, patchsz=7, lr=3e-4, num_filters=n, is_bn=True, is_dropout=True)
            acc_list.append(test_acc)
        mean = np.mean(acc_list)
        std = np.std(acc_list)
        print('data=', name)
        print('num filters={}'.format(n))
        print('test acc mean:{}%'.format(mean))
        print('test acc std:{}'.format(std))


# test SSRN with and without BN and dropout
for name in DATA_NAME:
    acc_list = []
    for run in range(RUN):
        test_acc = main(run=run, data_name=name, patchsz=7, lr=3e-4, num_filters=24, is_bn=True, is_dropout=True)
        acc_list.append(test_acc)
    mean = np.mean(acc_list)
    std = np.std(acc_list)
    print('data=', name)
    print('BN and Dropout')
    print('test acc mean:{}%'.format(mean))
    print('test acc std:{}'.format(std))

for name in DATA_NAME:
    acc_list = []
    for run in range(RUN):
        test_acc = main(run=run, data_name=name, patchsz=7, lr=3e-4, num_filters=24, is_bn=True, is_dropout=False)
        acc_list.append(test_acc)
    mean = np.mean(acc_list)
    std = np.std(acc_list)
    print('data=', name)
    print('BN')
    print('test acc mean:{}%'.format(mean))
    print('test acc std:{}'.format(std))

for name in DATA_NAME:
    acc_list = []
    for run in range(RUN):
        test_acc = main(run=run, data_name=name, patchsz=7, lr=3e-4, num_filters=24, is_bn=False, is_dropout=True)
        acc_list.append(test_acc)
    mean = np.mean(acc_list)
    std = np.std(acc_list)
    print('data=', name)
    print('Dropout')
    print('test acc mean:{}%'.format(mean))
    print('test acc std:{}'.format(std))

for name in DATA_NAME:
    acc_list = []
    for run in range(RUN):
        test_acc = main(run=run, data_name=name, patchsz=7, lr=3e-4, num_filters=24, is_bn=False, is_dropout=False)
        acc_list.append(test_acc)
    mean = np.mean(acc_list)
    std = np.std(acc_list)
    print('data=', name)
    print('No BN and Dropout')
    print('test acc mean:{}%'.format(mean))
    print('test acc std:{}'.format(std))

# TEST SSRN with different input patch size
for name in DATA_NAME:
    for p in PATCHSZ:
        acc_list = []
        for run in range(RUN):
            test_acc = main(run=run, data_name=name, patchsz=p, lr=3e-4, num_filters=24, is_bn=True, is_dropout=True)
            acc_list.append(test_acc)
        mean = np.mean(acc_list)
        std = np.std(acc_list)
        print('data=', name)
        print('patch size={}'.format(p))
        print('test acc mean:{}%'.format(mean))
        print('test acc std:{}'.format(std))



