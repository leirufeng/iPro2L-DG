
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import tensorflow as tf

from sklearn.model_selection import KFold

from Model import *

from feature_encoding import *
from utils import *

if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # Read the training set
    train_pos_seqs = np.array(read_fasta('../data/strong.txt'))
    train_neg_seqs = np.array(read_fasta('../data/weak.txt'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)

    train_C2 = np.array(to_C2_code(train_seqs)).astype(np.float32)

    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_C2, train_properties_code), axis=1)

    train_label = np.array([1] * 1591 + [0] * 1791).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)


    BATCH_SIZE = 30
    EPOCHS = 300

    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    ten_all_performance = []

    # Cycle 10 times
    for k in range(10):
        print('*' * 30 + ' the ' + str(k + 1) + ' cycle ' + '*' * 30)

        model = build_model()

        all_performance = []

        # 5-fold cross-validations
        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
            print('*' * 30 + ' the ' + str(fold_count + 1) + ' fold ' + '*' * 30)

            trains, val = train[train_index], train[val_index]
            trains_label, val_label = train_label[train_index], train_label[val_index]

            model.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=True,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                      verbose=1)


            val_pred = model.predict(val, verbose=1)


            # Sn, Sp, Acc, MCC, AUC, F1-score
            Sn, Sp, Acc, MCC, f1_score = show_performance(val_label[:, 1], val_pred[:, 1])
            AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])

            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1-score = %f' % (Sn, Sp, Acc, MCC, AUC, f1_score))

            performance = [Sn, Sp, Acc, MCC, AUC, f1_score]
            all_performance.append(performance)

        del model
        all_performance = np.array(all_performance)
        all_mean_performance = np.mean(all_performance, axis=0)
        ten_all_performance.append(all_mean_performance)


    print('---------------------------------------------10-cycle-result---------------------------------------')
    print(np.array(ten_all_performance))
    print('---------------------------------------------10-cycle-mean-result---------------------------------------')
    performance_mean = performance_mean(np.array(ten_all_performance))
    pd.DataFrame(np.array(ten_all_performance)).to_csv('../files/2layer_10_cycle_result.csv',index=False)





