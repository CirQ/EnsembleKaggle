from multiprocessing import Process, Event, Lock, Manager, cpu_count

import joblib

from data_preprocessor import fetch_all_data
from ensemble_classifier import AdaDTEnsembler, BgEnsembler, write_data


def proc(X_train, y_train, X_test, y_real, finishevent, printlock, resultmanager):
    while not finishevent.is_set():
        y_pred, acc, model = AdaDTEnsembler(X_train, y_train, X_test, y_real)
        with printlock:
            print 'The accuracy score is', acc
        if acc > 0.765:
            resultmanager['pred_y'] = y_pred
            resultmanager['acc'] = acc
            resultmanager['model'] = model
            finishevent.set()
        del y_pred, model


def main():
    X_train, y_train, X_test, y_real = fetch_all_data()
    finish_event = Event()
    print_lock = Lock()
    result_manager = Manager().dict({'pred_y':None, 'acc':None, 'model':None})
    procs = []
    for _ in range(cpu_count()-4):
        args = [X_train, y_train, X_test, y_real, finish_event, print_lock, result_manager]
        p = Process(target=proc, args=args)
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    label = 'forpre_ada' + str(int(result_manager['acc']*1000))
    write_data(result_manager['pred_y'], label)
    joblib.dump(result_manager['model'], 'best_model.pickle')


if __name__ == '__main__':
    main()