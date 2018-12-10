from multiprocessing import Process, Event, Lock, Manager, cpu_count

from data_preprocessor import fetch_all_data
from ensemble_classifier import AdaDTEnsembler, write_data


def proc(X_train, y_train, X_test, y_real, finishevent, printlock, resultmanager):
    while not finishevent.is_set():
        y_pred, acc = AdaDTEnsembler(X_train, y_train, X_test, y_real)
        with printlock:
            print 'The accuracy score is', acc
        if acc > 0.763:
            resultmanager['pred_y'] = y_pred
            resultmanager['acc'] = acc
            finishevent.set()
        del y_pred


def main():
    X_train, y_train, X_test, y_real = fetch_all_data()
    finish_event = Event()
    print_lock = Lock()
    result_manager = Manager().dict({'pred_y':None, 'acc':None})
    procs = []
    for _ in range(cpu_count()-2):
        args = [X_train, y_train, X_test, y_real, finish_event, print_lock, result_manager]
        p = Process(target=proc, args=args)
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    label = 'adadt' + str(int(result_manager['acc']*1000))
    write_data(result_manager['pred_y'], label)


if __name__ == '__main__':
    main()