import numpy as np

def load_tsv_dataset(file):
    return np.loadtxt(file, delimiter='\t', encoding='utf-8')


def sigmoid(x):
    e = np.exp(x)
    return e / (1 + e)


def sgd(theta, X, Y, learning_rate):
    # implement in vector form
    for i in range(X.shape[0]):
        x = X[i].reshape(1, X.shape[1])
        y = Y[i].reshape(1,1)
        z = x.T @ (sigmoid(x @ theta) - y)
        theta = theta - learning_rate * z
    return theta


def train(theta, X, y, num_epoch, learning_rate):
    for _ in range(num_epoch):
        theta = sgd(theta, X, y, learning_rate)
    return theta


def predict(theta, X):
    # implement in vector form
    pred = []
    y_pred = sigmoid(X @ theta)

    for i in y_pred:
        if i >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return np.array(pred)
    

def compute_error(y_pred, y):
    # implement in vector form
    error = 0
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            error += 1
    return error / len(y)


def write_metrics(train_err, test_err, metrics_out):
    with open(metrics_out, 'w+') as f:
        w = "error(train): " + "{:.6f}".format(train_err) + "\n"
        w += "error(test): " + "{:.6f}".format(test_err) + "\n"
        f.write(w)
    return


def logistic_reg(formatted_train, formatted_test, metrics_out, num_epochs, learning_rate):
    train_loaded = load_tsv_dataset(formatted_train)
    y = np.array([item[0] for item in train_loaded])
    X = np.array([item[1:] for item in train_loaded])

    theta = np.zeros((X[0].size, 1))

    learned_theta = train(theta, X, y, num_epochs, learning_rate)
    train_pred = predict(learned_theta, X)
    train_err = compute_error(train_pred, y)

    test_loaded = load_tsv_dataset(formatted_test)
    X_test = np.array([item[1:] for item in test_loaded])
    y_test = np.array([item[0] for item in test_loaded])

    test_pred = predict(learned_theta, X_test)
    test_err = compute_error(test_pred, y_test)

    write_metrics(train_err, test_err, metrics_out)
    return


logistic_reg("./data/output/embedded_train_small.tsv", "./data/output/embedded_test_small.tsv", "./data/output/metrics.txt", 500, 0.001)