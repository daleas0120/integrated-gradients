
import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def timestamp():
    time_cur = datetime.datetime.now()
    print("datetime : ", time_cur.strftime("%y%m/%d/%H:%M%S"))
    stamp = time_cur.strftime("%y%m%d%H%M%S")
    return stamp

def plot_scatter_with_error(y, pred_y, pred_std):
    plt.errorbar(y, pred_y, yerr= pred_std, fmt='o', markersize=10, ecolor="black", markeredgecolor ="black",color="b")
    y_max = max(y.max(), (pred_y + pred_std).max())
    y_min = min(y.min(), (pred_y - pred_std).min())
    plt.plot([y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min)],
            [y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min)], "k-")
    plt.xlim(y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min))
    plt.ylim(y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min))
    plt.xlabel("actual y")
    plt.ylabel("estimated_y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

def draw_scatter(y, y_pred):
    plt.rcParams["font.size"] = 18
    plt.scatter(y, y_pred, c="blue")
    y_max = max(y.max(), y_pred.max())
    y_min = min(y.min(), y_pred.min())
    plt.plot([y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min)],
            [y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min)], "k-")
    plt.xlim(y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min))
    plt.ylim(y_min - 0.05*(y_max - y_min),y_max + 0.05*(y_max - y_min))
    plt.xlabel("actual y")
    plt.ylabel("estimated y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()  
    
def mean_absolute_percentage_error(y, y_pred):
    return np.mean(np.abs((y - y_pred)/y_pred))*100

def score(y, y_pred):
    print(f"r2:{r2_score(y, y_pred):.3}")
    print(f"RMSE:{mean_squared_error(y, y_pred):.3}")
    print(f"MAE:{mean_absolute_error(y, y_pred):.3}")
    try:
        print(f"MAPE:{mean_absolute_percentage_error(y, y_pred):.3}")
    except:
        pass

def confidence_curve(y_true, y_pred, sigma):
    errors = np.abs(y_true - y_pred)
    sorted_indices = np.argsort(sigma)
    sorted_errors = errors[sorted_indices]
    fractions = np.linspace(0, 1, len(y_true)+1) # modified
    confidence_errors = [np.mean(sorted_errors[:int(f * len(sorted_errors))]) for f in fractions]
    return fractions, confidence_errors[::-1]

def oracle_curve(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    sorted_errors = np.sort(errors)
    fractions = np.linspace(0, 1, len(y_true+1)) # modified
    oracle_errors = [np.mean(sorted_errors[:int(f * len(sorted_errors))]) for f in fractions]
    return fractions, oracle_errors[::-1]

def calculate_auco(confidence_errors, oracle_errors):
    len_arr = min(len(confidence_errors), len(oracle_errors))
    differences = np.abs(np.array(confidence_errors[:len_arr]) - np.array(oracle_errors[:len_arr]))

    return np.mean(differences[:-1])
