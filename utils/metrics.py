import re
import numpy as np
from poicontrastive import POIContrastiveLoss
from scipy import linalg

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return np.sqrt(np.mean(np.power(real - pred, 2)))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    
def get_NRMSE(pred, real):
    return np.sqrt(np.mean(np.power(real - pred, 2))) / np.mean(np.abs(real))

def get_MAPE(pred, real):
    index = np.where(real != 0)
    p = pred[index]
    r = real[index]
    return np.mean(np.abs(p - r) / r)

def get_POIContrastive(pred, real):
    sup_loss = POIContrastiveLoss()
    loss = sup_loss(pred, real)
    return loss.item()

def get_FID(pred, real):
    p = pred.reshape(-1, 24)
    r = real.reshape(-1, 24)
    mu1, sigma1 = np.mean(p, axis=0), np.cov(p, rowvar=False)
    mu2, sigma2 = np.mean(r, axis=0), np.cov(r, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    cov_mean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)
    return fid

def get_dis(pred, real, label):
    n_label = np.max(label) + 1
    res = np.array([])
    for i in range(n_label):
        r = real[np.where(label == i)]
        p = pred[np.where(label == i)]
        r = np.mean(r, axis=0)
        d = p - r
        res = np.concatenate([res, np.sqrt(np.sum(d ** 2, axis=1))])
    return np.mean(res)

def get_cossim(pred, real, label):
    n_label = np.max(label) + 1
    res = np.array([])
    for i in range(n_label):
        r = real[np.where(label == i)]
        p = pred[np.where(label == i)]
        r = np.mean(r, axis=0)
        d = np.sqrt(np.sum(p ** 2, axis=1)) * np.sqrt(np.sum(r ** 2))
        res = np.concatenate([res, (p @ r) / d])
    return np.mean(res)


def get_cross(pred, real, label):
    p_res = np.arange(pred.shape[1]).reshape(1, -1)
    r_res = np.arange(real.shape[1]).reshape(1, -1)
    n_label = np.max(label) + 1
    for i in range(n_label):
        r = real[np.where(label == i)]
        p = pred[np.where(label == i)]
        for j in range(p.shape[0]):
            r_res = np.concatenate([r_res, r])
            p_res = np.concatenate([p_res, np.array(list(p[j].reshape(1, -1)) * p.shape[0])])
    return r_res[1:], p_res[1:]

