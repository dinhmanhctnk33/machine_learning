from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(18)

# =========================
# 1. Tao du lieu
# =========================

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)

K = 3

original_label = np.asarray(
    [0] * N + [1] * N + [2] * N
).T


# =========================
# 2. Hien thi ket qua
# =========================

def kmeans_display(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.show()


# =========================
# 3. Khoi tao tam cum
# =========================

def kmeans_init_centroids(X, k):
    # Chon ngau nhien k hang cua ma tran X
    # lam cac tam cum ban dau
    return X[np.random.choice(X.shape[0], k, replace=False)]


# =========================
# 4. Gan nhan cho cac diem
# =========================

def kmeans_assign_labels(X, centroids):
    # Tinh khoang cach giua tung diem va cac tam cum
    D = cdist(X, centroids)

    # Tra ve vi tri cua tam cum gan nhat
    return np.argmin(D, axis=1)


# =========================
# 5. Kiem tra dieu kien dung
# =========================

def has_covered(centroids, new_centroids):
    # Tra ve True neu hai tap tam cum giong nhau
    return set(
        [tuple(a) for a in centroids]
    ) == set(
        [tuple(a) for a in new_centroids]
    )


# =========================
# 6. Cap nhat tam cum
# =========================

def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))

    for k in range(K):
        # Lay cac diem thuoc cum k
        Xk = X[labels == k, :]

        # Tinh trung binh de tim tam cum moi
        centroids[k, :] = np.mean(Xk, axis=0)

    return centroids


# =========================
# 7. Thuat toan K-means
# =========================

def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0

    while True:
        # Gan nhan cho tung diem
        labels.append(
            kmeans_assign_labels(X, centroids[-1])
        )

        # Cap nhat tam cum
        new_centroids = kmeans_update_centroids(
            X,
            labels[-1],
            K
        )

        # Kiem tra dieu kien dung
        if has_covered(centroids[-1], new_centroids):
            break

        centroids.append(new_centroids)
        it += 1

    return centroids, labels, it


# =========================
# 8. Chay K-means
# =========================

centroids, labels, it = kmeans(X, K)

print('Center found by our algorithm:')
print(centroids[-1])

print('\nNumber of iterations:', it)

# Hien thi ket qua
kmeans_display(X, labels[-1])