import numpy as np

class CSP:
    def __init__(self, m=2, reg=0.0):
        self.m = m
        self.reg = reg
        self.W = None
        self.evals = None

    def _trial_covariance(self, X_trial: np.ndarray) -> np.ndarray:
        Xc = X_trial - X_trial.mean(axis=1, keepdims=True)
        cov = (Xc @ Xc.T) / (Xc.shape[1] - 1)
        return cov

    def _mean_normalized_cov(self, X: np.ndarray) -> np.ndarray:
        n_ch, _, n_tr = X.shape
        C_sum = np.zeros((n_ch, n_ch), dtype=np.float64)
        for k in range(n_tr):
            Ck = self._trial_covariance(X[:, :, k])
            tr = np.trace(Ck)
            if tr <= 0:
                continue
            C_sum += Ck / tr
        C_mean = C_sum / n_tr
        return C_mean

    def fit(self, XA: np.ndarray, XB: np.ndarray):
        if XA.ndim != 3 or XB.ndim != 3:
            raise ValueError("XA and XB must be 3D: (channels, samples, trials)")

        C1 = self._mean_normalized_cov(XA)
        C2 = self._mean_normalized_cov(XB)

        if self.reg > 0:
            n_ch = C1.shape[0]
            C1 = (1 - self.reg) * C1 + self.reg * np.eye(n_ch) * np.trace(C1) / n_ch
            C2 = (1 - self.reg) * C2 + self.reg * np.eye(n_ch) * np.trace(C2) / n_ch

        Cc = C1 + C2

        # Solve the generalized eigenvalue problem C1*W = lambda*(C1+C2)*W
        evals, evecs = np.linalg.eig(np.linalg.pinv(Cc) @ C1)

        # Sort eigenvalues in descending order (like MATLAB)
        idx = np.argsort(np.real(evals))[::-1]
        self.evals = np.real(evals[idx])
        self.W = np.real(evecs[:, idx])  # Columns are spatial filters

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("CSP must be fitted before calling transform.")
        
        n_ch, n_samp, n_tr = X.shape
        Z = np.zeros((n_ch, n_samp, n_tr), dtype=np.float64)
        for k in range(n_tr):
            Z[:, :, k] = self.W.T @ X[:, :, k]
        return Z

    def compute_features(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        Z = self.transform(X)  # (n_ch, n_samp, n_trials)
        n_ch, _, n_tr = Z.shape

        if 2 * self.m > n_ch:
            raise ValueError(f"2m={2 * self.m} should not be greater than the number of channels ({n_ch}).")

        # Select first m and last m components
        comps = list(range(self.m)) + list(range(n_ch - self.m, n_ch))
        feats = np.zeros((n_tr, 2 * self.m), dtype=np.float64)

        for k in range(n_tr):
            Zk = Z[comps, :, k]
            var = np.var(Zk, axis=1, ddof=1)
            var = var / (np.sum(var) + eps)
            feats[k, :] = np.log(var + eps)

        return feats
