
def Adam(X, y, epochs = 500, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, batch_size = 64):
          from math import ceil
          from math import sqrt
          import numpy as np 
          Xo = np.ones((X.shape[0], 1))
          X = np.hstack((Xo, X))
          theta = np.zeros((X.shape[1] , 1))
          Vt = 0
          Svt = 0

          Epsilon = 1e-8
          all_theta = []

          y_pred = []
          Loss = []

          callback = 0

          m = X.shape[0]

          for e in range(epochs):

            for i in range(0, m ,batch_size):
              if(i+ batch_size < m):
                X_mini = X[i:i+batch_size]
                y_mini = y[i:i+batch_size]
              else:
                X_mini = X[i:]
                y_mini = y[i:]    


              h = X_mini @ theta

              y_pred.append(h)

              error =  (h - y_mini)

              J = 1/(2 * batch_size) * error.T @ error

              d_theta = 1/batch_size * X_mini.T @ error

              Vt = beta1 * Vt + (1 - beta1) * d_theta

              Svt = beta2 * Svt + (1 - beta2) * (d_theta.T @ d_theta)

              Vcorr = Vt / (1 - beta1**(i+1))

              Scorr = Svt / (1 - beta2**(i+1))


              GradVec_Norm = np.linalg.norm(d_theta)

              theta = theta - alpha / (sqrt(Scorr) + Epsilon) * Vcorr

              all_theta.append(theta)


              Loss.append(J)

            if (e > 0) and (abs(Loss[(e - 1)] - Loss[e]) < 0.001):
                callback += 1

            if (min(Loss) <= 0.35) or (callback == 2):
              break

          Loss = np.array(Loss).reshape(-1)
          y_pred = np.concatenate(y_pred).ravel()
          y_pred = y_pred.reshape(-1, X.shape[0])
          h = y_pred[-1].reshape(-1,1)
          all_theta = np.concatenate(all_theta).ravel()
          all_theta = all_theta.reshape(-1, X.shape[1])

          iterations = ceil(m / batch_size) * (e+1)
          return all_theta, y_pred, Loss, J, h, e, theta

