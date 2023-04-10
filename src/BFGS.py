#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def BFGS(X, y, epoch):

          Xo = np.ones((X.shape[0], 1))
          X = np.hstack((Xo, X))

          m = len(X)
          theta_1 = np.zeros((X.shape[1] , 1))
          theta = np.ones((X.shape[1], 1)) * 0.01
          B = np.eye(X.shape[1])

          H = []
          all_d_theta=[]
          all_theta=[]
          Loss = []
          callback = 0

          B_inv = B.T


          h = X @ theta_1
          error = h - y

          J = 1 / (2 * m) * error.T @ error
          d_theta = (X.T @ error ) / len(X)


          all_d_theta.append(d_theta)  
          all_theta.append(theta_1)

          Loss.append(J[0][0])


          for i in range(epoch):

            all_theta.append(theta)
            delta_x = all_theta[-1] - all_theta[-2]
            h = X @ theta
            H.append(h)

            error = (h - y)

            J = 1 / (2 * m) * error.T @ error

            Loss.append(J[0][0])

            d_theta = 1 / m * (X.T @ error)
            all_d_theta.append(d_theta)

            delta_y = all_d_theta[-1] - all_d_theta[-2]



            GradVec_Norm = np.linalg.norm(d_theta)


            fraction1 = (np.eye(X.shape[1]) - ((delta_x @ delta_y.T) / (delta_y.T @ delta_x)))
            fraction2 = (np.eye(X.shape[1]) - ((delta_y @ delta_x.T) / (delta_y.T@delta_x)))
            fraction3 = ((delta_x@delta_x.T) / (delta_y.T@delta_x))


            B_inv = fraction1 @ B_inv @ fraction2 + fraction3
            theta = theta - B_inv @ d_theta



            if (i > 0) and (abs(Loss[i - 1] - J) < 0.001):
              callback += 1

            if J <= 0.001 or GradVec_Norm <= 0.001 or callback == 2:
              break

          Loss = np.array(Loss).reshape(-1)
          H = np.concatenate(H).ravel()
          H = H.reshape(-1, X.shape[0])
          all_theta = np.concatenate(all_theta).ravel()
          all_theta = all_theta.reshape(-1, X.shape[1])
          return H, all_theta, Loss, i

