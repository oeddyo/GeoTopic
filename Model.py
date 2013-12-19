__author__ = 'eddiexie'
__author__ = 'eddiexie'

import prepare_data
import logging
import sys
import numpy as np
import scipy.stats
import scipy.sparse
import time
import random

from sklearn.preprocessing import normalize


class Model():
    """Log probability for full covariance matrices.
    """

    def __init__(self, Z = 10, R = 20, n_iteration=50):
        print 'Fetching data...'
        ((self.doc_user, self.user_real_ids, self.user_docs, self.user_day_count\
            , self.user_loc, self.doc_loc, self.created_time, self.day_of_week\
            , self.lat, self.lng, self.texts, self.doc, self.vectorizer),
         (
            self.test_doc_user, self.test_user_real_ids, self.test_user_docs
            , self.test_user_day_count, self.test_user_loc, self.test_doc_loc
            , self.test_created_time, self.test_day_of_week, self.test_lat, \
            self.test_lng, self.test_texts, self.test_doc, self.test_vectorizer
        )) = prepare_data.get_formatted_data(500)

        print 'doc shape = ', self.doc.shape
        print self.vectorizer.get_feature_names()


        self.n_doc, self.n_word = self.doc.shape
        self.R = R
        self.Z = Z
        self.r_d = np.random.random((R, self.n_doc)) # update in E step

        # the followings update in M step
        self.n_iteration = n_iteration
        self.w_z = np.random.random((self.n_word, Z))
        self.w_z = self._normalize_column(self.w_z)

        self.z_r = np.random.random((Z, R))
        self.z_r = self._normalize_column(self.z_r)

        self.r_alpha = np.random.random((1, R))
        self.r_alpha = self.r_alpha*1.0/sum(self.r_alpha)

        self.back = np.zeros((1, self.n_word))

        for w in range(self.n_word):
            self.back[0, w] = self.doc[:, w].sum()*1.0/(self.doc.sum())

        print self.back
        self.para_lambda = 0.9


        self.mus = np.zeros((R, 2))

        self.zwr = np.random.random((self.Z, self.n_word, self.R))
        print self.zwr.shape
        for w in range(self.n_word):
            for r in range(self.R):
                self.zwr[:, w, r] /= self.zwr[:, w, r].sum()

        for i, mu in enumerate(self.mus):
            self.mus[i, 0] = random.random() + 40
            self.mus[i, 1] = random.random() + -73
        self.sgs = np.zeros((R, 2, 2)) + np.identity(2)*1

        for n, loc in enumerate(self.doc_loc):
            print "['"+str(i)+"', "+str(loc[0]) + ", "+str(loc[1])+ ", " + "3],"


    def _normalize_column(self, m):
        column_sum = m.sum(axis=0)
        #new_matrix = m / column_sum[np.newaxis, :]
        for idx, col in enumerate(column_sum):
            m[:, idx] /= column_sum[idx]
        return m
        #return new_matrix

    def _get_w_r(self, w, r):
        prob = 0.0
        for z in range(self.Z):
            prob += self.back[0, w]*self.para_lambda + (1-self.para_lambda)* self.w_z[w, z] * self.z_r[z, r]
        return prob

    def _log_multivariate_normal_density_full(self, X, mu, cv, min_covar=1e-7):
        from scipy import linalg
        if hasattr(linalg, 'solve_triangular'):
            # only in scipy since 0.9
            solve_triangular = linalg.solve_triangular
        else:
            # slower, but works
            solve_triangular = linalg.solve
        n_dim = 2
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob = - .5 * (np.sum(cv_sol ** 2) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

        return log_prob

    def _get_log_l_r(self, d, r):
        probs = self._log_multivariate_normal_density_full(self.doc_loc[d],
                                                      self.mus[r, :], self.sgs[r, :, :], 1e-5)
        return probs

    def _get_loglikelihood(self):
        log_likelihood = 0.0
        for d in range(self.n_doc):
            tmp_prob = 0.0
            for r in range(self.R):
                #log_likelihood += self._get_log_joint(r, d)
                tmp_prob += np.exp(self._get_log_joint(r, d))
            if tmp_prob<=0.0:
                return -float("Inf")
            log_likelihood += np.log(tmp_prob)
        return log_likelihood

    def _get_log_joint(self, r, d):
        prob = np.log(self.r_alpha[0, r])
        for w in self.doc[d, :].nonzero()[1]:
            prob += self.doc[d, w] * np.log(self._get_w_r(w, r))
        prob += self._get_log_l_r(d, r)
        return prob


    def _log_norm(self, prob_list):
        t_max = np.max(prob_list)
        #log_deno = t_max + np.log(sum(np.exp(prob_list - t_max)))
        log_deno = t_max + np.log(np.exp(prob_list - t_max).sum())

        print "log_deno = ", log_deno
        prob_list -= log_deno
        return np.exp(prob_list)

    def e_step(self):
        print 'e_step...'
        for d in range(self.n_doc):
            for r in range(self.R):
                tmp_joint = self._get_log_joint(r, d)
                self.r_d[r, d] = tmp_joint
            self.r_d[:, d] = self._log_norm(self.r_d[:, d])

        print 'sample r_d', self.r_d[:, 0]
        for w in range(self.n_word):
            for r in range(self.R):
                deno = np.dot(self.w_z[w, :], self.z_r[:, r]).sum()
                for z in range(self.Z):
                    deno += self.w_z[w,z]*self.z_r[z, r]

                for z in range(self.Z):
                    self.zwr[z, w, r] = (1-self.para_lambda)*self.w_z[w,z]*self.z_r[z,r] \
                                        /(self.para_lambda*self.back[0, w] + (1-self.para_lambda)*deno)


                #self.zwr[:, w, r] /= self.zwr[:, w, r].sum()
        print 'sample zwr = ', self.zwr[:, 0, 0]

    def m_step(self):
        print 'm_step...'
        #alpha = 3
        for r in range(self.R):
            self.r_alpha[0, r] = (self.r_d[r, :].sum()*1.0 )/(self.n_doc )


        for r in range(self.R):
            denominator = self.r_d[r, :].sum()
            tmp = np.zeros((1, 2))
            for d in range(self.n_doc):
                tmp += self.r_d[r, d]*self.doc_loc[d]
            self.mus[r, :] = tmp*1.0/denominator

        for r in range(self.R):
            denominator = self.r_d[r, :].sum()
            tmp = np.zeros((2,2))
            for d in range(self.n_doc):
                dif = (self.doc_loc[d, None] - self.mus[r, None])

                tmp += self.r_d[r, d] * np.dot(dif.T, dif)
                #print 'cov shape = ', np.dot(dif.T, dif)
            self.sgs[r, :, :] = tmp/denominator

            self.sgs[r, :, :] += np.identity(2)*1e-5

        print self.mus
        print self.sgs

        for r in range(self.R):
            for z in range(self.Z):
                tmp = 0.0
                for d in range(self.n_doc):
                    tmp += np.dot(np.asarray(self.doc[d, :].todense()) ,  self.zwr[z, :, r]).sum()*self.r_d[r, d]
                self.z_r[z, r] = tmp
            self.z_r[:, r] = self.z_r[:, r]/self.z_r[:, r].sum()
        print 'sample z_r', self.z_r[:, 0]


        for z in range(self.Z):
            for w in range(self.n_word):
                tmp = 0.0
                for r in range(self.R):
                    #for d in range(self.n_doc):
                    #    tmp += self.doc[d, w] * self.r_d[r,d] * self.zwr[z,w,r]
                    #print self.doc[:, w].todense().shape, self.r_d[r, :].shape
                    #print 'notice!!! ', np.dot(np.asarray(self.doc[:, w].todense()), self.r_d[r, :])
                    tmp += np.dot(np.asarray(self.doc[:, w].todense().T), self.r_d[r, :].T).sum() * self.zwr[z,w,r]

                self.w_z[w, z] = tmp
            self.w_z[:, z] = self.w_z[:, z]/self.w_z[:, z].sum()
        print 'sample w_z', self.w_z[:, 0]


    def fit(self):
        for iteration in range(self.n_iteration):
            self.e_step()
            self.m_step()
            print "current log likelihood = ", self._get_loglikelihood()
            for i in range(self.Z):
                print 'for topic ', i
                idx = np.argsort(self.w_z[:, i])[::-1][:100]

                for id in idx:
                    print self.vectorizer.get_feature_names()[id],
                print '\n'
        pass


model = Model()
model.fit()