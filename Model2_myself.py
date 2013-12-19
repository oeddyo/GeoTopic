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

import time

class Model():
    """Log probability for full covariance matrices.
    """

    def __init__(self, Z = 30, R = 30, n_iteration=50):
        print 'Fetching data...'
        ((self.doc_user, self.user_real_ids, self.user_docs, self.user_day_count\
            , self.user_loc, self.doc_loc, self.created_time, self.day_of_week\
            , self.lat, self.lng, self.texts, self.doc, self.vectorizer),
         (
            self.test_doc_user, self.test_user_real_ids, self.test_user_docs
            , self.test_user_day_count, self.test_user_loc, self.test_doc_loc
            , self.test_created_time, self.test_day_of_week, self.test_lat, \
            self.test_lng, self.test_texts, self.test_doc, self.test_vectorizer
        )) = prepare_data.get_formatted_data(20000)

        print 'doc shape = ', self.doc.shape
        print self.vectorizer.get_feature_names()


        self.n_doc, self.n_word = self.doc.shape
        self.R = R
        self.Z = Z

        self.z_r_d = np.random.random((self.Z, self.R, self.n_doc))
        for d in range(self.n_doc):
            self.z_r_d[:, :, d] /= self.z_r_d[:, :, d].sum()


        # the followings update in M step
        self.n_iteration = n_iteration
        self.w_z_log = np.random.random((self.n_word, Z))
        self.w_z_log = self._normalize_column(self.w_z_log)
        self.w_z_log = np.log(self.w_z_log)


        self.z_r_log = np.random.random((Z, R))
        self.z_r_log = self._normalize_column(self.z_r_log)
        self.z_r_log = np.log(self.z_r_log)

        self.r_alpha = np.random.random((1, R))
        self.r_alpha = self.r_alpha*1.0/sum(self.r_alpha)

        self.back = np.zeros((1, self.n_word))

        for w in range(self.n_word):
            self.back[0, w] = self.doc[:, w].sum()*1.0/(self.doc.sum())

        print self.back
        self.para_lambda = 0.9


        self.mus = np.zeros((R, 2))



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
            prob += self.back[0, w]*self.para_lambda + (1-self.para_lambda)* self.w_z_log[w, z] * self.z_r_log[z, r]
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
                tmp_prob += np.exp(self._get_log_joint(r, d)).sum()

                #for z in range(self.Z):
                #    tmp_prob += np.exp(self._get_log_joint(z, r, d))

            log_likelihood += np.log(tmp_prob)
        return log_likelihood



    def _get_log_joint(self, r, d):
        tmp_prob = np.log(self.r_alpha[0, r])
        tmp_prob += self._get_log_l_r(d, r)

        """
        p1 = 0.0
        for w in self.doc[d, :].nonzero()[1]:
            #prob += self.doc[d, w] * np.log(self.w_z_log[w, z]*self.z_r_log[z, r])
            prob += self.doc[d, w] * (self.w_z_log[w, z]  + self.z_r_log[z, r])
            p1 += self.doc[d, w] * (self.w_z_log[w, z]  + self.z_r_log[z, r])

        """

        #print 'A', (self.w_z_log + self.z_r_log[:, r].T)[:, z].sum(), 'B', (self.w_z_log[:, z] + self.z_r_log[z,r]).sum()
        tmp_matrix = (self.doc[d, :]*(self.w_z_log + self.z_r_log[:, r].T)) + tmp_prob

        #prob = tmp_prob + (self.doc[d, :] * (self.w_z_log[:, z] + self.z_r_log[z,r])).sum()



        #print tmp_matrix[0, z], 'B', (self.doc[d, :] * (self.w_z_log[:, z] + self.z_r_log[z,r])).sum()

        #print tmp_matrix[0,z], prob
        #return prob
        return tmp_matrix

    def _log_norm(self, prob_list):
        original_shape = prob_list.shape
        s_list = prob_list.reshape((1, original_shape[0]*original_shape[1]))

        t_max = np.max(s_list)
        log_deno = t_max + np.log(np.exp(s_list - t_max).sum())

        s_list -= log_deno

        #s_list = np.reshape(original_shape)
        #print np.exp(s_list)
        return np.exp(s_list.reshape((original_shape)))

        #return np.exp(prob_list)

    def e_step(self):
        print 'e_step...'
        start_time = time.time()

        for d in range(self.n_doc):
            for r in range(self.R):
                #for z in range(self.Z):
                #   self.z_r_d[z, r, d] = self._get_log_joint(z, r, d)
                self.z_r_d[:, r, d] = self._get_log_joint(r, d)
            self.z_r_d[:, :, d] = self._log_norm(self.z_r_d[:, :, d])
        end_time = time.time()
        print 'E step uses ', end_time - start_time


    def m_step(self):
        start_time = time.time()
        print 'm_step...'
        for r in range(self.R):
            self.r_alpha[0, r] = self.z_r_d[:, r, :].sum()*1.0/self.n_doc

        for r in range(self.R):
            denominator = self.z_r_d[:, r, :].sum()
            tmp = np.zeros((1, 2))
            for d in range(self.n_doc):
                tmp += self.z_r_d[:, r, d].sum()*self.doc_loc[d]
            print "tmp = ", tmp, 'deno = ', denominator
            self.mus[r, :] = tmp*1.0/denominator

        for r in range(self.R):
            denominator = self.z_r_d[:, r, :].sum()
            tmp = np.zeros((2,2))
            for d in range(self.n_doc):
                dif = (self.doc_loc[d, None] - self.mus[r, None])
                weight = self.z_r_d[:, r, d].sum()
                tmp += weight * np.dot(dif.T, dif)
            self.sgs[r, :, :] = tmp/denominator

            self.sgs[r, :, :] += np.identity(2)*1e-7

        print self.mus
        print self.sgs

        for r in range(self.R):
            for z in range(self.Z):
                #tmp = 0.0
                #for d in range(self.n_doc):
                #    tmp += self.doc[d, :].sum() * self.z_r_d[z,r,d]
                #print self.doc.sum(axis=1).T.shape
                #print self.z_r_d[z, r, :].shape
                tmp = np.dot(self.doc.sum(axis=1).T ,  self.z_r_d[z, r, :]).sum()
                self.z_r_log[z, r] = tmp
            self.z_r_log[:, r] = self.z_r_log[:, r]/self.z_r_log[:, r].sum()
            self.z_r_log[:, r] = np.log(self.z_r_log[:, r])
        print 'sample z_r', self.z_r_log[:, 0]

        for z in range(self.Z):
            for w in range(self.n_word):
                #tmp = 0.0
                #for d in range(self.n_doc):
                #    tmp += self.z_r_d[z, :, d].sum()*self.doc[d,w]
                self.w_z_log[w, z] = (self.z_r_d[z, :, :].sum(axis=0)*self.doc[:, w]).sum()
            self.w_z_log[:, z] = self.w_z_log[:, z]/self.w_z_log[:, z].sum()
            self.w_z_log[:, z] = np.log(self.w_z_log[:, z])

        print 'sample w_z', self.w_z_log[:, 0]
        end_time = time.time()
        print 'M step uses ', end_time - start_time
    def fit(self):
        for iteration in range(self.n_iteration):
            self.e_step()
            self.m_step()
            print "current log likelihood = ", self._get_loglikelihood()
            for i in range(self.Z):
                print 'for topic ', i
                idx = np.argsort(self.w_z_log[:, i])[::-1][:100]

                for id in idx:
                    print self.vectorizer.get_feature_names()[id],
                print '\n'
        pass


model = Model()
model.fit()