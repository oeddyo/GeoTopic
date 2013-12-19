__author__ = 'eddiexie'

import prepare_data
import numpy as np
import scipy
import statsmodels.sandbox.distributions.mv_normal as mvd

class LDA():
    def __init__(self, Z=10, R=10, n_iteration=5000):
        print 'Fetching data...'
        ((self.doc_user, self.user_real_ids, self.user_docs, self.user_day_count \
              , self.user_loc, self.doc_loc, self.created_time, self.day_of_week \
              , self.lat, self.lng, self.texts, self.doc, self.vectorizer),
         (
             self.test_doc_user, self.test_user_real_ids, self.test_user_docs
             , self.test_user_day_count, self.test_user_loc, self.test_doc_loc
             , self.test_created_time, self.test_day_of_week, self.test_lat, \
             self.test_lng, self.test_texts, self.test_doc, self.test_vectorizer
         )) = prepare_data.get_formatted_data(500)

        print 'doc shape = ', self.doc.shape
        self.beta1 = 0.01
        self.beta0 = 0.01
        self.R = R
        self.n_doc, self.n_word = self.doc.shape
        self.r_doc = np.zeros((1, self.n_doc))
        self.n_iteration = n_iteration

        self.cnt_r_doc = np.zeros((self.R, self.n_doc))
        self.cnt_r_word = np.zeros((self.R, self.n_word))


        self.mu0 = self.doc_loc.mean(axis=0).reshape(1,2)
        dot = np.dot((self.doc_loc - self.mu0).T, (self.doc_loc - self.mu0))
        self.sg = np.diag(np.diag(dot))*1.0/self.n_doc  # Murphy book page 133
        self.k0 = 0.01
        self.v0 = 2+2    # D+2


        self.SS = np.zeros((self.R, 2, 2))  # murphy book 134. store sigma_{1}{N}{x_i*x_i.T}. un-center version
        self.mus = np.zeros((self.R, 2))

    def student_logpdf(self, x):
        pass

    def init(self):
        # initialize parameters here
        for d in range(self.n_doc):
            r = np.random.multinomial(1, np.asarray([1.0/self.R]*self.R)).argmax()
            self.r_doc[0, d] = r
            self.cnt_r_doc[r, d] += 1
            for w in self.doc[d, :].nonzero()[1]:
                self.cnt_r_word[r, w] += 1

        # compute SS and mus
        print 'computing mean'
        for r in range(self.R):
            self.mus[r] = self.doc_loc[self.r_doc[0, :]==r, :].mean(axis=0)
        for r in range(self.R):
            #xs = (self.doc_loc[self.r_doc[0, :]==r, :] - self.mus[r])
            xs = self.doc_loc[self.r_doc[0, :]==r, :]
            self.SS[r, :, :] += np.dot(xs.T, xs)

            if r == 5:
                print 'ss = !!!', self.SS[r, :, :]

    def get_p_l_r(self, r, d):

        N = self.cnt_r_doc[r, :].sum()
        #mu = self.k0 +

        return 1.0
        """
        p = np.log(1.0/np.pi)
        N = self.cnt_r_doc[r, :].sum()

        _dif = (self.doc_loc[d, :] - self.mus[r]).reshape(1, 2)

        lambda_n = self.sg + self.SS[r, :, :] + (self.k0*(N+1)/(self.k0+N+1))*(
            np.dot(_dif.T, _dif)
        )

        vn = self.v0 + N + 1
        p += scipy.special.multigammaln(vn, 2) - scipy.special.multigammaln(vn - 1, 2)

        _dif = (self.doc_loc[d, :] - self.mus[r]).reshape(1, 2)
        substract_sg = np.dot( _dif.T, _dif)
        denominator_mu = (self.mus[r] * N + 1 - self.doc_loc[d, :])*1.0/(N + 1 - 1)
        _dif = (denominator_mu - self.mu0).reshape(1, 2)
        lambda_n_expect_i = self.sg + (self.SS[r, :, :] - substract_sg) + (self.k0*N/(self.k0 +N))*(
            np.dot(_dif.T, _dif)
        )

        p += (np.linalg.det(lambda_n_expect_i) - np.linalg.det(lambda_n))

        p += self.k0 + N - (self.k0 + N+1)

        print 'pp = ', np.exp(p)

        return np.exp(p)
        """

        """

        N = self.cnt_r_doc[r, :].sum()

        numerator_N = N + 1
        denominator_N = N
        _dif = (self.mus[r] - self.mu0).reshape(1, 2)
        numerator_sg = self.sg + self.SS[r, :, :] + (self.k0*numerator_N/(self.k0 + numerator_N))*(
            np.dot(_dif.T, _dif))

        _dif = (self.doc_loc[d, :] - self.mus[r]).reshape(1, 2)
        substract_sg = np.dot( _dif.T, _dif)

        denominator_mu = (self.mus[r] * numerator_N - self.doc_loc[d, :])*1.0/(numerator_N - 1)

        _dif = (denominator_mu - self.mu0).reshape(1, 2)
        denominator_sg = self.sg + (self.SS[r, :, :] - substract_sg) + (self.k0*denominator_N/(self.k0 +denominator_N))*(
            np.dot(_dif.T, _dif)
        )

        kn = self.k0 + numerator_N
        vn = self.v0 + numerator_N

        numerator = mvd.MVT(self.mus[r], numerator_sg*kn*1.0/(kn*(vn - 2 + 1)), vn-2+1).pdf(self.doc_loc[d, :])

        denominator = mvd.MVT(denominator_mu, denominator_sg*(kn-1)*1.0/(kn-1)*(vn-1-2+1), vn-1-2+1).pdf(self.doc_loc[d, :])


        #print 'ppb ', numerator, denominator, numerator*1.0/denominator

        return numerator*1.0/denominator
        #return numerator*1.0/denominator

        # mu except document d

        """





    def fit(self):
        self.init()
        print 'initial sg = ', self.sg

        for iteration in range(self.n_iteration):
            print 'iteration... ', iteration

            for d in range(self.n_doc):
                cur_r = self.r_doc[0, d]

                N = self.cnt_r_doc[cur_r, :].sum()
                self.cnt_r_doc[cur_r, d] -= 1

                #take out mu
                tmp_mu = self.mus[cur_r]
                self.mus[cur_r] = (self.mus[cur_r]*N - self.doc_loc[d, :])*1.0/(N-1)

                #take out cov
                loc = self.doc_loc[d, :].reshape(1, 2)
                self.SS[cur_r, :, :] -= np.dot(loc.T, loc)

                words_in_tweets = self.doc[d, :].nonzero()[1]
                for w in words_in_tweets:
                    self.cnt_r_word[cur_r, w] -= 1

                sample_prob = [0.0]*self.R
                for r in range(self.R):
                    p_r = (self.cnt_r_doc[r, d] + self.beta1)*1.0/(self.cnt_r_doc[:, d].sum() + self.R*self.beta1)
                    p_w_r = 1.0
                    wcount = 0
                    for w in words_in_tweets:
                        for j in range(self.doc[d, w]):
                            p_w_r *= (self.cnt_r_word[r, w] + self.beta0 + j)*1.0/(
                                self.cnt_r_word[r, :].sum() + self.n_word*self.beta0 + wcount)
                            wcount += 1

                    #p_l_r = self.get_p_l_r(r, d)
                    current_r_doc = self.cnt_r_doc[r, :].sum()
                    _mu = (self.k0/(self.k0 + current_r_doc))*self.mu0 + (current_r_doc*1.0/(self.k0 + current_r_doc))*self.mus[r]

                    assert (_mu.shape == (1,2))
                    vn = self.v0 + current_r_doc
                    kn = self.k0 + current_r_doc
                    _sn = self.sg + self.SS[r, :, :] + self.k0*np.dot(self.mu0.T, self.mu0) - kn*np.dot(_mu.T, _mu )
                    _sn *= ((kn+1)*1.0/(kn*(vn-2+1)))

                    #print 'vn, kn = ', vn, kn
                    #print 'current mu and sg, r, mus[r], sgs[r], N =  ', _mu[0], _sn, r, self.mus[r], self.SS[r, :, :], current_r_doc
                    p_l_r = mvd.MVT(_mu[0], _sn, vn-2+1).pdf(self.doc_loc[d, :])

                    #print 'now plr = ', p_l_r
                    #p_l_r = 1.0
                    sample_prob[r] = p_r * p_w_r * p_l_r

                sample_prob = np.asarray(sample_prob)
                sample_prob /= sample_prob.sum()
                #print sample_prob
                new_r = np.random.multinomial(1, sample_prob).argmax()

                #add words stats back
                for w in words_in_tweets:
                    self.cnt_r_word[new_r, w] += 1
                self.r_doc[0, d] = new_r

                #add loc stats back
                N = self.cnt_r_doc[new_r, :].sum()
                self.mus[new_r] = (N*self.mus[new_r] + self.doc_loc[d, :])*1.0/(N+1)

                # add SS stats back
                loc = self.doc_loc[d, :].reshape(1,2)
                self.SS[new_r, :, :] += np.dot(loc.T, loc)

                #add count back
                self.cnt_r_doc[new_r, d] += 1



            for phi in range(self.R):
                _tmp = (self.cnt_r_word[phi, :]+self.beta0)*1.0/(self.cnt_r_word[phi, :].sum())
                #print _tmp
                for idx in _tmp.argsort()[::-1][:30]:
                    print self.vectorizer.get_feature_names()[idx],

                N = self.cnt_r_doc[phi, :].sum()
                kn = self.k0 + N
                vn = self.v0 + N

                cur_mu = self.mus[phi].reshape(1, 2)*N/(self.k0+N) + self.k0*1.0*self.mu0/(self.k0+N)
                sn = self.SS[phi, :, :] + self.sg + self.k0*np.dot(self.mu0.T, self.mu0) - kn*np.dot(cur_mu.T, cur_mu)
                print cur_mu, sn*1.0/(vn+2+2)



model = LDA()
model.fit()

