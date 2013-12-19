__author__ = 'eddiexie'

import prepare_data
import numpy as np
import scipy
import statsmodels.sandbox.distributions.mv_normal as mvd
import time

class LDA():
    def __init__(self, Z=5, R=20, n_iteration=5000):
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

        self.beta1 = 0.1
        self.beta0 = 0.01
        self.beta2 = 0.1
        self.beta3 = 0.01

        self.gamma = 20.0
        self.R = R
        self.Z = Z
        self.alpha = (self.Z/50.0)/10

        self.n_doc, self.n_word = self.doc.shape

        self.r_doc = np.zeros((1, self.n_doc))
        self.z_doc = np.zeros((1, self.n_doc))
        self.y_word = {}

        self.n_iteration = n_iteration


        self.cnt_y_word_sum = np.zeros((1, 3))
        self.cnt_y_word = np.zeros((3, self.n_word))

        self.cnt_r_word = np.zeros((self.R, self.n_word))
        self.cnt_r_word_sum = np.zeros((1, self.R))

        self.cnt_z_word = np.zeros((self.Z, self.n_word))
        self.cnt_z_word_sum = np.zeros((1, self.Z))

        self.cnt_r_doc = np.zeros((self.R, self.n_doc))
        self.cnt_r_doc_sum = np.zeros((1, self.n_doc))

        self.cnt_r_topic = np.zeros((self.R, self.Z))


        self.mu0 = self.doc_loc.mean(axis=0).reshape(1,2)
        dot = np.dot((self.doc_loc - self.mu0).T, (self.doc_loc - self.mu0))
        self.sg = np.diag(np.diag(dot))*1.0/self.n_doc  # Murphy book page 133
        self.k0 = 0.01
        self.v0 = 2+2    # D+2

        self.SS = np.zeros((self.R, 2, 2))  # murphy book 134. store sigma_{1}{N}{x_i*x_i.T}. un-center version
        self.mus = np.zeros((self.R, 2))

        self.words_in_tweets = []

        for d in range(self.n_doc):
            self.words_in_tweets.append( self.doc[d, :].nonzero()[1] )


    def student_logpdf(self, x):
        pass

    def init(self):
        # initialize parameters here
        for d in range(self.n_doc):
            r = int(np.random.multinomial(1, np.asarray([1.0/self.R]*self.R)).argmax())
            self.r_doc[0, d] = r
            self.cnt_r_doc[r, d] += 1

            z = int(np.random.multinomial(1, np.asarray([1.0/self.Z]*self.Z)).argmax())
            self.z_doc[0, d] = z
            self.cnt_r_topic[r, z] += 1

            for w in self.doc[d, :].nonzero()[1]:
                y = int(np.random.multinomial(1, [1.0/3]*3).argmax())
                self.y_word[(d, w)] = y
                if y == 1:
                    self.cnt_z_word[z, w] += 1
                elif y == 2:
                    self.cnt_r_word[r, w] += 1
                self.cnt_y_word[y, w] += 1

            self.cnt_r_doc_sum[0, r] = self.cnt_r_doc[r, :].sum()
            self.cnt_z_word_sum[0, z] = self.cnt_z_word[z, :].sum()
            self.cnt_r_word_sum[0, r] = self.cnt_r_word[r, :].sum()

            for tmp_y in range(3):
                self.cnt_y_word_sum[0, tmp_y] = self.cnt_y_word[tmp_y, :].sum()

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

    def sample_r(self, d):
        cur_r = self.r_doc[0, d]

        N = self.cnt_r_doc[cur_r, :].sum()
        self.cnt_r_doc[cur_r, d] -= 1
        self.cnt_r_doc_sum[0, cur_r] -= 1

        words_in_tweet = self.words_in_tweets[d]

        for w in words_in_tweet:
            cur_y = self.y_word[d, w]
            self.cnt_y_word[cur_y, w] -= 1
            self.cnt_y_word_sum[0, cur_y] -= 1
            if cur_y == 2:
                self.cnt_r_word[cur_r, w] -= 1
                self.cnt_r_word_sum[0, cur_r] -= 1

        #take out mu
        self.mus[cur_r] = (self.mus[cur_r]*N - self.doc_loc[d, :])*1.0/(N-1)

        #take out cov
        loc = self.doc_loc[d, :].reshape(1, 2)
        self.SS[cur_r, :, :] -= np.dot(loc.T, loc)

        cur_z = self.z_doc[0, d]
        self.cnt_r_topic[cur_r, cur_z] -= 1

        sample_prob = [0.0]*self.R
        for r in range(self.R):
            #p_r = (self.cnt_r_doc[r, d] + self.beta1)*1.0/(self.cnt_r_doc[:, d].sum() + self.R*self.beta1)
            p_r = (self.cnt_r_doc[r, d] + self.beta1)*1.0/(self.cnt_r_doc_sum[0, d] + self.R*self.beta1)

            #current_r_doc = self.cnt_r_doc[r, :].sum()
            current_r_doc = self.cnt_r_doc_sum[0, r]

            _mu = (self.k0/(self.k0 + current_r_doc))*self.mu0 + (current_r_doc*1.0/(self.k0 + current_r_doc))*self.mus[r]

            vn = self.v0 + current_r_doc
            kn = self.k0 + current_r_doc
            _sn = self.sg + self.SS[r, :, :] + self.k0*np.dot(self.mu0.T, self.mu0) - kn*np.dot(_mu.T, _mu )
            _sn *= ((kn+1)*1.0/(kn*(vn-2+1)))

            p_l_r = mvd.MVT(_mu[0], _sn, vn-2+1).pdf(self.doc_loc[d, :])
            print type(r), type(cur_z)
            p_z_r = (self.cnt_r_topic[r, cur_z] + self.alpha)*1.0/(self.cnt_r_topic[r, :].sum() + self.Z*self.alpha)

            p_w_zry = 1.0
            wcount = 0
            for w in words_in_tweet:
                for j in range(self.doc[d, w]):
                    p_w_zry *= (self.cnt_r_word[r, w] + self.beta3 + j)*1.0/(
                        self.cnt_r_word_sum[0, r] + self.n_word*self.beta3 + wcount)
                    wcount += 1

            sample_prob[r] = p_r * p_l_r * p_z_r * p_w_zry

        sample_prob = np.asarray(sample_prob)
        sample_prob /= sample_prob.sum()
        #print sample_prob
        new_r = int(np.random.multinomial(1, sample_prob).argmax())

        self.r_doc[0, d] = new_r

        #add loc stats back
        N = self.cnt_r_doc[new_r, :].sum()
        self.mus[new_r] = (N*self.mus[new_r] + self.doc_loc[d, :])*1.0/(N+1)

        # add SS stats back
        loc = self.doc_loc[d, :].reshape(1,2)
        self.SS[new_r, :, :] += np.dot(loc.T, loc)

        #add count back
        self.cnt_r_doc[new_r, d] += 1
        self.cnt_r_doc_sum[0, new_r] += 1
        self.cnt_r_topic[new_r, cur_z] += 1

        for w in words_in_tweet:
            cur_y = self.y_word[d, w]
            self.cnt_y_word[cur_y, w] += 1
            self.cnt_y_word_sum[0, cur_y] += 1
            if cur_y == 2:
                self.cnt_r_word[new_r, w] += 1
                self.cnt_r_word_sum[0, new_r] += 1


    def sample_topic(self, d):
        cur_r = self.r_doc[0, d]
        cur_z = self.z_doc[0, d]

        self.cnt_r_topic[cur_r, cur_z] -= 1

        words_in_tweet = self.words_in_tweets[d]
        for w in words_in_tweet:
            cur_y = self.y_word[(d, w)]
            self.cnt_y_word[cur_y, w] -= 1
            self.cnt_y_word_sum[0, cur_y] -= 1
            if cur_y == 1:
                self.cnt_z_word[cur_z, w] -= 1
                self.cnt_z_word_sum[0, cur_z] -=1
            #elif cur_y == 2:
            #    self.cnt_r_word[cur_r, w] -= 1
            #    self.cnt_r_word_sum[0, cur_r] -= 1

        sample_prob = [0.0]*self.Z
        for z in range(self.Z):
            p_z_r = (self.cnt_r_topic[cur_r, z] + self.alpha)*1.0/(self.cnt_r_topic[cur_r, :].sum() + self.Z*self.alpha)
            p_w_z = 1.0
            wcount = 0
            for w in words_in_tweet:
                for j in range(self.doc[d, w]):
                    p_w_z *= (self.cnt_z_word[z, w] + self.beta0 + j)*1.0/(
                        self.cnt_z_word_sum[0, z] + self.n_word*self.beta0 + wcount)
                    wcount += 1

            sample_prob[z] = p_z_r * p_w_z
        sample_prob = np.asarray(sample_prob)
        sample_prob /= sample_prob.sum()
        new_z = int(np.random.multinomial(1, sample_prob).argmax())
        print 'new_z type = ', type(new_z)
        self.cnt_r_topic[cur_r, new_z] += 1
        self.z_doc[0, d] = new_z
        for w in words_in_tweet:
            cur_y = self.y_word[(d, w)]
            self.cnt_y_word[cur_y, w] += 1
            self.cnt_y_word_sum[0, cur_y] += 1
            if cur_y == 1:
                self.cnt_z_word[new_z, w] += 1
                self.cnt_z_word_sum[0, new_z] +=1
            #elif cur_y == 2:
            #    self.cnt_r_word[cur_r, w] += 1
            #    self.cnt_r_word_sum[0, cur_r] += 1


    def sample_y(self, d):
        # sample all the words in d
        words_in_tweets = self.words_in_tweets[d]

        for w in words_in_tweets:
            cur_y = self.y_word[(d, w)]
            self.cnt_y_word[cur_y, w] -= 1
            self.cnt_y_word_sum[0, cur_y] -= 1

            p_base = [0.0]*3
            for y_type in range(3):
                p_base[y_type] = (self.cnt_y_word_sum[0, y_type] + self.gamma)/(self.cnt_y_word_sum[0, :].sum() + 3*self.gamma)

            p0 = p_base[0] * (self.cnt_y_word[0, w] + self.beta2)/(self.cnt_y_word_sum[0, 0] + self.n_word*self.beta2)
            p1 = p_base[1] * (self.cnt_y_word[1, w] + self.beta0)/(self.cnt_y_word_sum[0, 1] + self.beta0)
            p2 = p_base[2] * (self.cnt_y_word[2, w] + self.beta3)/(self.cnt_y_word_sum[0, 2] + self.beta3)

            p_sum = p0 + p1 + p2
            new_y = np.random.multinomial(1, [p0*1.0/p_sum, p1*1.0/p_sum, p2*1.0/p_sum])

            self.cnt_y_word[new_y, w] += 1
            self.cnt_y_word_sum[0, new_y] += 1


    def fit(self):
        self.init()
        print 'initial sg = ', self.sg

        for iteration in range(self.n_iteration):
            print 'iteration... ', iteration

            start_time = time.time()
            for d in range(self.n_doc):
                if d % 100 == 0:
                    print 'working on doc ', d
                self.sample_r(d)
                self.sample_topic(d)
                self.sample_r(d)
            end_time = time.time()

            print 'Iteration used ', end_time - start_time

            print 'global topics:'
            show_topics = []
            for phi in range(self.Z):
                _tmp = (self.cnt_z_word[phi, :]+self.beta0)*1.0/(self.cnt_z_word[phi, :].sum() + self.beta0*self.n_word)
                print phi,
                current_topic = []
                for idx in _tmp.argsort()[::-1][:10]:
                    print (self.vectorizer.get_feature_names()[idx], _tmp[idx]),
                    current_topic.append((self.vectorizer.get_feature_names()[idx], _tmp[idx]))
                show_topics.append(current_topic)
                print '\n'


            _tmp = self.cnt_y_word[0, :]
            print 'stopwords:'
            for idx in _tmp.argsort()[::-1][:20]:
                print self.vectorizer.get_feature_names()[idx],
            print '\n'
            print 'local topics:'
            for region in range(self.R):
                N = self.cnt_r_doc[region, :].sum()
                kn = self.k0 + N
                vn = self.v0 + N
                cur_mu = self.mus[region].reshape(1, 2)*N/(self.k0+N) + self.k0*1.0*self.mu0/(self.k0+N)
                sn = self.SS[region, :, :] + self.sg + self.k0*np.dot(self.mu0.T, self.mu0) - kn*np.dot(cur_mu.T, cur_mu)
                print cur_mu
                print sn*1.0/(vn+2+2)

                region_topic = np.asarray(self.cnt_r_word[region, :])
                tmp = (region_topic + self.beta3) / (region_topic.sum() + self.beta3*self.n_word)

                for idx in tmp.argsort()[::-1][:10]:
                    print self.vectorizer.get_feature_names()[idx],
                print '\n'


model = LDA()
model.fit()

