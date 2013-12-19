__author__ = 'eddiexie'

import prepare_data
import numpy as np

class LDA():
    def __init__(self, Z=2, R=30, n_iteration=5000):
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
        print self.vectorizer.get_feature_names()

        self.n_doc, self.n_word = self.doc.shape
        self.n_iteration = n_iteration
        self.Z = Z  # number of topics

        self.cnt_topic_word = np.zeros((Z, self.n_word, 2))
        self.cnt_doc_topic = np.zeros((self.n_doc, self.Z, 2))
        #self.cnt_doc = np.zeros((1, self.n_doc))
        self.cnt_topic = np.zeros((1, self.Z))


        self.cnt_doc_x = np.zeros((self.n_doc, 2))
        self.cnt_word_x = np.zeros((self.n_word, 2))

        self.x_word_in_doc = {}
        self.z_word_in_doc = {}

        self.gamma = 0.9
        self.alpha = 0.001
        self.beta = 0.001
        self.beta2 = 0.0001


        for d in range(self.n_doc):
            for w in self.doc[d, :].nonzero()[1]:
                b = np.random.randint(2)
                z = np.random.randint(self.Z)

                self.x_word_in_doc[(d, w)] = b
                self.z_word_in_doc[(d, w)] = z
                self.cnt_doc_x[d, b] += 1
                self.cnt_word_x[w, b] += 1


                self.cnt_doc_topic[d, z, b] += 1
                self.cnt_topic_word[z, w, b] += 1







    def phi(self):

        num = self.cnt_topic_word + self.beta
        for i in range(self.Z):
            num[i, :] /= num[i, :].sum()
        #num /= np.sum(num, axis = 0)

        return num

    def omega(self):
        num = self.cnt_word_x[:, 1] + self.beta2
        print num.shape
        for i in range(self.n_word):
            num[i] /= num.sum()
        return num

    def do_modify_stats(self, d, w, z, _add):
        if _add == -1:
            if self.cnt_doc_topic[d, z] <0:
                print 'wrong for cnt_doc_topic'
            if self.cnt_doc_x[d, 0] < 0:
                print 'wrong for cnt_doc_x[d, 0]'
            if self.cnt_topic_word[z, w] <=0:
                print 'wrong for cnt_topic_word'
            if self.cnt_topic[0, z] <= 0:
                print 'wrong for cnt_topic[0, z]'

        self.cnt_doc_topic[d, z] += _add  #ok
        self.cnt_doc_x[d, 0] += _add      #3
        self.cnt_word_x[w, 0] += _add
        self.cnt_topic_word[z, w] += _add #p
        self.cnt_topic[0, z] += _add      #3





    def sample_index(self, p):
        return np.random.multinomial(1, p).argmax()

    def fit(self):
        for iteration in range(self.n_iteration):
            print 'iteration... ', iteration
            for d in range(self.n_doc):
                if d % 100 == 0:
                    print 'in doc ', d
                for w in self.doc[d, :].nonzero()[1]:

                    cur_z = self.z_word_in_doc[(d,w)]

                    self.cnt_topic_word[cur_z, w, 0] -= 1
                    self.cnt_doc_topic[d, cur_z, 0] -= 1

                    if self.cnt_topic_word[cur_z, w, 0]<0:
                        print 'ERROR! self.cnt_topic_word', self.cnt_topic_word[cur_z, w, 0]
                    if self.cnt_doc_topic[d, cur_z, 0] <0:
                        print 'ERROR! self.cnt_doc_topic[d, cur_z]...'

                    tmp = [0]*self.Z

                    for k in range(self.Z):
                        tmp[k] = (self.cnt_doc_topic[d, k, 0] + self.alpha)*(self.cnt_topic_word[k, w, 0] + self.beta)/(
                            (self.cnt_doc_topic[d, :, 0].sum() + self.alpha*self.Z)*(
                                self.cnt_topic_word[k, :, 0].sum() + self.beta*self.n_word)
                        )

                    tmp = np.asarray(tmp)
                    tmp /= tmp.sum()

                    print 'tmp is ', tmp


                    #print tmp
                    new_z = self.sample_index(tmp)
                    self.z_word_in_doc[(d, w)] = new_z
                    print 'and I sample -> ', new_z
                    #print 'new z = ', new_z

                    self.cnt_topic_word[new_z, w, 0] += 1
                    self.cnt_doc_topic[d, new_z, 0] += 1


                    cur_b = self.x_word_in_doc[(d, w)]
                    if cur_b == 0:
                        self.cnt_topic_word[new_z, w] -= 1
                        self.cnt_doc_topic[d, new_z] -= 1

                    self.cnt_doc_x[d, cur_b] -= 1
                    self.cnt_word_x[w, cur_b] -= 1


                    if self.cnt_doc_x[d, cur_b]<0:
                        print 'ERROR! self.cnt_doc_x[d, cur_b]'
                    if self.cnt_word_x[w, cur_b]<0:
                        print 'ERROR! self.cnt_word_x[w, cur_b]'

                    tmp = [0]*2
                    p0 =  (self.cnt_doc_x[d, 1] + self.gamma)*1.0/(self.cnt_doc_x[d, :].sum() + self.gamma*2)

                    tmp[1] = p0 * (self.cnt_word_x[w, 1] + self.beta2) /(self.cnt_word_x[:, 1].sum() + self.beta2*self.n_word)

                    tmp[0] = (1-p0) * (self.cnt_doc_topic[d, new_z] + self.alpha)*(self.cnt_topic_word[new_z, w] + self.beta)/(
                            (self.cnt_doc_topic[d, :].sum() + self.alpha*self.Z)*(
                                self.cnt_topic_word[new_z, :].sum() + self.beta*self.n_word)
                        )
                    pp = tmp[1]*1.0/(tmp[1] + tmp[0])
                    #print 'sample b prob = ', pp
                    new_b = np.random.binomial(1,pp)
                    #print 'new b = ', new_b

                    self.cnt_doc_x[d, new_b] += 1
                    self.cnt_word_x[w, new_b] += 1

                    self.x_word_in_doc[(d, w)] = new_b

                    if new_b == 0:
                        self.cnt_topic_word[new_z, w] += 1
                        self.cnt_doc_topic[d, new_z] += 1

                    self.cnt_doc_x[d, new_b] += 1
                    self.cnt_word_x[w, new_b] += 1


                    """
                    z = self.z_word_in_doc[(d, w)]
                    self.do_modify_stats(d, w, z, -1)

                    tmp = [0]*self.Z
                    for k in range(self.Z):
                        tmp[k] = (self.cnt_doc_topic[d, k] + self.alpha)*(self.cnt_topic_word[k, w] + self.beta)*1.0 /\
                                 (self.cnt_topic[0, k] + self.n_word*self.beta)
                    tmp = np.asarray(tmp)
                    tmp /= tmp.sum()
                    z = self.sample_index(tmp)

                    self.do_modify_stats(d, w, z, 1)
                    self.z_word_in_doc[(d, w)] = z
                    """

            phi = self.phi()

            print 'Printing topics...'

            for k in range(self.Z):
                for w in np.argsort(phi[k, :])[::-1][:30]:
                    print "'", self.vectorizer.get_feature_names()[w], "'",
                print '\n'



            ome = self.omega()

            print 'stop words...'
            for w in np.argsort(ome)[::-1][:30]:
                print "'", self.vectorizer.get_feature_names()[w], "'", ome[w]
            print '\n'


            print 'Iteration %d ' % (iteration)

model = LDA()
model.fit()

