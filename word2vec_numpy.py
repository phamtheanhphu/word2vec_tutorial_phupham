import numpy as np
import operator
from collections import defaultdict
from tqdm import tqdm

np.random.seed(0)  # set the seed for reproducibility


class Word2Vec():

    def __init__(self):

        self.d = settings['d']
        self.eta = settings['learning_rate']
        self.max_iteration = settings['max_iteration']
        self.window_size = settings['window_size']
        pass

    # Tạo tập dữ liệu huấn luyện cho mô hình
    def generate_training_data(self, corpus):

        # duyệt qua từng câu -> từng từ để đếm số lần xuất hiện cửa từng từ
        # {'một': 7, 'màu': 10, 'xanh': 4, 'chấm': 3, ... 'tang': 1, 'đi': 1, 'vội': 1}
        word_counts = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                word_counts[word] += 1

        # đếm số lượng từ có trong tập dữ liệu -> tổng 37 từ
        self.v_count = len(word_counts.keys())

        # xây dựng tập các từ đăng trưng -> tập N
        # ['chinh', 'chiếc', 'chiến', 'chiều',...'đen', 'đi', 'đã', 'đẹp', 'đồng']
        self.words_list = sorted(list(word_counts.keys()), reverse=False)

        # tạo tập chỉ mục [từ - id]
        # {'chinh': 0, 'chiếc': 1, 'chiến': 2, 'chiều': 3, 'chàm': 4, 'chấm': 5, .. 'đẹp': 35, 'đồng': 36}
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))

        # tạo tập chỉ mục [id - từ] - ngược loại so với self.word_index
        # {0: 'chinh', 1: 'chiếc', 2: 'chiến', 3 ...  34: 'đã', 35: 'đẹp', 36: 'đồng'}
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []

        # duyệt qua từng câu trong tập dữ liệu
        for sentence in corpus:

            sent_len = len(sentence)

            # duyệt qua từng từ trong mỗi câu
            for i, word in enumerate(sentence):

                # chuyển mỗi từ về dạng vector one-hot
                # một [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ... 0, 0, 0]
                # màu [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ... 0, 0, 0]
                w_target = self.word2onehot(sentence[i])

                # sinh tập từ ngữ cảnh cho mỗi từ i dựa trên window size
                # [[0, .. 0, 0],[0, 0, ... 0], [0, ... 0, 0, 1, 0, 0]]
                w_context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])

        return np.array(training_data)

    # hàm trung bình mũ SoftMax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # hàm giúp chuyển đổi 1 từ sang dạng one-hot vector
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # suy diễn tiến feed_forward
    def feed_forward(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # lan truyền ngược back_propagation
    def back_propagation(self, e, h, x):

        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # cập nhật lại trong số cho W1 và W2
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    # huấn luyện mô hình Word2Vec
    def train(self, training_data):

        # process_bar
        process_bar = tqdm(range(int(self.max_iteration)))

        # khởi tại ma hai ma trận trọng số W1 và W2 có kích thước lần lượt là W1 = |N x d| và W2=|d X N|
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.d))  # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.d, self.v_count))  # context matrix

        # lặp qua một số lần nhất định để tối ưu hóa mô hình theo GradientDescent
        for i in process_bar:

            # thiết lập giá trị tổng lỗi ban đầu là 0
            self.loss = 0

            # lần lượt lặp qua từng từ và tập các từ ngữ cảnh tương ứng của nó
            # đề ở dạng one-hot vector
            for w_t, w_c in training_data:

                # quá trình suy diễn tiến (feed forward)
                y_pred, h, u = self.feed_forward(w_t)

                # tính toán sự khác biệt giữa (y_pred) dự đoán và (y) mong muốn
                EI = np.sum([np.subtract(y_pred, word_context) for word_context in w_c], axis=0)

                # quá trình lan truyền ngược để khắc phục lỗi sai lệt giữa y_pred và y (back propagation)
                self.back_propagation(EI, h, w_t)

                # tính toán lại tổng hệ số lỗi của mô hình thông qua hàm loss function được định nghĩa bởi Mikolov
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            process_bar.set_description("Hệ số lỗi (loss): %0.8f - Số lần lặp: %i" % (self.loss, i + 1))

    # tìm kiếm top từ tương đồng của 1 từ
    def word_sim(self, word, top_n):

        print('Tìm top-%s từ tương đồng với từ: [%s]' % (top_n, word))

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        word_sim = {}

        for i in range(self.v_count):
            v_w2 = self.w1[i]

            # tính độ tương đồng dựa trên độ đo cosine
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=operator.itemgetter(1), reverse=True)
        for index, (word, sim) in enumerate(words_sorted[1:top_n + 1]):
            print('%s.%s\t\t%s' % (index + 1, word, sim))

        print('---')


settings = {}

settings['d'] = 5  # số chiều của vector sẽ embed mỗi từ
settings['window_size'] = 2  # kích thước window của mỗi từ mô hình Skip-grams
settings['max_iteration'] = 1000  # số lần lặp tối đa của quá trình huấn luyện Gradient Descent
settings['learning_rate'] = 0.01  # tốc độ học của mô hình tối ưu Gradient Descent

sentences = [
    'Một màu xanh xanh chấm thêm vàng vàng',
    'Một màu xanh chấm thêm vàng cánh đồng hoang vu',
    'Một màu nâu nâu một màu tím tím',
    'Màu nâu tím mắt em tôi ôi đẹp dịu dàng',
    'Một màu xanh lam chấm thêm màu chàm',
    'Thời chinh chiến đã xa rồi sắc màu tôi',
    'Một màu đen đen một màu trắng trắng',
    'Chiều hoang vắng chiếc xe tang đi vội vàng'
]

# tiền xử lý tách câu -> từ ra từng mảng riêng biệt
corpus = [[word.lower() for word in line.strip().split(' ')] for line in sentences]

# khởi tạo mô hình Word2Vec
w2v = Word2Vec()

# tạo dữ liệu huấn luyện từ tập dữ liệu
training_data = w2v.generate_training_data(corpus)

# tiến hành huấn luyện tập dữ liệu bằng mô hình Word2Vec
w2v.train(training_data)

# thử nghiệm tìm kiếm sự tương đồng giữa các từ
w2v.word_sim('màu', 5)
w2v.word_sim('xe', 5)
