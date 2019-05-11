from gensim.models import Word2Vec

with open('tuyen_ngon_doc_lap.txt', 'r', encoding='utf8') as file:
    corpus = [[word.replace(',', '').replace('.', '') for word in line.strip().split()] for line in file]

GensimWord2Vec_model = Word2Vec(corpus,
                                size=100,
                                min_count=8,  # số lần xuất hiện thấp nhất của mỗi từ vựng
                                window=2,  # khai báo kích thước windows size
                                sg=1,  # sg = 1 sử dụng mô hình skip-grams - sg=0 -> sử dụng CBOW
                                workers=1
                                )
print('Tìm top-10 từ tương đồng với từ: [dân]')
for index, word_tuple in enumerate(GensimWord2Vec_model.wv.similar_by_word("dân")):
    print('%s.%s\t\t%s' % (index, word_tuple[0], word_tuple[1]))
