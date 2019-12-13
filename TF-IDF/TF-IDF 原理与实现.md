# 一、什么是 TF-IDF？

**TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率)**是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

上述引用总结就是, **一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。**这也就是TF-IDF的含义。

TF-IDF分为 TF 和 IDF，下面分别介绍这个两个概念。

## 1.1 TF

**TF(Term Frequency, 词频)**表示词条在文本中出现的频率，这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否）。TF用公式表示如下
$$
TF_{i,j}=\frac{n_{i,j}}{\sum_{k}{n_{k,j}}}\tag{1}
$$
其中，$n_{i,j}$ 表示词条 $t_i$ 在文档 $d_j$ 中出现的次数，$TF_{i,j}$ 就是表示词条 $t_i$ 在文档 $d_j$ 中出现的频率。

但是，需要注意， 一些通用的词语对于主题并没有太大的作用， 反倒是一些出现频率较少的词才能够表达文章的主题， 所以单纯使用是TF不合适的。权重的设计必须满足：一个词预测主题的能力越强，权重越大，反之，权重越小。所有统计的文章中，一些词只是在其中很少几篇文章中出现，那么这样的词对文章的主题的作用很大，这些词的权重应该设计的较大。IDF就是在完成这样的工作。

## 1.2 IDF

**IDF(Inverse Document Frequency, 逆文件频率)**表示关键词的普遍程度。如果包含词条 $i$ 的文档越少， IDF越大，则说明该词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到
$$
IDF_i=\log\frac{\left|D \right|}{1+\left|j: t_i \in t_j\right|}\tag{2}
$$
其中，$\left|D \right|$ 表示所有文档的数量，$\left|j: t_i \in t_j\right|$ 表示包含词条 $t_i$ 的文档数量，为什么这里要加 1 呢？主要是**防止包含词条 $t_i$ 的数量为 0 从而导致运算出错的现象发生**。

某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于**过滤掉常见的词语，保留重要的词语**，表达为
$$
TF \text{-}IDF= TF \cdot IDF\tag{3}
$$

# 二、Python 实现 

我们用相同的语料库，分别使用 Python 手动实现、使用gensim 库函数以及 sklearn 库函数计算 TF-IDF。

## 2.1 Python 手动实现

+ 输入语料库

```python
corpus = ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document']
words_list = list()
for i in range(len(corpus)):
    words_list.append(corpus[i].split(' '))
print(words_list)
```

```
[['this', 'is', 'the', 'first', 'document'], 
['this', 'is', 'the', 'second', 'second', 'document'], 
['and', 'the', 'third', 'one'], 
['is', 'this', 'the', 'first', 'document']]
```

+ 统计词语数量

```python
from collections import Counter
count_list = list()
for i in range(len(words_list)):
    count = Counter(words_list[i])
    count_list.append(count)
print(count_list)
```

```
[Counter({'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1}), 
Counter({'second': 2, 'this': 1, 'is': 1, 'the': 1, 'document': 1}), 
Counter({'and': 1, 'the': 1, 'third': 1, 'one': 1}), 
Counter({'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1})]
```

+ 定义函数

```python
import math
def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)
```

+ 输出结果

```python
for i, count in enumerate(count_list):
    print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
    scores = {word : tf_idf(word, count, count_list) for word in count}
    sorted_word = sorted(scores.items(), key = lambda x : x[1], reverse=True)
    for word, score in sorted_word:
        print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
```

```
第 1 个文档 TF-IDF 统计信息
	word: first, TF-IDF: 0.05754
	word: this, TF-IDF: 0.0
	word: is, TF-IDF: 0.0
	word: document, TF-IDF: 0.0
	word: the, TF-IDF: -0.04463
第 2 个文档 TF-IDF 统计信息
	word: second, TF-IDF: 0.23105
	word: this, TF-IDF: 0.0
	word: is, TF-IDF: 0.0
	word: document, TF-IDF: 0.0
	word: the, TF-IDF: -0.03719
第 3 个文档 TF-IDF 统计信息
	word: and, TF-IDF: 0.17329
	word: third, TF-IDF: 0.17329
	word: one, TF-IDF: 0.17329
	word: the, TF-IDF: -0.05579
第 4 个文档 TF-IDF 统计信息
	word: first, TF-IDF: 0.05754
	word: is, TF-IDF: 0.0
	word: this, TF-IDF: 0.0
	word: document, TF-IDF: 0.0
	word: the, TF-IDF: -0.04463
```

## 2.2 使用 gensim 算法包实现

使用和 2.1 节相同的语料库 `corpus`，过程如下

+ 获取每个词语的 id 和词频

```python
from gensim import corpora
# 赋给语料库中每个词(不重复的词)一个整数id
dic = corpora.Dictionary(words_list)
new_corpus = [dic.doc2bow(words) for words in words_list]
# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
print(new_corpus)
```

```
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], 
[(0, 1), (2, 1), (3, 1), (4, 1), (5, 2)], 
[(3, 1), (6, 1), (7, 1), (8, 1)], 
[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]
```

+ 查看每个词语对应的 id

```python
print(dic.token2id)
```

```python
{'document': 0, 'first': 1, 'is': 2, 'the': 3, 'this': 4, 'second': 5, 'and': 6, 'one': 7, 'third': 8}
```

+ 训练gensim模型并且保存它以便后面的使用

```python
# 训练模型并保存
from gensim import models
tfidf = models.TfidfModel(new_corpus)
tfidf.save("tfidf.model")
# 载入模型
tfidf = models.TfidfModel.load("tfidf.model")
# 使用这个训练好的模型得到单词的tfidf值
tfidf_vec = []
for i in range(len(corpus)):
    string = corpus[i]
    string_bow = dic.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)
# 输出 词语id与词语tfidf值
print(tfidf_vec)
```

```python
[[(0, 0.33699829595119235), (1, 0.8119707171924228), (2, 0.33699829595119235), (4, 0.33699829595119235)], 
[(0, 0.10212329019650272), (2, 0.10212329019650272), (4, 0.10212329019650272), (5, 0.9842319344536239)], 
[(6, 0.5773502691896258), (7, 0.5773502691896258), (8, 0.5773502691896258)], 
[(0, 0.33699829595119235), (1, 0.8119707171924228), (2, 0.33699829595119235), (4, 0.33699829595119235)]]
```

+ 句子测试

```python
# 测试一个句子
test_words = "i is the first one"
string_bow = dic.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_tfidf)
```

```
[(0, 0.33699829595119235), (1, 0.8119707171924228), (2, 0.33699829595119235), (4, 0.33699829595119235)]
```

> 这里需要注意的是，在打印 tf-idf 值的时候会发现只会显示部分词语，这是因为 gensim 会自动的去除停用词。

## 2.3 使用 sklearn 算法包实现

+ 调包

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)
# 得到语料库所有不重复的词
print(tfidf_vec.get_feature_names())
# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)
# 得到每个句子所对应的向量，向量里数字的顺序是按照词语的id顺序来的
print(tfidf_matrix.toarray())
```

```
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
{'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
[[0.         0.43877674 0.54197657 0.43877674 0.         0.
  0.35872874 0.         0.43877674]
 [0.         0.27230147 0.         0.27230147 0.         0.85322574
  0.22262429 0.         0.27230147]
 [0.55280532 0.         0.         0.         0.55280532 0.
  0.28847675 0.55280532 0.        ]
 [0.         0.43877674 0.54197657 0.43877674 0.         0.
  0.35872874 0.         0.43877674]]
```

具体的 notebook 文件可以见我的 [github 代码]

# 三、参考

[1] https://zh.wikipedia.org/wiki/Tf-idf

[2] https://blog.csdn.net/zrc199021/article/details/53728499

[3] https://www.zybuluo.com/lianjizhe/note/1212780

