# 基于双数组的前缀树
+ 执行`DATrie.py`，函数输入为
```python
words = ["清华", "清华大学", "清新", "中华", "中华人民", "华人","CSDN","Python","Python3","Java","贝壳找房"]
dat = DATrie()
dat.build(words)
#dat.print_trie()
print(dat.match("清"))
print(dat.match("Python"))
print(dat.match("中华人民共和国"))
print(dat.match("贝壳找房"))
print(dat.start_with("贝壳"))
print(dat.start_with("Pytho"))
print(dat.start_with("中"))
```
+ 输出结果
```python
[]
[7]
[3, 4]
[10]
True
True
True
```


# 基于双数组前缀树的中文分词
+ 执行`segment.py`，函数输入为
```python
words = ["中国", "人名", "人民", "孙健", "CSDN", "java",
    "java学习", "部分"]
tree = DATrie.DATrie()
tree.build(words)
text = "中国人名识别是中国人民的一个骄傲.孙健人民在CSDN中学到了很多最早iteye是java学习笔记叫javaeye但是java123只是一部分"
print(segment(tree,text))
```
+ 输出结果
```python
['中国', '人名', '识', '别', '是', '中国', '人民', '的', '一', '个', '骄', '傲', '.', '孙健', '人民', '在', 'CSDN', '中', '学', '到', '了', '很', '多', '最', '早', 'i', 't', 'e', 'y', 'e', '是', 'java学习', '笔', '记', '叫', 'java', 'e', 'y', 'e', '但', '是', 'java', '1', '2', '3', '只', '是', '一', '部分']
```
可以看到目前基于这种方式的中文分析效果不是很好，非常依赖于前缀树的大小，基于机器学习的方法，可以提高分词的准确度。
