#!/usr/bin/env/python2
# -*- coding: utf-8 -*-

# 链表法
class TrieNode():
    """前缀树节点"""
    def __init__(self):
        self.is_word = False
        self.children = dict()

class Trie():
    """前缀树"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self,word):
        p = self.root
        for i in range(len(word)):
            if not p.children.get(word[i]):
                p.children[word[i]] = TrieNode()
            p = p.children[word[i]]
        p.is_word = True

    def search(self,word):
        p = self.find(word)
        return True if p and p.is_word else False

    def start_with(self,word):
        p = self.find(word)
        return True if p else False

    def find(self,word):
        p = self.root
        for i in range(len(word)):
            if p:
                p = p.children.get(word[i])
        return p

if __name__ == '__main__':
    trie = Trie()
    words = [u"清华大学",u"华中科技大学"]
    for i in words:
        trie.insert(i)

    print trie.search(u"清华大学") # 返回true
    print trie.search(u"清华") # 返回false
    print trie.start_with(u"华中")  # 返回true
    print trie.start_with(u"清华") # 返回true
    print trie.start_with(u"华科")  # 返回false
    trie.insert(u"清华")
    print trie.search(u"清华") # 返回true
