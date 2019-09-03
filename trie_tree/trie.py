#!/usr/bin/env/python2
# -*- coding: utf-8 -*-

# 单数组法
class TrieNode():
    """前缀树节点"""
    def __init__(self):
        self.is_word = False
        self.children = [None] * 26

class Trie():
    """前缀树"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self,word):
        p = self.root
        for i in range(len(word)):
            if not p.children[ord(word[i]) - ord('a')]:
                p.children[ord(word[i]) - ord('a')] = TrieNode()
            p = p.children[ord(word[i]) - ord('a')]
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
                p = p.children[ord(word[i]) - ord('a')]
        return p

if __name__ == '__main__':
    trie = Trie()
    words = ["apple","app"]
    for i in words:
        trie.insert(i)

    print trie.search("apple") # 返回true
    print trie.search("ap") # 返回false
    print trie.start_with("ap") # 返回true
    trie.insert("app")
    print trie.search("app") # 返回true