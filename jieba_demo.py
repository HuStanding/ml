#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import jieba

str = "现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功"
res = "/".join(jieba.cut(str))

print res
