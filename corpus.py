# -*- coding: utf-8 -*-
import os
import sys
import re
from gensim import corpora, matutils
import MeCab

DATA_DIR_PATH = './text/'
DICTIONARY_FILE_NAME = 'livedoordic.txt'
mecab = MeCab.Tagger('mecabrc')


def get_class_id(file_name):
    '''
    ファイル名から、クラスIDを決定する。
    学習データを作るときに使っています。
    '''
    dir_list = get_dir_list()
    dir_name = next(filter(lambda x: x in file_name, dir_list), None)
    if dir_name:
        return dir_list.index(dir_name)
    return None


def get_dir_list():
    '''
    ライブドアコーパスが./text/ の下にカテゴリ別にあるからそのカテゴリ一覧をとってるだけ
    '''
    tmp = os.listdir(DATA_DIR_PATH)
    if tmp is None:
        return None
    return sorted([x for x in tmp if os.path.isdir(DATA_DIR_PATH + x)])


def get_file_content(file_path):
    '''
    1つの記事を読み込み
    '''
    ret = ''
    with open(file_path) as f:
        tmp = [line for line in f.readlines()][2:]  # ライブドアコーパスが3行目から本文はじまってるから
        ret = ''.join(tmp)

    return ret


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next


def check_stopwords(word):
    '''
    ストップワードだったらTrueを返す
    '''
    if re.search(r'^[0-9]+$', word):  # 数字だけ
        return True
    return False


def get_words(contents):
    '''
    記事群のdictについて、形態素解析して返す
    '''
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''
    return [token for token in tokenize(content) if not check_stopwords(token)]


def filter_dictionary(dictionary):
    '''
    低頻度と高頻度のワードを除く感じで
    '''
    dictionary.filter_extremes(no_below=20, no_above=0.3)  # この数字はあとで変えるかも
    return dictionary


def get_contents():
    '''
    livedoorニュースのすべての記事をdictでまとめておく
    '''
    dir_list = get_dir_list()

    if dir_list is None:
        return None

    ret = {}
    for dir_name in dir_list:
        file_list = os.listdir(DATA_DIR_PATH + dir_name)

        if file_list is None:
            continue
        for file_name in file_list:
            if dir_name in file_name:  # LICENSE.txt とかを除くためです。。
                ret[file_name] = get_file_content(DATA_DIR_PATH + dir_name + '/' + file_name)

    return ret


def get_vector(dictionary, content):
    '''
    ある記事の特徴語カウント
    '''
    tmp = dictionary.doc2bow(get_words_main(content))
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    return dense


def get_dictionary(create_flg=False, file_name=DICTIONARY_FILE_NAME):
    '''
    辞書を作る
    '''
    if create_flg or not os.path.exists(file_name):
        # データ読み込み
        contents = get_contents()
        # 形態素解析して名詞だけ取り出す
        words = get_words(contents)
        # 辞書作成、そのあとフィルタかける
        dictionary = filter_dictionary(corpora.Dictionary(words))
        # 保存しておく
        if file_name is None:
            sys.exit()
        dictionary.save_as_text(file_name)

    else:
        # 通常はファイルから読み込むだけにする
        dictionary = corpora.Dictionary.load_from_text(file_name)

    return dictionary


if __name__ == '__main__':
    get_dictionary(create_flg=True)
