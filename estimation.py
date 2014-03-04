# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import corpus


def main():
    # 辞書の読み込み
    dictionary = corpus.get_dictionary(create_flg=False)
    # 記事の読み込み
    contents = corpus.get_contents()

    # 特徴抽出
    data_train = []
    label_train = []
    for file_name, content in contents.items():
        data_train.append(corpus.get_vector(dictionary, content))
        label_train.append(corpus.get_class_id(file_name))

    # 分類器
    estimator = RandomForestClassifier()

    # 学習
    estimator.fit(data_train, label_train)

    # 学習したデータを予測にかけてみる（ズルなので正答率高くないとおかしい）
    print("==== 学習データと予測データが一緒の場合")
    print(estimator.score(data_train, label_train))

    # 学習データと試験データに分けてみる
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.5)

    # 分類器をもう一度定義
    estimator2 = RandomForestClassifier()

    # 学習
    estimator2.fit(data_train_s, label_train_s)
    print("==== 学習データと予測データが違う場合")
    print(estimator2.score(data_test_s, label_test_s))

    # グリッドサーチやってみる
    tuned_parameters = [{'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150], 'max_features': ['auto', 'sqrt', 'log2', None]}]

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=2, scoring='accuracy', n_jobs=-1)
    clf.fit(data_train_s, label_train_s)

    print("==== グリッドサーチ")
    print("  ベストパラメタ")
    print(clf.best_estimator_)

    print("トレーニングデータでCVした時の平均スコア")
    for params, mean_score, all_scores in clf.grid_scores_:
            print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = label_test_s, clf.predict(data_test_s)
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    main()
