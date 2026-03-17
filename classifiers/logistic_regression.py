#!/usr/bin/env python3

import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_predict(train_file, dev_file):
    texts = []
    labels = []

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, labels)

    dev_basename = dev_file.rsplit(".", 1)[0]
    output_file = f"results/{dev_basename}-pred-logreg.txt"

    with open(dev_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                text = parts[0]
                X_dev = vectorizer.transform([text])
                pred = clf.predict(X_dev)[0]
                f_out.write(f"{text}\t{pred}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <train-corpus-file> <dev-corpus-file>", file=sys.stderr)
        sys.exit(1)

    train_file = sys.argv[1]
    dev_file = sys.argv[2]

    train_and_predict(train_file, dev_file)
