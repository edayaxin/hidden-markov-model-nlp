import numpy as np
import sys

# train_input = "./toydata/toytrain.txt"
# test_input = "./toydata/toytest.txt"
# index_to_word = "./toydata/toy_index_to_word.txt"
# index_to_tag = "./toydata/toy_index_to_tag.txt"

train_input = "./fulldata/trainwords.txt"
test_input = "./fulldata/testwords.txt"
index_to_word = "./fulldata/index_to_word.txt"
index_to_tag = "./fulldata/index_to_tag.txt"

hmmprior = "hmmprior.txt"
hmmemit = "hmmemit.txt"
hmmtrans = "hmmtrans.txt"

# python learnhmm.py ./toydata/toytrain.txt ./toydata/toy_index_to_word.txt ./toydata/toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt 

# train_input = sys.argv[1]
# index_to_word = sys.argv[2]
# index_to_tag = sys.argv[3]

# hmmprior = sys.argv[4]
# hmmemit = sys.argv[5]
# hmmtrans = sys.argv[6]

word_dic = {}
tag_dic = {}

index = 0
with open(index_to_word) as f:
	w = f.readline()
	while w:
		w = w.strip("\n")
		word_dic[w] = index
		index += 1
		w = f.readline()

index = 0
with open(index_to_tag) as f:
	t = f.readline()
	while t:
		t = t.strip("\n")
		tag_dic[t] = index
		index += 1
		t = f.readline()

prior = np.ones(len(tag_dic))
emit = np.ones((len(tag_dic), len(word_dic)))
trans = np.ones((len(tag_dic), len(tag_dic)))

seqCnt = 0
with open(train_input) as f:
	seq = f.readline()

	while seq: 
		if seqCnt == 10:
			break
		seqCnt += 1
		seq = seq.strip("\n").split(" ")
		prior[tag_dic[seq[0].split("_")[1]]] += 1

		if len(seq) == 1:
			curTag = seq[0].split("_")[1]
			curWord = seq[0].split("_")[0]
			emit[tag_dic[curTag]][word_dic[curWord]] += 1

		for i in range(0, len(seq)-1):
			curTag = seq[i].split("_")[1]
			curWord = seq[i].split("_")[0]
			nextTag = seq[i+1].split("_")[1]
			nextWord = seq[i+1].split("_")[0]
			emit[tag_dic[curTag]][word_dic[curWord]] += 1
			if  i == len(seq) - 2:
				emit[tag_dic[nextTag]][word_dic[nextWord]] += 1

			trans[tag_dic[curTag]][tag_dic[nextTag]] += 1

		seq = f.readline()

# print emit
# print emit.sum(axis=1)
prior = prior/prior.sum()

for i in range(len(emit)):
	emit[i] = emit[i]/emit[i].sum()

for i in range(len(trans)):
	trans[i] = trans[i]/trans[i].sum()

with open(hmmemit, "w") as f:
	for row in emit:
		line = ' '.join(str(e) for e in row)
		f.write(line)
		f.write("\n")

with open(hmmprior, "w") as f:
	for ele in prior:
		f.write(str(ele))
		f.write("\n")

with open(hmmtrans, "w") as f:
	for row in trans:
		line = ' '.join(str(e) for e in row)
		f.write(line)
		f.write("\n")		

