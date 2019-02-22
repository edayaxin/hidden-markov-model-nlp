import numpy as np
import sys

def readMatrix(filepath):
	res = []
	with open(filepath, "r") as f:
		row = f.readline()
		while row:
			row = row[:-1]
			rowlist = row.split(" ")
			rowlist = [float(e) for e in rowlist]
			res.append(rowlist)
			row = f.readline()
	res = np.array(res)
	return res

def normalize(matrix, col):
	colSum = 0
	for i in range(len(matrix)):
		colSum = colSum + matrix[i][col]

	for i in range(len(matrix)):
		matrix[i][col] =  matrix[i][col]/colSum

	return matrix

# test_input = "./toydata/toytrain.txt"
# index_to_word = "./toydata/toy_index_to_word.txt"
# index_to_tag = "./toydata/toy_index_to_tag.txt"

train_input = "./fulldata/trainwords.txt"
test_input = "./fulldata/testwords.txt"
index_to_word = "./fulldata/index_to_word.txt"
index_to_tag = "./fulldata/index_to_tag.txt"

hmmprior = "hmmprior.txt"
hmmemit = "hmmemit.txt"
hmmtrans = "hmmtrans.txt"

metrics = "metrics.txt"
predictedtest = "predictedtest.txt"

# python forwardbackward.py ./toydata/toytrain.txt ./toydata/toy_index_to_word.txt 
# ./toydata/toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predictedtest.txt metrics.tx

# test_input = sys.argv[1]
# index_to_word = sys.argv[2]
# index_to_tag = sys.argv[3]

# hmmprior = sys.argv[4]
# hmmemit = sys.argv[5]
# hmmtrans = sys.argv[6]
# predictedtest = sys.argv[7]
# metrics = sys.argv[8]

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
		tag_dic[index] = t
		index += 1
		t = f.readline()

tagCnt = len(tag_dic)
prior = readMatrix(hmmprior)
emit = readMatrix(hmmemit)
trans = readMatrix(hmmtrans)

predicted_output = open(predictedtest, "w")
metrics_output = open(metrics, "w")
with open(train_input, "r") as f:
	seq = f.readline()
	logLikelihood = 0
	sampleCnt = 0
	totalWordsCnt = 0
	errCnt = 0
	while seq:
		sampleCnt += 1
		seq = seq.strip("\n").split(" ")

		words = map(lambda x: word_dic[x.split("_")[0]], seq)
		words_letter = map(lambda x: x.split("_")[0], seq)
		tags = map(lambda x: x.split("_")[1], seq)

		seqCnt = len(words)
		totalWordsCnt += seqCnt
		alpha = np.zeros((tagCnt, seqCnt))
		beta = np.zeros((tagCnt, seqCnt))

		# calculate alpha
		for j in range(0, tagCnt):
			alpha[j][0] = prior[j][0]*emit[j][words[0]]
		# if seqCnt > 1:
		alpha = normalize(alpha, 0)

		for t in range(1, seqCnt):
			for j in range(0, tagCnt):
				a_jt = 0
				for k in range(0, tagCnt):
					a_jt = a_jt + alpha[k][t-1] * trans[k][j]
				alpha[j][t] = a_jt*emit[j][words[t]]

			#normalization
			if t < seqCnt - 1:
				alpha = normalize(alpha, t)


		# calculate beta
		for j in range(0, tagCnt):
			beta[j][seqCnt-1] = 1

		for t in range(seqCnt-2, -1, -1):
			for j in range(0, tagCnt):
				for k in range(0, tagCnt):
					beta[j][t] = beta[j][t] + beta[k][t+1]*trans[j][k]*emit[k][words[t+1]]

			# normalization
			beta = normalize(beta, t)

		# get tag
		predict_tags = [-1 for i in range(seqCnt)]
		for t in range(seqCnt):
			maxProb = -1
			predict_tag = None
			for j in range(tagCnt):
				if maxProb < alpha[j][t]*beta[j][t]:
					maxProb = alpha[j][t]*beta[j][t]
					predict_tag = tag_dic[j]
			if tags[t] != predict_tag:
				errCnt += 1	
			predict_tags[t] = predict_tag


		alphaT_sum = 0
		for j in range(tagCnt):
			alphaT_sum += alpha[j][seqCnt-1]
		logLikelihood += np.log(alphaT_sum)

		# print "prediction:", predict_tags
		s = ""
		for i in range(len(predict_tags)):
			s = s + words_letter[i]+"_"+predict_tags[i]+" "
		s = s[:-1] + "\n"
		predicted_output.write(s)
		seq = f.readline()


	logLikelihood = logLikelihood/sampleCnt
	accuracy = float(totalWordsCnt - errCnt) / totalWordsCnt
	s = "Average Log-Likelihood: " + str(logLikelihood) + "\n" + "Accuracy: " + str(accuracy) + "\n"

	metrics_output.write(s)

	# print "logLikelihood sum", logLikelihood
	# print "sampleCnt", sampleCnt	
	print "logLikelihood", logLikelihood
	# print "accuracy", accuracy

predicted_output.close()
metrics_output.close()





