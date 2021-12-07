import matplotlib.pyplot as plt
import numpy as np

mlp = open("accuracyMLP.txt","r")
bernoulli = open("accuyracyBernoulli.txt","r")
passive = open("accuracyPassive.txt","r")
acc_mlp = []
acc_bernoulli = []
acc_passive = []
batch = []
with open('accuracyMLP.txt') as mlp:
    for line in mlp:
        acc_mlp.append(float(line))
    mlp.close()
with open('accuyracyBernoulli.txt') as bernoulli:
    for line in bernoulli:
        acc_bernoulli.append(float(line))
    bernoulli.close()
with open('accuracyPassive.txt') as passive:
    for line in passive:
        acc_passive.append(float(line))
    passive.close()
for i in range(1,len(acc_mlp) +1):
  batch.append(i)
width = 0.25
x = np.arange(len(batch))

bar1 = plt.bar(x, acc_mlp, width, color = 'r')
bar2 = plt.bar(x+width, acc_bernoulli, width, color = 'g')
bar3 = plt.bar(x + (width*2), acc_passive, width, color = 'b')
plt.xlabel("Batches")
plt.ylabel("Accuracy")
plt.title("Accuracy vs models")
plt.xticks(x+width*3,batch)
plt.legend( (bar1, bar2, bar3, ("MLP","Bernoulli","Passive") )
plt.show()
plt.title('Accuracy vs models')
plt.xlabel('batch')
plt.ylabel('Accuracy')
plt.plot(batch, acc_mlp, label = "MLP")
plt.plot(batch, acc_bernoulli, label = "Bernoulli")
plt.plot(batch, acc_passive, label = "Passive")
plt.legend()
plt.show()