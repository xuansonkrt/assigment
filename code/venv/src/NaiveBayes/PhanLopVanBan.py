import  warnings
warnings.filterwarnings('ignore')

#load data set from training data
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset="train", shuffle=True)
#print('twenty_train: ', twenty_train)


print("The number of training examples", len(twenty_train.data))
#print top five training examples
print("top 5:", twenty_train.data[0:5])

# You can check the target names (categories) and some data files by following commands.
print('target: ', twenty_train.target_names)
targets = twenty_train.target
print(targets)

print(len(targets))

print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file
