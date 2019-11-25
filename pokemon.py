import requests as req
from bs4 import BeautifulSoup
import json
import pprint
import pandas as pd
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn import tree
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sb

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
pokemons = {}
types_id = ['Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting',
			'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock',
			'Ghost', 'Dark', 'Dragon', 'Steel', 'Fairy']

def get_data():
	url = 'https://pokemondb.net/pokedex/all'
	get_req = req.get(url)
	soup = BeautifulSoup(get_req.content, 'html.parser')

	rows = soup.find('table').find('tbody').find_all('tr')
	for r in rows:
		cells = r.find_all('td')
		_id = cells[0].get_text()
		name = cells[1].get_text()
		_type = cells[2].get_text()
		total = cells[3].get_text()
		hp = cells[4].get_text()
		atk = cells[5].get_text()
		df = cells[6].get_text()
		spatk = cells[7].get_text()
		spdf = cells[8].get_text()
		spd = cells[9].get_text()
		
		type_split = _type.split(' ')
		primary_type = type_split[0]
		type_id = types_id.index(primary_type)
		pokemons[_id] = {
			'Name':name,
			'Type':_type,
			'Total':int(total),
			'HP':int(hp),
			'TypeID':type_id,
			'Attack':int(atk),
			'Defense':int(df),
			'SpecialAttack':int(spatk),
			'SpecialDefense':int(spdf),
			'Speed':int(spd)
		}

	Pokemon = pd.DataFrame.from_dict(pokemons, columns=['Name','Type','Total','TypeID','HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'], orient='index')
	last_col = [0 for x in range(len(Pokemon.index))]
	Pokemon['Legendary'] = last_col
	return Pokemon

def get_legendary(database):
	url = 'https://nintendo.fandom.com/wiki/Legendary_Pok%C3%A9mon'
	soup = BeautifulSoup(req.get(url).content, 'html.parser')

	for link in soup.find_all('a'):
		try:
			href = link.get('href')
			href = [x.strip() for x in href.split('/')]
			pok_name = href[2]
			found = database.loc[database['Name'] == pok_name]
			if len(found.index) > 0:
				database.loc[found.index[0], 'Legendary'] = 1
		except:
			pass

def tree_plot(clf, feat):
    """
    Funciton that plots the classification tree
    -------

    :param clf: classifier used (gini or entropy)
    :param feat: list of features to use for classify
    :param title: string
    """
    dot_data = tree.export_graphviz(clf, feature_names=feat, class_names=['NonLegendary', 'Legendary'], filled=True,
                                    rounded=True, special_characters=True, out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('DecisionTree.png')

def PCA(PkDf):
	pca = PCA(n_components=2)
	x = PkDf.loc[:, 'HP':'Speed'].values
	y = PkDf.loc[:, 'Legendary'].values

	x = StandardScaler().fit_transform(x)
	# print(x)
	principal_components = pca.fit_transform(x)
	# print(pd.DataFrame(principal_components))
	concat = pd.concat([pd.DataFrame(principal_components, columns=['A','B']), PkDf[['Legendary']]], axis=1)	
	concat.plot(kind='scatter', x='A', y='B', c=PkDf['Legendary'].apply(lambda x: colors[x]))
	plt.show()

def predict(PkDf, n):
	np.random.seed(69)
	PkDf = PkDf.sample(frac=1)
	train = np.random.rand(len(PkDf)) < n
	X = PkDf[train]
	Y = PkDf[~train]

	return X, Y


if __name__ == '__main__':
	PkDf = get_data()
	get_legendary(PkDf)
	PkDf.to_excel('PokemonDatabase.xlsx')
	sb.set()
	PkDf = pd.read_excel('PokemonDatabase.xlsx')
	features = ['TypeID', 'HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']
	clf = tree.DecisionTreeClassifier('gini')
	clf = clf.fit(PkDf.loc[:, 'TypeID':'Speed'], PkDf.loc[:, 'Legendary'])
	tree_plot(clf, features)
	

	colors = {0:'red', 1:'blue'}
	labels = {0:'NonLegendary', 1:'Legendary'}
	samples = np.linspace(0.1,0.99,50)
	# acc_plot = []
	# for n in samples:
	# 	print(n)
	# 	train, test = predict(PkDf, n=n)
	# 	# print(test.loc[:,'Legendary'])
	# 	clf = tree.DecisionTreeClassifier('gini')
	# 	clf = clf.fit(train.loc[:, 'HP':'Speed'], train.loc[:, 'Legendary'])	

	# 	prediction = clf.predict(test.loc[:, 'HP':'Speed'])
	# 	# print(list(test.loc[:,'Legendary']),prediction)
	# 	acc = 0
	# 	for i in range(len(prediction)):
	# 		# print('{} ---> {}'.format(list(test.loc[:,'Name'])[i], labels[prediction[i]]))
	# 		if prediction[i] == list(test.loc[:,'Legendary'])[i]:
	# 			acc+=1
	# 	# print(acc/len(prediction))
	# 	acc_plot.append(1 - (acc/len(prediction)))
	# plt.plot(acc_plot)
	# plt.show()

	# import keras
	# from keras.models import Sequential
	# from keras.layers import Dense

	# train, test = predict(PkDf, n=0.7)
	# # print(train)
	# x_train = train.loc[:,'SpecialAttack':'Speed']
	# y_train = train.loc[:,'Legendary']
	# x_test =  test.loc[:,'SpecialAttack':'Speed']
	# y_test = test.loc[:, 'Legendary']

	# model = Sequential()
	# model.add(Dense(500, activation='relu', input_dim=3))
	# model.add(Dense(200, activation='relu'))
	# model.add(Dense(50, activation='relu'))
	# model.add(Dense(1, activation='sigmoid'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# model.fit(x_train, y_train, epochs=50)

	# print('-------\nTEST\n------')
	# pred = model.predict(x_test)
	# score = model.evaluate(x_test, y_test)
	# print(score[1])



