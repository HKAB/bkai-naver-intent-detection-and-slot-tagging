



if __name__ == '__main__':

	with open('training_data/seq.in', 'r') as f:
		text = f.read().splitlines()

	with open('training_data/seq.out', 'r') as f:
		label = f.read().splitlines()


