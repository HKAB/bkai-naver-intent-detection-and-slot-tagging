
def rotate(text, labels, intents):
	augment_seq_in = []
	augment_seq_out = []
	augment_intent = []
	for line, label, intent in zip(text, labels, intents):
		# augment_seq_in.append(line)
		# augment_seq_out.append(label)
		# augment_intent.append(intent)

		line_e = line.strip().split()
		label_e = label.strip().split()

		count_start_O = 0
		for l in label_e:
			if (l == "O"):
				count_start_O+=1
			else:
				break

		count_end_O = 0
		for l in reversed(label_e):
			if (l == "O"):
				count_end_O += 1
			else:
				break

		count_end_O = len(label_e) - count_end_O
		if (count_start_O >= 2 and count_end_O >= 1):
			new_line_e = line_e[count_start_O:count_end_O] + line_e[:count_start_O] + line_e[count_end_O:] 
			new_label_e = label_e[count_start_O:count_end_O] + label_e[:count_start_O] + label_e[count_end_O:] 

			augment_seq_in.append(' '.join(new_line_e).strip())
			augment_seq_out.append(' '.join(new_label_e).strip())
			augment_intent.append(intent)
		elif (count_start_O >= 2):
			new_line_e = line_e[count_start_O:] + line_e[:count_start_O]
			new_label_e = label_e[count_start_O:] + label_e[:count_start_O]

			augment_seq_in.append(' '.join(new_line_e).strip())
			augment_seq_out.append(' '.join(new_label_e).strip())
			augment_intent.append(intent)

	# with open('augment_seq_in.txt', 'w') as f:
	# 	f.write('\n'.join(augment_seq_in))
	# with open('augment_seq_out.txt', 'w') as f:
	# 	f.write('\n'.join(augment_seq_out))
	# with open('augment_label.txt', 'w') as f:
	# 	f.write('\n'.join(augment_seq_out))
		
	return augment_seq_in, augment_seq_out, augment_intent

if __name__ == '__main__':
	text = ["bạn tăng bóng led 3 ở phòng tắm xông hơi 2 phòng 10 lên 26 phần trăm hộ mình với bôi đen bạn bỏ dấu ngoặc kép ra nhé v"]
	labels = ["O O B-devicedevice I-devicedevice I-devicedevice O B-roomroom I-roomroom I-roomroom I-roomroom I-roomroom B-floornumberfloornumber I-floornumberfloornumber O B-change-valuesyspercentage I-change-valuesyspercentage I-change-valuesyspercentage O O O O O O O O O O O O O"]
	intents = ['smart.home.set.level']
	
	print(rotate_sentence(text, labels, intents))
# 	with open('training_data/seq.in', 'r') as f:
# 		text = f.read().splitlines()

# 	with open('training_data/seq.out', 'r') as f:
# 		labels = f.read().splitlines()

