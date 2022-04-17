from vncorenlp import VnCoreNLP
from tqdm import tqdm


annotator = VnCoreNLP(	"./VnCoreNLP-master/VnCoreNLP-1.1.1.jar", 
						annotators="wseg,pos,ner,parse", 
						max_heap_size='-Xmx2g') 

train_dir = "./bkai/train/"
dev_dir = "./bkai/dev/"
test_dir = "./bkai/test/"

with open(train_dir + "seq.in", "r", encoding="utf-8") as f:
    lines = ""
    for line in tqdm(f):
        lines = lines + " ".join(annotator.tokenize(line.strip())[0]).strip() + "\n"
    with open(train_dir + "seq_segment.in", "w", encoding="utf-8") as f_seg:
    	f_seg.write(lines)

with open(dev_dir + "seq.in", "r", encoding="utf-8") as f:
    lines = ""
    for line in tqdm(f):
        lines = lines + " ".join(annotator.tokenize(line.strip())[0]).strip() + "\n"
    with open(dev_dir + "seq_segment.in", "w", encoding="utf-8") as f_seg:
    	f_seg.write(lines)

with open(test_dir + "seq.in", "r", encoding="utf-8") as f:
    lines = ""
    for line in tqdm(f):
        lines = lines + " ".join(annotator.tokenize(line.strip())[0]).strip() + "\n"
    with open(test_dir + "seq_segment.in", "w", encoding="utf-8") as f_seg:
    	f_seg.write(lines)

print("Segmentation done!")