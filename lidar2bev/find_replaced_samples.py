import pickle
from collections import Counter

pkl_path = '/home/rlab10/OpenPCDet/data/kitti/kitti_train_dataset.pkl'
train_txt_path = '/home/rlab10/OpenPCDet/data/kitti/ImageSets/train.txt'

with open(train_txt_path, 'r') as f:
    train_frame_ids = [line.strip() for line in f.readlines()]  

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

pkl_frame_ids = [sample[0]['frame_id'] for sample in data]  

frame_id_counter = Counter(pkl_frame_ids)

missing_or_replaced = []
for idx, frame_id in enumerate(train_frame_ids):
    if frame_id != pkl_frame_ids[idx]:
        missing_or_replaced.append((frame_id, pkl_frame_ids[idx]))


replaced_ids = [replacement for _, replacement in missing_or_replaced]
unique_replacements = [replacement for replacement in replaced_ids if frame_id_counter[replacement] == 1]
duplicate_replacements = [replacement for replacement in replaced_ids if frame_id_counter[replacement] > 1]

for original, replacement in missing_or_replaced:
    print(f"Original: {original} → Replaced: {replacement}")
