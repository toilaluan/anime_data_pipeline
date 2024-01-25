import json
import tqdm

N = 300000
metadata_lat_file = '/home/ai/luantt/custom_scripts/data/all_lat.json'
data_file = '/home/ai/luantt/custom_scripts/data/ordered_tag_main_data.json'
output_file = f'/home/ai/luantt/custom_scripts/data/final_train_metdata_{N}.json'

metadata_lat = json.load(open(metadata_lat_file))
data = json.load(open(data_file))
output = {}

for item in tqdm.tqdm(data):
    if item['aesthetic_score'] < 0.8:
        continue
    h, w = metadata_lat[item['image_path']]['train_resolution']
    output[item['image_path']] = metadata_lat[item['image_path']]
    output[item['image_path']]['tags'] = item['ordered_tags']
    if len(output) >= N:
        break
print(len(output))
json.dump(output, open(output_file, 'w'), indent=4)