import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

main_folder = "/home/ai/data"
main_data_file = "/home/ai/luantt/custom_scripts/data/cafe_aesthetic_1_5M.json"
main_data = json.load(open(main_data_file))

platform_characters_raw = open("/home/ai/luantt/custom_scripts/data/platform_characters.csv").readlines()
platform_characters_raw = [x.split(',')[2].strip().replace("_", " ") for x in platform_characters_raw[1:]]
platform_characters = []
platform_series = []
for character_raw in platform_characters_raw:
    try:
        index = character_raw.index('(')
        character = character_raw[:index].strip()
        series = character_raw[index+1:-1].strip().replace('(', '').replace(')', '')
        platform_characters.append(character)
        platform_series.append(series)
    except ValueError:
        platform_characters.append(character_raw)

platform_characters = list(set(platform_characters))
platform_series = list(set(platform_series))

print(f"Number of characters: {len(platform_characters)}")
print(f"Samples: {platform_characters[:10]}")
print(f"Number of series: {len(platform_series)}")
print(f"Samples: {platform_series[:10]}")

output_file = "/home/ai/luantt/custom_scripts/data/main_data.json"

def add_tag_data(item):
    new_item = {}
    image_path = list(item.keys())[0]
    aesthetic_score = item[image_path]
    image_id = os.path.basename(image_path).split('_')[0]
    txt_file = f"{main_folder}/{image_id}.txt"
    raw_txt = open(txt_file).readline().strip()
    rating, tags = raw_txt.split(',', 1)
    new_item['rating'] = rating.strip()
    tags = tags.strip()
    tags = [tag.strip() for tag in tags.split(',')]
    tags = [tag for tag in tags if tag != '']
    new_item['tags'] = tags
    new_item['aesthetic_score'] = aesthetic_score
    new_item['image_path'] = image_path
    return new_item

if __name__ == "__main__":
    with Pool(64) as p:
        main_data = list(tqdm(p.imap(add_tag_data, main_data), total=len(main_data)))
    json.dump(main_data, open(output_file, 'w'), indent=4)