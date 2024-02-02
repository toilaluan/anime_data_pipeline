import json
import pandas as pd
from tqdm import tqdm
import glob
from multiprocessing import Pool
import re

trigger_non_char_words = [':', 'x', 'resolution', 'aspect', 'ratio']
tagger_tags = pd.read_csv('/home/ai/luantt/custom_scripts/data/selected_tags.csv')
tagger_tags_name = tagger_tags['name'].values
tagger_tags_category = tagger_tags['category'].values
base_tags = {k.replace('_', ' '):v for k, v in zip(tagger_tags_name, tagger_tags_category)}

def build_an_item(item):
    tags = item['tags']
    non_character_tags = []
    character_tags = []
    prefix_tags = []
    tags = list(set(tags))
    for tag in tags:
        if 'girl' in tag or 'boy' in tag:
            prefix_tags.append(tag)
            continue
        category = base_tags.get(tag, -1)
        if category == -1:
            is_character = True
            for w in trigger_non_char_words:
                if w in tag:
                    is_character = False
                    break
            if not is_character:
                non_character_tags.append(tag)
            else:
                character_tags.append(tag)
        else:
            non_character_tags.append(tag)
    non_character_tags = sorted(non_character_tags, key=lambda x: -len(x))
    character_tags = sorted(character_tags, key=lambda x: -len(x))
    ordered_tags = ','.join(prefix_tags) + ',' + ','.join(character_tags) + ',' + ','.join(non_character_tags)
    ordered_tags = ordered_tags.strip(',')
    item['ordered_tags'] = ordered_tags
    return item

if __name__ == "__main__":
    main_data = json.load(open('/home/ai/luantt/custom_scripts/data/main_data.json'))
    with Pool(16) as p:
        results = list(tqdm(p.imap(build_an_item, main_data), total=len(main_data)))
    json.dump(results, open('/home/ai/luantt/custom_scripts/data/ordered_tag_main_data.json', 'w'), indent=4)
    sample