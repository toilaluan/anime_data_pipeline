import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
class NovelAITagOrder:
    def __init__(self):
        tagger_tags = pd.read_csv('assets/selected_tags.csv')
        tagger_tags_name = tagger_tags['name'].values
        tagger_tags_category = tagger_tags['category'].values
        self.base_tags = {k.replace('_', ' '):v for k, v in zip(tagger_tags_name, tagger_tags_category)}
        self.trigger_non_char_words = [':', 'x', 'resolution', 'aspect', 'ratio']

    def build_an_item(self, item):
        tags = item['tags']
        non_character_tags = []
        character_tags = []
        prefix_tags = []
        tags = list(set(tags))
        for tag in tags:
            if 'girl' in tag or 'boy' in tag:
                prefix_tags.append(tag)
                continue
            category = self.base_tags.get(tag, -1)
            if category == -1:
                is_character = True
                for w in self.trigger_non_char_words:
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
    def __call__(self, metadata, workers=1):
        keys = list(metadata.keys())
        values = list(metadata.values())
        if workers > 1:
            with Pool(workers) as p:
                items = list(tqdm(p.map(self.build_an_item, values), total=len(values)))
        else:
            items = []
            for value in tqdm(values):
                items.append(self.build_an_item(value))
        metadata = {k:v for k, v in zip(keys, items)}
        return metadata