from dataruu.bucketing import BucketManager
from dataruu.tags_ordering import NovelAITagOrder
from dataruu.tagger import Tagger
import argparse
import glob
import json
import random
from PIL import Image
from tqdm import tqdm
from dataruu.utils import config
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dirs", help="directory contains images, txts", type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument("--out_json", type=str, help="output json file")
    parser.add_argument("--bucket.no_upscale", action="store_true", help="do not upscale images", default=False)
    parser.add_argument("--bucket.max_reso", type=str, default="1024,1024", help="define max pixels threshold for buckets")
    parser.add_argument("--bucket.min_size", type=int, default=256, help="define min size threshold for buckets")
    parser.add_argument("--bucket.max_size", type=int, default=2048, help="define max size threshold for buckets")
    parser.add_argument("--bucket.reso_steps", type=int, default=64, help="define steps for buckets")
    parser.add_argument("--aesthetic.filter", action="store_true", help="skip aesthetic score", default=False)
    parser.add_argument("--aesthetic.threshold", type=float, default=0.5, help="define threshold for aesthetic score")
    parser.add_argument("--aesthetic.files", type=lambda s: [item.strip() for item in s.split(',')], default=None, help="aesthetic scores file")
    parser.add_argument("--tags.use_synthesize", action="store_true", help="synthesize tags")
    args = config(parser)
    return args

def main(args):
    bucket_manager = BucketManager(
        no_upscale=args.bucket.no_upscale,
        max_reso=tuple(map(int, args.bucket.max_reso.split(','))),
        min_size=args.bucket.min_size,
        max_size=args.bucket.max_size,
        reso_steps=args.bucket.reso_steps
    )
    image_paths = []
    for image_dir in args.image_dirs:        
        image_paths.extend(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpeg"))
        print(len(image_paths))
    random.shuffle(image_paths)
    # image_paths = image_paths[-1000:]
    metadata = bucket_manager(image_paths)

    total_tags = {}
    ratings = {}
    for image_path in image_paths:
        image_dir = image_path.split('/')[:-1]
        image_dir = '/'.join(image_dir)
        image_name = image_path.split('/')[-1].split('.')[0].split('_')[0]
        txt_path = f"{image_dir}/{image_name}.txt"
        item_tags = []
        rating = ""
        if os.path.exists(txt_path) and not args.tags.use_synthesize:
            with open(txt_path) as f:
                tags = f.readline()
                rating, tags = tags.split(',', 1)
                tags = tags.strip()
                tags = tags.split(",")
                tags = [tag.strip() for tag in tags]
                rating = rating.strip()
                item_tags = tags
        total_tags[image_path] = item_tags
        ratings[image_path] = rating
    for image_key in tqdm(metadata.keys(), total=len(metadata)):
        metadata[image_key]['tags'] = total_tags[image_key]
        metadata[image_key]['rating'] = ratings[image_key]

    if args.aesthetic.filter:
        assert args.aesthetic.files is not None, "aesthetic files is required"
        aesthetic_scores = []
        for file in args.aesthetic.files:
            aesthetic_scores.extend(json.load(open(file)))
        aesthetic_scores = {list(d.keys())[0]: list(d.values())[0] for d in aesthetic_scores}
        aesthetic_scores = {k.split('/')[-1].split('_')[0]: v for k,v in aesthetic_scores.items()}
        print("Aesthetic Scores:", len(aesthetic_scores))
        filtered_metadata = {}
        print("Before filtering: ", len(metadata))
        total_error = 0
        for image_key in tqdm(metadata.keys(), total=len(metadata)):
            image_name = image_key.split('/')[-1].split('_')[0]
            if image_name not in aesthetic_scores:
                total_error += 1
                continue
            aesthetic_score = aesthetic_scores[image_name]
            if aesthetic_score >= args.aesthetic.threshold:
                filtered_metadata[image_key] = metadata[image_key]
        metadata = filtered_metadata
        print("After filtering: ", len(metadata))
        print(f"Total errors: {total_error}")
        print("Filtered metadata by aesthetic score")
    
    if args.tags.use_synthesize:
        tagger = Tagger()
        print("Synthesizing tags")
        for image_key in tqdm(metadata.keys(), total=len(metadata)):
            image = Image.open(image_key)
            rating, character_res, general_res = tagger.predict(image)
            character_tags = [(k, v) for k, v in character_res.items() if v > 0.4]
            general_tags = [(k, v) for k, v in general_res.items() if v > 0.4]
            # sort by confidence
            character_tags = sorted(character_tags, key=lambda x: x[1], reverse=True)
            general_tags = sorted(general_tags, key=lambda x: x[1], reverse=True)
            tags = [t[0] for t in character_tags + general_tags]
            metadata[image_key]['ordered_tags'] = ','.join(tags)
            metadata[image_key]["tags"] = ','.join(tags)
            metadata[image_key]['rating'] = rating
    else:
        print("Using tags from txt files")
        tag_order = NovelAITagOrder()
        metadata = tag_order(metadata)
    
    print(len(metadata))
    

    with open(args.out_json, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)