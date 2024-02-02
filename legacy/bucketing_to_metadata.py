import argparse
import os
import json

from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

import library.model_util as model_util
import library.train_util as train_util

def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    # assert args.bucket_reso_steps % 8 == 0, f"bucket_reso_steps must be divisible by 8 / bucket_reso_stepは8で割り切れる必要があります"
    if args.bucket_reso_steps % 8 > 0:
        print(f"resolution of buckets in training time is a multiple of 8 / 学習時の各bucketの解像度は8単位になります")
    if args.bucket_reso_steps % 32 > 0:
        print(
            f"WARNING: bucket_reso_steps is not divisible by 32. It is not working with SDXL / bucket_reso_stepsが32で割り切れません。SDXLでは動作しません"
        )

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)]
    print(f"found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        print(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print(f"no metadata / メタデータファイルがありません: {args.in_json}")
        return
    # bucketのサイズを計算する
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert len(max_reso) == 2, f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )
    if not args.bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        print(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
        )

    # 画像をひとつずつ適切なbucketに割り当てながらlatentを計算する
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                # train_util.cache_batch_latents(vae, True, bucket, args.flip_aug, False)
                bucket.clear()
    data = [ip for ip in image_paths]

    bucket_counts = {}
    for image_path in tqdm(data, smoothing=0.0):
        if image_path is None:
            continue
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
            continue

        image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in metadata:
            metadata[image_key] = {}

        # 本当はこのあとの部分もDataSetに持っていけば高速化できるがいろいろ大変

        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        # メタデータに記録する解像度はlatent単位とするので、8単位で切り捨て
        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)

        if not args.bucket_no_upscale:
            # upscaleを行わないときには、resize後のサイズは、bucketのサイズと、縦横どちらかが同じであることを確認する
            assert (
                resized_size[0] == reso[0] or resized_size[1] == reso[1]
            ), f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
            assert (
                resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
            ), f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

        assert (
            resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
        ), f"internal error resized size is small: {resized_size}, {reso}"



        # バッチへ追加
        image_info = train_util.ImageInfo(image_key, 1, "", False, image_path)
        image_info.latents_npz = ''
        image_info.bucket_reso = reso
        image_info.resized_size = resized_size
        image_info.image = image
        bucket_manager.add_image(reso, image_info)

        # バッチを推論するか判定して推論する
        process_batch(False)

    # 残りを処理する
    process_batch(True)

    bucket_manager.sort()
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            print(f"bucket {i} {reso}: {count}")
    img_ar_errors = np.array(img_ar_errors)
    print(f"mean ar error: {np.mean(img_ar_errors)}")

    # metadataを書き出して終わり
    print(f"writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="1024,1024",
        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）",
    )
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale", action="store_true", help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します"
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
