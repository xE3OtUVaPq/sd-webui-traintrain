from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import random
import torch.nn.functional as F

test = False

def make_dataloaders(t):
    find_filesets(t)                    #画像、テキスト、キャプションのパスを取得
    make_buckets(t)                     #画像サイズのリストを作成
    load_resize_image_and_text(t)       #画像を読み込み、画像サイズごとに振り分け、リサイズ、テキストの読み込み
                                        #t.image_bucketsは画像サイズをkeyとしたimage,txt, captionのリスト
    encode_image_text(t)                #画像とテキストをlatentとembeddingに変換

    dataloaders = []                    #データセットのセットを作成
    for key in t.image_buckets:
        if test: save_images(t, key, t.image_buckets_raw[key])
        dataset = LatentsConds(t, t.image_buckets[key])
        if dataset.__len__() > 0:
            dataloaders.append(DataLoader(dataset, batch_size=t.train_batch_size, shuffle=True))

    return dataloaders

class ContinualRandomDataLoader:
    def __init__(self, dataloaders):
        self.original_dataloaders = dataloaders
        self.epoch = 0
        self.data = len(self.original_dataloaders) > 0
        self._reset_iterators()

    def _reset_iterators(self):
        # すべての DataLoader から新しいイテレータを生成
        self.dataloaders = list(self.original_dataloaders)
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iterators:
            # すべての DataLoader が終了したらリセット
            self._reset_iterators()

        while self.iterators:
            # ランダムに DataLoader を選択
            idx = random.randrange(len(self.iterators))
            try:
                return next(self.iterators[idx])
            except StopIteration:
                # 終了した DataLoader をリストから削除
                self.iterators.pop(idx)
                self.dataloaders.pop(idx)

        # すべての DataLoader が終了した場合
        self.epoch += 1
        raise StopIteration

class LatentsConds(Dataset):
    def __init__(self, t, latents_conds):
        self.latents_conds = latents_conds
        self.batch_size = t.train_batch_size
        self.revert = t.diff_revert_original_target
        self.latents_conds = self.latents_conds * t.image_num_multiply
        if t.train_batch_size > len(self.latents_conds):
            self.latents_conds = self.latents_conds * t.train_batch_size

    def __len__(self):
        return len(self.latents_conds)

    def __getitem__(self, i):
        batch = {}
        if isinstance(self.latents_conds[i], tuple):
            origs, targs = self.latents_conds[i]
            if self.revert:
                targs, origs = origs, targs
            orig_latent, orig_mask, orig_cond1, orig_cond2 = origs
            targ_latent, targ_mask, targ_cond1, targ_cond2 = targs

            batch["orig_latent"] = orig_latent.squeeze()
            batch["targ_latent"] = targ_latent.squeeze()
            if orig_cond1 is not None: batch["orig_cond1"] = orig_cond1 if isinstance(orig_cond1, str) else orig_cond1.squeeze().cpu()
            if orig_cond2 is not None: batch["orig_cond2"] = orig_cond2 if isinstance(orig_cond2, str) else orig_cond2.squeeze().cpu()
            if targ_cond1 is not None: batch["targ_cond1"] = targ_cond1 if isinstance(targ_cond1, str) else targ_cond1.squeeze().cpu()
            if targ_cond2 is not None: batch["targ_cond2"] = targ_cond2 if isinstance(targ_cond2, str) else targ_cond2.squeeze().cpu()
            if isinstance(orig_mask, torch.Tensor): batch["mask"] = orig_mask.squeeze().cpu()

        else:
            latent, mask, cond1, cond2 = self.latents_conds[i]
            batch["latent"] = latent.squeeze().cpu()
            if cond1 is not None: batch["cond1"] = cond1 if isinstance(cond1, str) else cond1.squeeze().cpu()
            if cond2 is not None: batch["cond2"] = cond2 if isinstance(cond2, str) else cond2.squeeze().cpu()
            if isinstance(mask, torch.Tensor): batch["mask"] = mask.squeeze().cpu()
        return batch

TARGET_IMAGEFILES = ["jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp", "webp", "pcx", "ico"]

def make_buckets(t):
    increment = t.image_buckets_step # default : 256
    # 最大ピクセル数 resolutionは[x ,y]の配列。 y >= x
    max_pixels = t.image_size[0]*t.image_size[1]

    # 正方形は手動で追加
    max_buckets = set()
    max_buckets.add((t.image_size[0], t.image_size[0]))

    # 最小値から～
    width = t.image_min_length
    # ～最大値まで
    while width <= max(t.image_size):
        # 最大ピクセル数と最大長を越えない最大の高さ
        height = min(max(t.image_size), (max_pixels // width) - (max_pixels // width) % increment)
        ratio = width/height

        # アスペクト比が極端じゃなかったら追加、高さと幅入れ替えたものも追加。
        if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
            max_buckets.add((width, height))
            max_buckets.add((height, width))
        width += increment  # 幅を大きくして次のループへ

    sub_buckets = set()

    # 最小サイズから最大サイズまでの範囲で枠を生成
    for width in range(t.image_min_length, max(t.image_size) + 1, increment):
        for height in range(t.image_min_length, max(t.image_size) + 1, increment):
            if width * height <= max_pixels:
                ratio = width / height
                if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
                    if (width, height) not in max_buckets:
                        sub_buckets.add((width, height))
                    if (height, width) not in max_buckets:
                        sub_buckets.add((height, width))

    # アスペクト比に基づいて枠を並べ替え
    max_buckets = list(max_buckets)
    max_ratios = [w / h for w, h in max_buckets]
    max_buckets = np.array(max_buckets)[np.argsort(max_ratios)]
    max_buckets = [tuple(x) for x in max_buckets]
    max_ratios = np.sort(max_ratios)

    sub_buckets = list(sub_buckets)
    sub_ratios = [w / h for w, h in sub_buckets]
    sub_buckets = np.array(sub_buckets)[np.argsort(sub_ratios)]
    sub_buckets = [tuple(x) for x in sub_buckets]
    sub_ratios = np.sort(sub_ratios)

    t.image_max_buckets_sizes = max_buckets
    t.image_max_ratios = max_ratios
    t.image_sub_buckets_sizes = sub_buckets
    t.image_sub_ratios = sub_ratios
    t.image_buckets_raw = {}
    t.image_buckets = {}
    print("max bucket sizes : ", max_buckets)
    #t.db("max bucket sizes : ", max_ratios)
    print("sub bucket sizes : ", sub_buckets)
    #t.db("sub bucket sizes : ", sub_ratios)
    for bucket in max_buckets + sub_buckets:
        t.image_buckets_raw[bucket] = []
        t.image_buckets[bucket] = []

def find_filesets(t):
    """
    Finds image files and associated text/caption files in the lora_data_directory.
    In Multi-ADDifT mode, it pairs original images with target images based on diff_target_name.

    Updates t.image_pathsets with the results.
    """
    pathsets = []
    t.image_pathsets = [] # Initialize as empty

    # --- Check 1: Validate lora_data_directory ---
    if not hasattr(t, 'lora_data_directory') or not t.lora_data_directory:
        print(f"Error: 'lora_data_directory' is not set or is empty in the trainer object.")
        return # Stop processing
    if not os.path.isdir(t.lora_data_directory):
        print(f"Error: Provided 'lora_data_directory' is not a valid directory: {t.lora_data_directory}")
        return # Stop processing
    # --- End Check 1 ---

    print(f"Scanning directory: {t.lora_data_directory}")

    # Walk through the folder and subfolders to find all relevant files first
    pathdict = {}
    try:
        for root, dirs, files in os.walk(t.lora_data_directory):
            # Skip hidden directories (optional, but often useful)
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]

            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(f".{ext}") for ext in TARGET_IMAGEFILES):
                    image_path = os.path.join(root, file)
                    base_name, ext = os.path.splitext(file)

                    # Derive potential filename tags (handle _id_ if present)
                    filename_tag_base = base_name.split("_id_")[0] if "_id_" in base_name else base_name
                    filename_tags = filename_tag_base.replace("_", ",") # Example transformation

                    # Check for corresponding text file
                    text_file_path = os.path.join(root, base_name + '.txt')
                    text_file_path = text_file_path if os.path.isfile(text_file_path) else None

                    # Check for corresponding caption file
                    caption_file_path = os.path.join(root, base_name + '.caption')
                    caption_file_path = caption_file_path if os.path.isfile(caption_file_path) else None

                    # Store preliminary info, using image_path as key
                    # Format: [full_image_path, full_text_path, full_caption_path, derived_filename_tags]
                    pathdict[image_path] = [image_path, text_file_path, caption_file_path, filename_tags]

    except Exception as e:
         print(f"An error occurred during initial file scan in '{t.lora_data_directory}': {e}")
         return # Stop processing

    # If not Multi-ADDifT, just use the found paths directly
    if t.mode != "Multi-ADDifT":
        # Convert dict values to list for pathsets
        pathsets = [value + [None, None] for value in pathdict.values()] # Add placeholders for pair_size, targ_path
        t.image_pathsets = pathsets
        print(f"Found {len(t.image_pathsets)} image files.")
        # --- Debug counts for non-Multi mode ---
        t.db("Images : ", len(t.image_pathsets))
        t.db("Texts : " , sum(1 for patch in t.image_pathsets if patch[1] is not None))
        t.db("Captions : " , sum(1 for patch in t.image_pathsets if patch[2] is not None))
        # --- End Debug counts ---
        return # Done for non-Multi mode

    # --- Logic specific to Multi-ADDifT ---
    print("Processing for Multi-ADDifT mode...")

    # --- Check 2: Validate diff_target_name ---
    if not hasattr(t, 'diff_target_name') or not t.diff_target_name or not isinstance(t.diff_target_name, str):
         print("Error: 'diff_target_name' (suffix for target images) is missing or invalid for Multi-ADDifT mode.")
         return # Stop processing, pairing is impossible
    # --- End Check 2 ---

    pairpathsets = []
    found_pairs_count = 0
    processed_targets = set() # Keep track of targets already added as part of a pair

    # Iterate through the found image paths (keys of pathdict)
    for image_path in pathdict:
        original_info = pathdict[image_path] # [img_path, txt_path, cap_path, tags]
        base_name, ext = os.path.splitext(image_path)

        # Construct the expected target image path
        diff_target_path = f"{base_name}{t.diff_target_name}{ext}"

        # Check if this constructed target path exists in our dictionary of found files
        if diff_target_path in pathdict:
            target_info = pathdict[diff_target_path] # [target_img_path, target_txt_path, etc.]

            # Ensure we haven't already processed this pair starting from the target
            if diff_target_path in processed_targets:
                continue

            try:
                # Get image sizes to determine minimum common size for resizing later
                with Image.open(original_info[0]) as orig_img:
                    orig_size = orig_img.size
                with Image.open(target_info[0]) as targ_img: # Use target_info[0] which is the target image path
                    targ_size = targ_img.size

                # Determine the size to resize both images to (minimum dimensions)
                pair_size = (min(orig_size[0], targ_size[0]), min(orig_size[1], targ_size[1]))

                # Add the original image entry with pairing info
                # Format: [img_path, txt_path, cap_path, tags, pair_size, target_img_path]
                pairpathsets.append(original_info + [pair_size, target_info[0]])

                # Add the target image entry, marking it has no further target (for dataset loading)
                # Format: [target_img_path, target_txt_path, target_cap_path, target_tags, pair_size, None]
                pairpathsets.append(target_info + [pair_size, None])

                processed_targets.add(diff_target_path) # Mark target as processed
                found_pairs_count += 1

            except FileNotFoundError:
                print(f"Warning: File not found when trying to open images for pairing: {original_info[0]} or {target_info[0]}")
            except Exception as e:
                print(f"Error processing pair {original_info[0]} / {target_info[0]}: {e}")
        # else:
            # No target found for this image_path, it might be a target itself (already handled by processed_targets)
            # or an original without a pair. In Multi-ADDifT, we often only care about pairs.
            # print(f"Debug: No target '{diff_target_path}' found for '{image_path}'") # Uncomment for verbose debugging

    t.image_pathsets = pairpathsets # Update with the processed pairs
    print(f"Multi-ADDifT: Found {found_pairs_count} pairs based on suffix '{t.diff_target_name}'. Total entries: {len(t.image_pathsets)} (original + target).")

    # Optional: Add debug counts for Multi mode if needed
    t.db("Paired Entries : ", len(t.image_pathsets))
    t.db("Texts : " , sum(1 for patch in t.image_pathsets if patch[1] is not None))
    t.db("Captions : " , sum(1 for patch in t.image_pathsets if patch[2] is not None))


import os
from PIL import Image


def load_resize_image_and_text(t):
    for img_path, txt_path, cap_path, filename, pair_size, targ_path in t.image_pathsets:
        if os.path.basename(img_path).startswith('.'):
            continue
        image = Image.open(img_path)
        usealpha = image.mode == "RGBA"

        if pair_size is not None and image.size != pair_size:
            image = image.resize(pair_size, Image.LANCZOS)

        #max sizes
        ratio = image.width / image.height
        ar_errors = t.image_max_ratios - ratio
        indice = np.argmin(np.abs(ar_errors))  # 一番近いアスペクト比のインデックス
        max = t.image_max_buckets_sizes[indice]
        ar_error = ar_errors[indice]

        def resize_and_crop(ar_error, image, bucket_width, bucket_height, disable_upscale):
            if (ar_error > 0 and image.width < bucket_width or
                ar_error <= 0 and image.height < bucket_height) and disable_upscale:
                return None, None

            if ar_error <= 0:  # 幅＜高さなら高さを合わせる
                temp_width = int(image.width*bucket_height/image.height)
                image = image.resize((temp_width, bucket_height))  # アスペクト比を変えずに高さだけbucketに合わせる
                left = (temp_width - bucket_width) / 2  # 切り取り境界左側
                right = bucket_width + left  # 切り取り境界右側
                image = image.crop((left, 0, right, bucket_height))  # 左右切り取り
            else:  # 幅高さを逆にしたもの
                temp_height = int(image.height*bucket_width/image.width)
                image = image.resize((bucket_width, temp_height))
                upper = (temp_height - bucket_height) / 2
                lower = bucket_height + upper
                image = image.crop((0, upper, bucket_width, lower))

            if usealpha:
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                # アルファチャンネルの抽出
                alpha_channel = tensor[3]
                # 透明な部分を0、透明でない部分を1に設定
                alpha_mask = (alpha_channel > 0.1).float()
                # マスクのサイズを取得
                H, W = alpha_mask.shape
                # 新しいサイズを計算（縦横8分の1）
                new_H, new_W = H // 8, W // 8
                # マスクを縦横8分の1にリサイズ
                mask = F.interpolate(alpha_mask.unsqueeze(0).unsqueeze(0), size=(new_H, new_W), mode='nearest')
                mask = torch.cat([mask]*4, dim=1)
            else:
                # アルファチャンネルがない場合の画像サイズの取得
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                _, H, W = tensor.shape  # アルファチャンネルがない場合のためにRGBチャンネルを無視
                new_H, new_W = H // 8, W // 8  # 新しいサイズを計算（縦横8分の1）
                # すべて1のテンソルを作成
                mask = torch.ones((1, 4, new_H, new_W))

            image = image.convert("RGB")

            return image, mask

        resized, alpha_mask = resize_and_crop(ar_error, image, *max, t.image_disable_upscale)
        if resized is not None:
            t.image_buckets_raw[max].append([resized, alpha_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path, targ_path])
            if t.image_mirroring:
                flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_mask = torch.flip(alpha_mask, [1]) if alpha_mask is not None else None
                t.image_buckets_raw[max].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path+"m", targ_path+"m" if targ_path is not None else targ_path])

        ar_errors = t.image_sub_ratios - ratio

        try:
            for _ in range(t.sub_image_num):
                indice = np.argmin(np.abs(ar_errors))  # 一番近いアスペクト比のインデックス
                sub = t.image_sub_buckets_sizes[indice]
                ar_error = ar_errors[indice]
                resized, alpha_mask  = resize_and_crop(ar_error, image, *sub, t.image_disable_upscale)
                if resized is not None:
                    t.image_buckets_raw[sub].append([resized, alpha_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path, targ_path])
                    if t.image_mirroring:
                        flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_mask = torch.flip(alpha_mask, [1]) if alpha_mask is not None else None
                        t.image_buckets_raw[sub].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename, img_path+"m", targ_path+"m" if targ_path is not None else targ_path])

                ar_errors[indice] = ar_errors[indice] + 1
        except:
            print("Failed to make sub-buckets; image bucket step or image minimum length is too big?")

    for key in t.image_buckets_raw:
        print(f"bucket {key} has {len(t.image_buckets_raw[key])} images")
        t.total_images += len(t.image_buckets_raw[key])

def load_text_files(file_path):
    if file_path is None:
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def encode_image_text(t):
    with torch.no_grad(), t.a.autocast():
        emp1, emp2 = t.text_model.encode_text(t.lora_trigger_word)
        bar = tqdm(total = t.total_images)
        for key in t.image_buckets_raw:
            pairdict = {}
            for image, mask, text, caption, filename, img_path, targ_path in t.image_buckets_raw[key]:
                latent = t.image2latent(t,image)
                if t.image_use_filename_as_tag:
                    prompt = t.lora_trigger_word + "," + filename
                elif text is not None:
                    prompt = t.lora_trigger_word + ", " + text
                elif caption is not None:
                    prompt = t.lora_trigger_word + ", " + caption
                else:
                    prompt = t.lora_trigger_word
                t.tagcount(prompt)
                if "BASE" not in t.network_blocks:
                    emb1, emb2 = (emp1, emp2) if prompt is None else t.text_model.encode_text(prompt)
                else:
                    emb1 = emb2 = prompt
                t.image_buckets[key].append([latent, mask, emb1, emb2])
                bar.update(1)
                pairdict[img_path] = [latent, mask, emb1, emb2, targ_path, image]

            if t.mode == "Multi-ADDifT":
                t.image_buckets[key] = []
                for img_path_key in pairdict:
                    if pairdict[img_path_key][4] in pairdict:
                        image_o = pairdict[img_path_key][5]
                        image_t = pairdict[pairdict[img_path_key][4]][5]

                        image_np = np.array(image_o, dtype=np.int16)
                        image_t_np = np.array(image_t, dtype=np.int16)

                        mask = image_np - image_t_np
                        mask = torch.tensor(mask, dtype=torch.float32)
                        mask = mask.abs().sum(dim=-1)
                        mask = torch.where(mask > 10, torch.tensor(1, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))

                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent.shape[2], latent.shape[3]), mode='nearest')
                        #save_image1(t, mask.squeeze(0) * 255, "mask")

                        mask = torch.cat([mask] * 4, dim=1)
                        t.image_buckets[key].append((pairdict[img_path_key][:-2], pairdict[pairdict[img_path_key][4]][:-2]))

def save_images(t,key,images):
    if not images: return
    path = os.path.join(t.lora_data_directory,"x".join(map(str, list(key))))
    os.makedirs(path, exist_ok=True)
    for i, image in enumerate(images):
        ipath = os.path.join(path, f"{i}.jpg")
        image[0].save(ipath)


def save_image1(t, image, dirname=""):
    path = os.path.join(t.lora_data_directory, dirname) if dirname else t.lora_data_directory
    os.makedirs(path, exist_ok=True)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)

        if image.ndim == 2:  # 2D配列ならグレースケール画像
            image = Image.fromarray(image.astype(np.uint8), mode='L')
        elif image.ndim == 3:
            if image.shape[0] in [3, 4]:  # (C, H, W) 形式なら (H, W, C) に変換
                image = np.moveaxis(image, 0, -1)
            image = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    try:
        image_path = os.path.join(path, "saved_image.png")
        image.save(image_path)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")
