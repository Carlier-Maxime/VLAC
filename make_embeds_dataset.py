import argparse
import os

import torch
import webdataset as wds
from tqdm import tqdm

from vlac import VLAC
from vlac.dataset.config import COYO_LENGTH
from vlac.dataset.dataset import COYOWebDatasetIterable
from vlac.utils.args import add_multiprocess_args, check_multiprocess_args


def get_args():
    parser = argparse.ArgumentParser(description="Make a Embeds Dataset for train encoder / decoder.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--part_len", type=int, default=1000)
    parser.add_argument("--digits_of_id", type=int, default=9)
    parser = add_multiprocess_args(parser)
    args = parser.parse_args()
    return check_multiprocess_args(parser, args, COYO_LENGTH // args.part_len)


def open_tar(start_id, args):
    return wds.TarWriter(os.path.join(args.output_dir, f"{start_id:0{args.digits_of_id}d}_{start_id+args.part_len:0{args.digits_of_id}d}.tar"), encoder=True)


def pre_save(tensor):
    return tensor.cpu().detach().clone().contiguous()


def check_tar_open(args, tar: wds.TarWriter | None, count: int, end_id: int = 0):
    start_id = (count // args.part_len) * args.part_len
    if count >= end_id or tar is None:
        if tar is not None: tar.close()
        tar = open_tar(start_id, args)
        end_id = start_id + args.part_len
    return tar, end_id


@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vlac = VLAC.from_pretrained(args.model).to(args.device)
    count = 0
    tar = None
    end_id = 0
    for data in tqdm(COYOWebDatasetIterable(vlac.vision_tower.image_processor, vlac.text_tokenizer, batch_size=args.batch_size), desc='Make Embeds', unit='batch'):
        batch_size = len(data["vision"])
        if args.procid is not None and args.procid != ((count+batch_size-1)//args.part_len) % args.ntasks and args.procid != (count//args.part_len) % args.ntasks:
            count += batch_size
            continue
        text_tokens, vision = data["text_tokens"], data["vision"].to(args.device)
        input_ids = text_tokens["input_ids"].to(args.device)
        attention_mask = text_tokens["attention_mask"].to(args.device)
        _, _, attention_mask, _, multimodal_embeds, multimodal_tokens = vlac.vlm.prepare_embeds_for_multimodal(input_ids, None, attention_mask, None, input_ids, vision)
        ids = torch.arange(attention_mask.shape[-1], device=args.device).unsqueeze(0).expand(attention_mask.shape[0], -1)
        limits = torch.where(attention_mask.bool(), ids, 0).add_(1).max(dim=1)[0].cpu().tolist()
        attention_mask = attention_mask.cpu().split(1)
        multimodal_embeds = multimodal_embeds.cpu().split(1)
        multimodal_tokens = multimodal_tokens.cpu().split(1)
        for mask, embeds, tokens, limit in tqdm(zip(attention_mask, multimodal_embeds, multimodal_tokens, limits), desc='Save Batch', unit='sample', total=batch_size, disable=batch_size <= 128, leave=False):
            if mask.sum() == 0:
                continue
            self_work = args.procid == (count // args.part_len) % args.ntasks
            if args.procid is not None and not self_work:
                count += 1
                continue
            tar, end_id = check_tar_open(args, tar, count, end_id=end_id)
            tar.write({
                "__key__": str(count),
                "mask.pth": pre_save(mask[:limit]),
                "embeds.pth": pre_save(embeds[:limit].clone()),
                "tokens.pth": pre_save(tokens[:limit].clone())
            })
            count += 1
    tar.close()


if __name__ == "__main__":
    main()
