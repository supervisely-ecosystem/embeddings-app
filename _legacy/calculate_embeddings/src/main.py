import supervisely as sly
from supervisely.sly_logger import logger
from dotenv import load_dotenv
import json
import os
import cv2
import numpy as np
import torch
from collections import defaultdict

import infer_utils


print(f"{torch.__version__}, {torch.version.cuda}, {torch.cuda.is_available()}")


def get_crops(img_np, yxyx_coords, hw_expand=(0, 0)):
    # crop image to sub_images by yxyx coords
    crops = []
    h, w = hw_expand
    for yxyx in yxyx_coords:
        y1, x1, y2, x2 = yxyx
        y1, y2 = y1 - h, y2 + h
        x1, x2 = x1 - w, x2 + w
        y1, x1, y2, x2 = [max(0, coord) for coord in (y1, x1, y2, x2)]  # clip coords
        crop = img_np[y1:y2, x1:x2, :]
        crops.append(crop)
    return crops


def extract_crops(
    img_np,
    labels,
    input_size_hw,
    instance_mode="image",
    hw_expand=(0, 0),
    resize_interpolation=cv2.INTER_LINEAR,
):
    # make all needed crops and resizes for image-annotation pair
    # return.__len__ may lay in [0, K+1], where K == n_instances.
    # return.__len__ can be 0 if mode is 'instance' and no any instances on image.
    assert instance_mode in [
        "image",
        "instance",
        "both",
    ], f"unexpected instance_mode {instance_mode}"
    accepted_geometry = (sly.Rectangle, sly.Bitmap, sly.Polygon, sly.Polyline)
    result_crops = []
    result_obj_cls = []
    result_yxyx = []
    if instance_mode in ["image", "both"]:
        img = cv2.resize(img_np, input_size_hw[::-1], interpolation=resize_interpolation)
        result_crops.append(img)
        result_yxyx.append([0, 0, img_np.shape[0], img_np.shape[1]])
        result_obj_cls.append(None)
    if instance_mode in ["instance", "both"]:
        yxyx_croods = []
        for label in labels:
            if not isinstance(label.geometry, accepted_geometry):
                continue
            if label.geometry.area < 9:
                continue
            rect = label.geometry.to_bbox()
            yxyx_croods.append([rect.top, rect.left, rect.bottom, rect.right])
            result_obj_cls.append(label.obj_class.name)
        crops = get_crops(img_np, yxyx_croods, hw_expand=hw_expand)
        crops = [
            cv2.resize(crop, input_size_hw[::-1], interpolation=resize_interpolation)
            for crop in crops
        ]
        result_yxyx += yxyx_croods
        result_crops += crops
    assert len(result_crops) == len(result_obj_cls) == len(result_yxyx)
    return result_crops, result_obj_cls, result_yxyx


def form_batches(crops, batch_size):
    idxs_split = list(range(0, len(crops), batch_size)) + [None]  # e.g: [0,5,10,15,None]
    crops_batched = []
    for i in range(len(idxs_split) - 1):
        batch = crops[idxs_split[i] : idxs_split[i + 1]]
        batch = np.stack(batch)
        crops_batched.append(batch)
    return crops_batched


def normalize(img_batch, mean, std, np_dtype=np.float32):
    # img_batch: [B,H,W,C]
    assert img_batch.shape[3] == 3
    mean = np.array(mean, dtype=np_dtype)
    std = np.array(std, dtype=np_dtype)
    return (img_batch - mean) / std


if __name__ == "__main__":

    # example_model_names = [
    #     "maxvit_large_tf_384.in21k_ft_in1k",
    #     "facebook/convnext-xlarge-224-22k",
    #     "beitv2_large_patch16_224",
    #     "beitv2_large_patch16_224_in22k",
    #     "openai/clip-vit-base-patch32",
    #     "openai/clip-vit-large-patch14",
    #     "facebook/flava-full",
    #     "facebook/convnext-large-384",
    #     "microsoft/beit-large-patch16-224-pt22k",
    #     "microsoft/beit-large-patch16-384",
    # ]

    model_name = "facebook/convnext-tiny-224"
    instance_mode = "both"
    batch_size = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size_api = 50
    instance_rect_expand = [0, 0]
    np_dtype = np.float32

    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api()

    project_id = sly.env.project_id(raise_not_found=False)
    dataset_id = sly.env.dataset_id(raise_not_found=False)
    team_id = sly.env.team_id(raise_not_found=False)
    model_name = os.environ["modal.state.model_name"]
    batch_size = int(os.environ["modal.state.batch_size"])

    save_name = model_name.replace("/", "_")

    assert (project_id is not None) or (
        dataset_id is not None
    ), "either project_id or dataset_id must be set in local.env"

    if project_id is not None:
        datasets = api.dataset.get_list(project_id)
    else:
        datasets = [api.dataset.get_info_by_id(dataset_id)]
        project_id = datasets[0].project_id
    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    if team_id is None:
        team_id = api.team.get_list()[0].id

    # init model
    print(f"Running model on {device}")
    model, cfg, format_input = infer_utils.create_model(model_name)
    model.to(device)
    input_size_hw = cfg["input_size"]
    resize_interpolation = cfg["interpolation"]

    # load embeddings if exists
    path_prefix = f"embeddings/{project_id}"
    save_paths = {
        "info": f"{path_prefix}/{save_name}_info.json",
        "cfg": f"{path_prefix}/{save_name}_cfg.json",
        "embeddings": f"{path_prefix}/{save_name}_embeddings.pt",
    }

    os.makedirs(path_prefix, exist_ok=True)
    if api.file.exists(team_id, "/" + save_paths["info"]):
        api.file.download(team_id, "/" + save_paths["info"], save_paths["info"])
        api.file.download(team_id, "/" + save_paths["embeddings"], save_paths["embeddings"])
        with open(save_paths["info"], "r") as f:
            info_old = json.load(f)
        embeddings = torch.load(save_paths["embeddings"], weights_only=False)
        print("embeddings loaded. n =", len(embeddings))
    else:
        info_old = {
            "dataset_id": [],
            "image_id": [],
            "object_id": [],
            "object_cls": [],
            "crop_yxyx": [],
            "updated_at": [],
        }
        embeddings = None

    def infer_one(image, labels, instance_mode):
        crops, crops_obj_cls, crops_yxyx = extract_crops(
            image,
            labels,
            input_size_hw=input_size_hw,
            instance_mode=instance_mode,
            hw_expand=instance_rect_expand,
            resize_interpolation=resize_interpolation,
        )
        if len(crops) == 0:
            return None
        crops_batched = form_batches(crops, batch_size=batch_size)
        features = []
        # infer model
        for img_batch in crops_batched:
            # 1. prepare input
            img_batch = img_batch.astype(np.float32) / 255
            img_batch = normalize(img_batch, cfg["mean"], cfg["std"], np_dtype=np_dtype)
            img_batch = img_batch.transpose(0, 3, 1, 2)
            inputs = format_input(torch.tensor(img_batch))
            # 2. run infer
            features_batch = infer_utils.get_features(model, inputs, pool_mode="auto")
            features.append(features_batch.cpu().numpy())
        return np.concatenate(features), crops_obj_cls, crops_yxyx

    # convert info to list [{},{},{},...]
    info_old_list = [
        dict(tuple(zip(info_old.keys(), vals))) for vals in zip(*list(info_old.values()))
    ]
    img_id2idxs = defaultdict(list)
    img_id2upd = defaultdict(list)
    for i in range(len(info_old_list)):
        info = info_old_list[i]
        img_id = info["image_id"]
        img_id2idxs[img_id].append(i)
        img_id2upd[img_id].append(info["updated_at"])

    to_del_img_ids = []
    to_add_info_list = []
    to_add_embeds = []

    all_dataset_img_ids = []
    for dataset in datasets:
        to_infer_img_ids = []
        all_image_info = api.image.get_list(dataset.id)
        img_id2info = {img_info.id: img_info for img_info in all_image_info}
        all_image_ids = [img.id for img in all_image_info]
        all_dataset_img_ids += all_image_ids
        for img_id, img_info in zip(all_image_ids, all_image_info):
            upd_at = img_id2upd.get(img_id)
            if upd_at is None:
                # new image found
                to_infer_img_ids.append(img_id)
            elif upd_at[0] != img_info.updated_at:
                # image updated
                to_del_img_ids.append(img_id)
                to_infer_img_ids.append(img_id)

        # Infer and collect info
        progress = sly.Progress(
            f"Infer dataset {dataset.name}", len(to_infer_img_ids), ext_logger=logger
        )
        for image_ids in sly.batched(to_infer_img_ids, batch_size=batch_size_api):
            images = api.image.download_nps(dataset.id, image_ids)
            anns_json = api.annotation.download_json_batch(dataset.id, image_ids)
            annotations = [sly.Annotation.from_json(ann, project_meta) for ann in anns_json]
            assert len(images) == len(annotations)
            for img_id, image, ann in zip(image_ids, images, annotations):
                new_embeds, crops_obj_cls, crops_yxyx = infer_one(
                    image, ann.labels, instance_mode="both"
                )
                if new_embeds is None:
                    continue
                to_add_embeds.append(new_embeds)

                # add info for image
                upd = img_id2info[img_id].updated_at
                info = {
                    "dataset_id": dataset.id,
                    "image_id": img_id,
                    "object_id": None,
                    "object_cls": None,
                    "crop_yxyx": crops_yxyx[0],
                    "updated_at": upd,
                }
                to_add_info_list.append(info)
                # add infos for crops
                for obj_cls, yxyx, label in zip(crops_obj_cls[1:], crops_yxyx[1:], ann.labels):
                    upd = label.geometry.updated_at
                    object_id = label.geometry.sly_id
                    info = {
                        "dataset_id": dataset.id,
                        "image_id": img_id,
                        "object_id": object_id,
                        "object_cls": obj_cls,
                        "crop_yxyx": yxyx,
                        "updated_at": upd,
                    }
                    to_add_info_list.append(info)
                progress.iter_done()

    # to remove imgs
    to_del_img_ids = list(set(img_id2upd) - set(all_dataset_img_ids))

    print("to_del:", len(to_del_img_ids))
    print("to_add:", len(to_add_info_list))
    if len(to_del_img_ids) == 0 and len(to_add_info_list) == 0:
        print("All embeddings are up to date!")
        exit()

    info_updated = info_old_list
    # 1. remove
    if to_del_img_ids:
        mask = [info["image_id"] not in to_del_img_ids for idx, info in enumerate(info_old_list)]
        mask = np.array(mask)
        embeddings = embeddings[mask]
        info_updated = [info_old_list[idx] for idx in mask.nonzero()[0]]
    # 2. append
    if to_add_embeds:
        to_add_embeds = np.concatenate(to_add_embeds)
        if embeddings is None:
            embeddings = to_add_embeds
        else:
            embeddings = np.concatenate([embeddings, to_add_embeds])
        info_updated += to_add_info_list
    assert len(info_updated) == len(embeddings)
    del info_old_list

    # convert list to dict
    info_updated_dict = defaultdict(list)
    for d in info_updated:
        for k, v in d.items():
            info_updated_dict[k].append(v)
    info_updated = info_updated_dict

    # Save
    with open(save_paths["info"], "w") as f:
        json.dump(info_updated, f)
    with open(save_paths["cfg"], "w") as f:
        json.dump(cfg, f)
    torch.save(embeddings, save_paths["embeddings"])
    print("uploading to team_files...")
    remote_paths = [f"/{p}" for p in list(save_paths.values())]
    api.file.upload_bulk(team_id, list(save_paths.values()), remote_paths)

    print("result shape:", embeddings.shape)
    print("done")
