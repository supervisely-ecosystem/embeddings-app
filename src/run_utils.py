import os
import json
from collections import defaultdict
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import torch
import sklearn.manifold
import sklearn.cluster
import sklearn.decomposition
import umap
import re


def calculate_projections(embeddings, all_info_list, projection_method, metric="euclidean", umap_min_dist=0.05):
    try:
        if projection_method == "PCA":
            decomp = sklearn.decomposition.PCA(2)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "UMAP":
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "PCA-UMAP":
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        elif projection_method == "t-SNE":
            decomp = sklearn.manifold.TSNE(2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "PCA-t-SNE":
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = sklearn.manifold.TSNE(2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1)
            projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"unexpexted projection_method {projection_method}")
    except Exception as e:
        print(e)
        raise RuntimeError(f"count of embeddings = {len(embeddings)}, not enough to calculate projections.")
    return projections


def upload_embeddings(embeddings, info_updated, cfg, api, path_prefix, save_paths, team_id):
    os.makedirs(path_prefix, exist_ok=True)
    save_paths = {k: save_paths[k] for k in ["info", "cfg", "embeddings"]}
    with open(save_paths["info"], "w") as f:
        json.dump(info_updated, f)
    with open(save_paths["cfg"], "w") as f:
        json.dump(cfg, f)
    torch.save(embeddings, save_paths["embeddings"])
    remote_paths = [f"/{p}" for p in list(save_paths.values())]
    api.file.upload_bulk(team_id, list(save_paths.values()), remote_paths)


def download_embeddings(api, path_prefix, save_paths, team_id):
    os.makedirs(path_prefix, exist_ok=True)
    api.file.download(team_id, "/" + save_paths["info"], save_paths["info"])
    api.file.download(team_id, "/" + save_paths["embeddings"], save_paths["embeddings"])
    api.file.download(team_id, "/" + save_paths["cfg"], save_paths["cfg"])
    with open(save_paths["info"], "r") as f:
        all_info = json.load(f)
    with open(save_paths["cfg"], "r") as f:
        cfg = json.load(f)
    embeddings = torch.load(save_paths["embeddings"], weights_only=False)
    return embeddings, all_info, cfg


def make_series(projections, all_info_list, project_meta):
    x = projections[:, 1].tolist()
    y = projections[:, 0].tolist()

    series = defaultdict(list)
    global_idxs_mapping = defaultdict(list)
    for i in range(len(all_info_list)):
        obj_cls = str(all_info_list[i]["object_cls"] or "Image")
        series[obj_cls].append({"x": x[i], "y": y[i]})
        global_idxs_mapping[obj_cls].append(i)

    series = [{"name": k, "data": v} for k, v in series.items()]
    obj2color = {x.name: sly.color.rgb2hex(x.color) for x in project_meta.obj_classes}
    obj2color["Image"] = "#222222"
    colors = [obj2color[s["name"]] for s in series]

    return series, colors, global_idxs_mapping


def normalize_string(s):
    return re.sub("[^A-Z0-9_()-]", "", s, flags=re.IGNORECASE)


def get_save_paths(model_name, project, projection_method=None, metric=None):
    save_name = model_name.replace("/", "_")
    path_prefix = f"embeddings/{normalize_string(project.name)}_{project.id}/{save_name}"
    save_paths = {
        "info": f"{path_prefix}/info.json",
        "cfg": f"{path_prefix}/cfg.json",
        "embeddings": f"{path_prefix}/embeddings.pt",
        "projections": f"{path_prefix}/projections_{projection_method}_{metric}.pt",
    }
    return path_prefix, save_paths


def get_calculated_models_for_project(files_list, project_info):
    """Finds for what models we have calculated embeddings for the project"""
    model_name2save_path = {}
    for x in files_list:
        x = x["path"]
        s = x.split("/")
        if s[-1] == "embeddings.pt" and s[-3] == f"{normalize_string(project_info.name)}_{project_info.id}":
            model_name2save_path[s[-2]] = "/".join(s[:-1])
    return model_name2save_path


def concat_table(rows1, rows2):
    rows = list(zip(*zip(*rows1), rows2))
    return rows


def get_rows(files_list, model_items, project_info):
    model_name2save_path = get_calculated_models_for_project(files_list, project_info)
    already_calculated = [item[0].replace("/", "_") in model_name2save_path for item in model_items]
    bool2str = {True: "✔", False: "✖"}
    already_calculated = [bool2str[x] for x in already_calculated]
    rows = concat_table(model_items, already_calculated)
    return rows


def get_devices():
    cuda_names = [f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(torch.cuda.device_count())]
    cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    device_names = cuda_names + ["cpu"]
    torch_devices = cuda_devices + ["cpu"]
    return device_names, torch_devices
