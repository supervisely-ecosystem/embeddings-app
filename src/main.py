import os
import json
from collections import defaultdict
import numpy as np

import supervisely as sly
from supervisely.app.content import StateJson, DataJson
from dotenv import load_dotenv
import torch
from typing import List, Union, Tuple
from supervisely.app.widgets import (
    Container,
    Card,
    LabeledImage,
    Text,
    RadioTable,
    Select,
    SelectString,
    InputNumber,
    BindedInputNumber,
    Input,
    Checkbox,
    Button,
    Field,
    Progress,
    IFrame,
    Bokeh,
    SelectDatasetTree,
    NotificationBox,
    Empty,
    Flexbox,
)

from . import run_utils
from . import calculate_embeddings


def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, team_id, project_info, project_meta, is_marked, tag_meta, issue_tag_meta
    dataset_ids = new_dataset_ids
    if dataset_ids:
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        sly.logger.info(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        sly.logger.info("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, team_id, project_info, project_meta = [None] * 5
    if dataset_ids or project_id:
        is_marked = False
        tag_meta = project_meta.get_tag_meta(tag_name)
        sly.logger.info(f"tag_meta exists: {bool(tag_meta)}")
        issue_tag_meta = project_meta.get_tag_meta(issue_tag_name)
        sly.logger.info(f"issue_tag_meta exists: {bool(issue_tag_meta)}")


### Globals init
available_projection_methods = ["UMAP", "PCA", "t-SNE", "PCA-UMAP", "PCA-t-SNE"]
tag_name = "MARKED"
issue_tag_name = "ISSUE"
instance_mode = None

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)


### Dataset selection
dataset_selector = SelectDatasetTree(project_id=project_id, multiselect=True, select_all_datasets=True)
card_project_settings = Card(title="Dataset selection", content=dataset_selector)

### Model selection
model_items = [
    ["facebook/metaclip-b16-fullcc2.5b", "599 MB", "Transformer"],
    ["openai/clip-vit-base-patch32", "605 MB", "Transformer"],
    ["openai/clip-vit-large-patch14", "1710 MB", "Transformer"],
    ["facebook/convnext-tiny-224", "114 MB", "ConvNet"],
    ["facebook/convnext-large-384", "791 MB", "ConvNet"],
    ["facebook/convnext-xlarge-224-22k", "1570 MB", "ConvNet"],
    ["facebook/flava-full", "1430 MB", "Transformer"],
    ["microsoft/beit-large-patch16-224-pt22k", "1250 MB", "Transformer"],
    ["microsoft/beit-large-patch16-384", "1280 MB", "Transformer"],
    ["beitv2_base_patch16_224.in1k_ft_in22k", "414 MB", "Transformer"],
    ["beitv2_large_patch16_224.in1k_ft_in22k_in1k", "1310 MB", "Transformer"],
    # ["maxvit_large_tf_384.in21k_ft_in1k", "849 MB", "ConvNet+Transformer"],  # now it is at pre-release in timm lib
]
files_list = api.file.list(team_id, "/embeddings")
rows = run_utils.get_rows(files_list, model_items, project_info)
column_names = ["Name", "Model size", "Architecture type", "Already calculated"]
table_model_select = RadioTable(column_names, rows)
table_model_select_f = Field(table_model_select, "Click on the table to select a model:")
input_select_model = Input("", placeholder="facebook/metaclip-l14-fullcc2.5b")
desc_select_model = Text(
    "Alternatively, you can paste model name from <a href='https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads' target='_blank'>hugging face</a> (some models may not work)."
)
device_names, torch_devices = run_utils.get_devices()
select_device = Select([Select.Item(v, l) for v, l in zip(torch_devices, device_names)])
select_device_f = Field(select_device, "Device")
input_batch_size = InputNumber(4, 1, 1024)
input_batch_size_f = Field(
    input_batch_size,
    "Batch size",
)
content = Container(
    [
        table_model_select_f,
        desc_select_model,
        input_select_model,
        select_device_f,
        input_batch_size_f,
    ]
)
card_model_selection = Card(title="Model selection", content=content)


### Preprocessing settings
select_instance_mode = SelectString(
    [
        "images",
        "objects",
        "both",
    ]
)
select_instance_mode_f = Field(
    select_instance_mode,
    "Instance mode",
    "Whether to run for images or for cropped objects in the images or both",
)
input_expand_wh = InputNumber(0, -10000, 10000)
input_expand_wh_f = Field(
    input_expand_wh,
    "Expand crops (px)",
    "Expand bounding boxes by the given number of pixels on all sides (both X and Y axes). This helps provide the model with some context around the boundaries of the objects.",
)
content = Container([select_instance_mode_f, input_expand_wh_f])
card_preprocessing_settings = Card(title="Preprocessing settings", content=content, collapsable=True)
# card_preprocessing_settings.collapse()


### Visualizer settings
select_projection_method = SelectString(available_projection_methods)
select_projection_method_f = Field(
    select_projection_method,
    "Projection method",
    "A decomposition method: how to project the high-dimensional embeddings onto 2D space for visualization.",
)
select_metric = SelectString(["euclidean", "cosine"])
select_metric_f = Field(select_metric, "Metric", "The parameter for projection method")
content = Container([select_projection_method_f, select_metric_f])
card_visualizer_settings = Card(title="Decomposition settings", content=content, collapsable=True)
card_visualizer_settings.collapse()


### Run section
btn_run = Button("Run")
check_force_recalculate = Checkbox("Force recalculate")
progress = Progress()
info_run = NotificationBox()
content = Container([btn_run, check_force_recalculate, progress, info_run])
card_run = Card(title="Run", content=content)

### Embeddings Chart Settings
dot_size_btn = Button("Change dots size", button_size="small", plain=True)
dot_size_num = InputNumber(min=0.01, value=0.05, step=0.01)
dot_size = Container([dot_size_num, dot_size_btn])

chart_size_input = BindedInputNumber(height=500, width=1000, min=300)
chart_size_btn = Button("Change chart size", button_size="small", plain=True)
chart_size = Container([chart_size_input, chart_size_btn])

zoom_to_figure = Checkbox("Zoom to object")
need_zoom = zoom_to_figure.is_checked()

chart_settings = Field(
    Container([dot_size, chart_size]),
    "Chart settings",
    "Change the size of the chart and the size of the dots on the chart",
)
chart_settings.hide()


### Embeddings Chart
bokeh = Bokeh(
    x_axis_visible=True,
    y_axis_visible=True,
    grid_visible=True,
    show_legend=True,
    legend_location="right",
    legend_click_policy="hide",
)
bokeh_iframe = IFrame()
card_chart = Card(content=bokeh_iframe, title="Embeddings chart", collapsable=True)
labeled_image = LabeledImage()
text = Text("no object selected")
show_all_anns = False
cur_info = None
btn_toggle = Button(f"Show all annotations: {show_all_anns}", "default", button_size="small")
btn_mark = Button(f"Assign tag 'MARKED'", button_size="small")
preview_widgets = Container([labeled_image, text, zoom_to_figure, btn_toggle, btn_mark])
preview_widgets.hide()


cur_infos = None
batch_text = Text()
issue_tag_text = Text()
add_issue_tag = Button(f"Asign 'ISSUE' tags", button_size="small", plain=True)
job_issue = Button(f"Create Labeling Job", button_size="small", plain=True)
batch_tagging_field = Field(
    Container([Flexbox([add_issue_tag, Empty(), job_issue]), issue_tag_text]),
    "Issues",
    "Assign 'ISSUE' tag to IMAGES or Create Labeling Job",
)
batch_tagging_cont = Container([batch_text, batch_tagging_field])
batch_tagging_cont.hide()

card_preview = Card(
    title="Preview card",
    content=Container(widgets=[preview_widgets, batch_tagging_cont, chart_settings]),
)
card_embeddings_chart = Container(widgets=[card_chart, card_preview], direction="horizontal", fractions=[3, 1])
card_embeddings_chart.hide()


app = sly.Application(
    layout=Container(
        widgets=[
            card_project_settings,
            card_model_selection,
            card_preprocessing_settings,
            card_visualizer_settings,
            card_run,
            card_embeddings_chart,
        ]
    )
)


@zoom_to_figure.value_changed
def on_zoom_change(value: bool):
    global need_zoom
    need_zoom = value

    if cur_info is not None and cur_info["object_id"] is not None:
        show_image(cur_info, project_meta)


@dot_size_btn.click
def change_dot_size():
    bokeh.update_point_size(dot_size_num.value)
    bokeh_iframe.set(bokeh.html_route_with_timestamp)


@chart_size_btn.click
def on_click():
    width, height = chart_size_input.get_value()
    bokeh.update_chart_size(width, height)
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height=f"{height + 50}px", width=f"{width + 50}px")


@btn_toggle.click
def toggle_ann():
    global show_all_anns
    show_all_anns = not show_all_anns
    btn_toggle.text = f"Show all annotations: {show_all_anns}"
    if cur_info:
        show_image(cur_info, project_meta)


@bokeh.value_changed
def on_click(selected_ids: List[List[int]]):
    global global_idxs_mapping, all_info_list, project_meta, is_marked, tag_meta, cur_infos

    issue_tag_text.text = ""
    batch_text.text = ""
    batch_tagging_cont.hide()
    preview_widgets.hide()

    if len(selected_ids) >= 1:
        curr_selected_ids = selected_ids[-1]
        maxlen_selected_ids = max(selected_ids, key=len)
        if len(curr_selected_ids) != len(maxlen_selected_ids):
            curr_selected_ids = maxlen_selected_ids
        selected_ids = curr_selected_ids

    selected_cnt = len(selected_ids)
    if selected_cnt == 1:
        batch_text.text = ""
        batch_tagging_cont.show()
        preview_widgets.show()
        info = all_info_list[selected_ids[0]]
        cur_infos = [info]
        if tag_meta is not None:
            tag = read_tag(info["image_id"], info["object_id"])
            is_marked = bool(tag)
            update_marked()
        show_image(info, project_meta)
        if btn_mark.is_hidden():
            btn_mark.show()
    elif selected_cnt > 1:
        batch_tagging_cont.show()
        cur_infos = [all_info_list[i] for i in selected_ids]
        obj_clss = list(set([info["object_cls"] for info in cur_infos]))
        is_objects = any([info["object_id"] is not None for info in cur_infos])
        is_images = any([info["object_id"] is None for info in cur_infos])
        both = is_objects and is_images

        t = f"{len(cur_infos)} "
        t += "items. " if both else "images. " if is_images else "objects. "
        t += f"Object classes: {str(obj_clss)}. "
        batch_text.set(t, "info")


@job_issue.click
def create_labeling_job():
    global cur_infos
    issue_tag_text.text = ""
    if cur_infos is not None:
        ds_id_to_img_ids = defaultdict(set)
        for info in cur_infos:
            ds_id_to_img_ids[info["dataset_id"]].add(info["image_id"])
        jobs = []
        for ds_id, img_ids in ds_id_to_img_ids.items():
            if len(img_ids) > 0:
                jobs.extend(
                    api.labeling_job.create(
                        f"Labeling job for {project_info.name} project embeddings",
                        ds_id,
                        [api.user.get_my_info().id],
                        images_ids=list(img_ids),
                        # include_images_with_tags=[issue_tag_name],
                    )
                )
        if len(jobs) > 0:
            ids = [job.id for job in jobs]
            issue_tag_text.set(f"Labeling jobs created IDs: {ids}", "success")
        else:
            issue_tag_text.set("No objects to create labeling job", "warning")


def update_marked():
    global is_marked
    if is_marked:
        btn_mark.text = "Remove tag 'MARKED'"
    else:
        btn_mark.text = "Assign tag 'MARKED'"


@add_issue_tag.click
def issue_tagging():
    global project_meta, cur_infos, issue_tag_meta
    if issue_tag_meta is None:
        sly.logger.debug("first marking, creating tag_meta")
        issue_tag_meta = sly.TagMeta(issue_tag_name, sly.TagValueType.NONE)
        project_meta, issue_tag_meta = get_or_create_tag_meta(project_id, issue_tag_meta)

    ds_ids_to_img_ids = defaultdict(set)
    for info in cur_infos:
        ds_ids_to_img_ids[info["dataset_id"]].add(info["image_id"])

    added = 0
    for ds_id, img_ids in ds_ids_to_img_ids.items():
        img_ids_to_mark = []
        img_ids = list(img_ids)
        for img_id, tag in zip(img_ids, read_img_tags(ds_id, img_ids, issue_tag_meta)):
            if tag is None:
                img_ids_to_mark.append(img_id)

        if len(img_ids_to_mark) > 0:
            add_img_tags(list(img_ids_to_mark), issue_tag_meta)
            added += len(img_ids_to_mark)

    if added > 0:
        issue_tag_text.set(f"Assigned 'ISSUE' tags: {added} images", "success")


@btn_mark.click
def on_mark():
    global project_info, project_meta, tag_meta, cur_info, is_marked
    if tag_meta is None:
        sly.logger.debug("first marking, creating tag_meta")
        tag_meta = sly.TagMeta(tag_name, sly.TagValueType.NONE)
        project_meta, tag_meta = get_or_create_tag_meta(project_id, tag_meta)
        is_marked = False
    img_id, obj_id = cur_info["image_id"], cur_info["object_id"]
    if is_marked:
        resp = remove_tag(img_id, obj_id)
    else:
        resp = add_tag(img_id, obj_id)
    tag = read_tag(img_id, obj_id)
    is_marked = bool(tag)
    update_marked()


def show_image(info, project_meta):
    global cur_info, show_all_anns, need_zoom
    cur_info = info
    image_id, obj_cls, obj_id = info["image_id"], info["object_cls"], info["object_id"]
    labeled_image.loading = True

    image = api.image.get_info_by_id(image_id)
    ann_json = api.annotation.download_json(image_id)
    if not show_all_anns:
        ann_json["objects"] = [obj for obj in ann_json["objects"] if obj["id"] == obj_id]
    ann = sly.Annotation.from_json(ann_json, project_meta) if len(ann_json["objects"]) else None

    if need_zoom and obj_id is not None:
        labeled_image.set(
            title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id, zoom_to=obj_id, zoom_factor=1.5
        )
    else:
        labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
    text.set("object class: " + str(obj_cls), "info")
    labeled_image.loading = False


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    update_globals(new_dataset_ids)
    update_table()
    update_marked()


def update_table():
    files_list = api.file.list(team_id, "/embeddings")
    rows = run_utils.get_rows(files_list, model_items, project_info)
    table_model_select.rows = rows


@btn_run.click
def run():
    global model_name, global_idxs_mapping, all_info_list, instance_mode

    selected_datasets = set()
    for dataset_id in dataset_selector.get_selected_ids():
        selected_datasets.add(dataset_id)
        for ds in api.dataset.get_nested(project_id=project_id, dataset_id=dataset_id):
            selected_datasets.add(ds.id)
    update_globals(list(selected_datasets))

    info_run.description = ""
    card_embeddings_chart.hide()
    btn_mark.hide()

    if not dataset_ids:
        info_run.description += "Dataset is not selected"
        return

    # 1. Read fields
    datasets = [api.dataset.get_info_by_id(i) for i in dataset_ids]
    if input_select_model.get_value():
        model_name = input_select_model.get_value()
    else:
        model_name = table_model_select.get_selected_row(StateJson())[0]
    instance_mode = str(select_instance_mode.get_value())
    expand_hw = [int(input_expand_wh.value)] * 2
    projection_method = str(select_projection_method.get_value())
    metric = str(select_metric.get_value())
    device = str(select_device.get_value())
    batch_size = int(input_batch_size.value)
    force_recalculate = bool(check_force_recalculate.is_checked())
    path_prefix, save_paths = run_utils.get_save_paths(model_name, project_info, projection_method, metric)

    # 2. Load embeddings if exist
    if api.file.exists(team_id, "/" + save_paths["info"]) and not force_recalculate:
        sly.logger.info("Found existing embeddings")
        info_run.description += "found existing embeddings<br>"
        embeddings, all_info, cfg = run_utils.download_embeddings(api, path_prefix, save_paths, team_id)
        if cfg is None:
            previous_instance_mode = "unknown"
        else:
            previous_instance_mode = cfg.get("instance_mode", None)
            if previous_instance_mode is None:
                previous_instance_mode = "unknown"
        if previous_instance_mode != instance_mode:
            info_run.description += "Instance mode was changed. Force embeddings recalculation.<br>"
            embeddings, all_info, cfg = None, None, None
            force_recalculate = True
            sly.logger.info(
                f"Force embeddings recalculation. Instance mode was changed from '{previous_instance_mode}' to '{instance_mode}'."
            )
        else:
            sly.logger.debug("embeddings downloaded. n =", len(embeddings))
    else:
        embeddings, all_info, cfg = None, None, None

    # 3. Calculate or update embeddings
    out = calculate_embeddings.calculate_embeddings_if_needed(
        api,
        model_name,
        datasets,
        device,
        batch_size,
        embeddings,
        all_info,
        cfg,
        instance_mode,
        expand_hw,
        project_meta,
        progress,
        info_run,
    )
    is_updated = out[-1]
    if is_updated:
        embeddings, all_info, cfg = out[:3]

    # 4. Save embeddings if it was updated
    is_updated = is_updated or force_recalculate
    if is_updated:
        if cfg is None:
            cfg = {"instance_mode": instance_mode}
        else:
            cfg["instance_mode"] = instance_mode

        sly.logger.info("uploading embeddings to team_files...")
        run_utils.upload_embeddings(embeddings, all_info, cfg, api, path_prefix, save_paths, team_id)

    # 5. Calculate projections or load from team_files
    all_info_list = [dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))]
    if api.file.exists(team_id, "/" + save_paths["projections"]) and not is_updated:
        info_run.description += "found existing projections<br>"
        sly.logger.info("downloading projections...")
        api.file.download(team_id, "/" + save_paths["projections"], save_paths["projections"])
        projections = torch.load(save_paths["projections"], weights_only=False)
    else:
        info_run.description += "Calculating projections...<br>"
        sly.logger.info("calculating projections...")
        if len(embeddings) <= 1:
            info_run.description += f"the count of embeddings (n={len(embeddings)}) must be > 1<br>"
            return
        try:
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        except RuntimeError:
            info_run.description += (
                f"the count of embeddings is {len(embeddings)}, not enough to use UMAP. Trying PCA instead...<br>"
            )
            projection_method = "PCA"
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        sly.logger.info("uploading projections to team_files...")
        torch.save(projections, save_paths["projections"])
        remote_path = f"/{save_paths['projections']}"
        api.file.upload(team_id, save_paths["projections"], remote_path)
    file_id = str(api.file.get_info_by_path(team_id, "/" + save_paths["embeddings"]).id)
    server_address = os.environ.get("SERVER_ADDRESS")
    if server_address:
        if sly.is_development():
            url = sly.utils.abs_url(f"files/{file_id}")
        else:
            url = f"/files/{file_id}"
        info_run.description += f"Embeddings were saved to Team Files: <a href={url}>{save_paths['embeddings']}</a><br>"

    # 6. Show chart
    obj_classes = list(set(all_info["object_cls"]))
    sly.logger.debug(f"n_classes = {len(obj_classes)}")
    series, pre_colors, global_idxs_mapping = run_utils.make_series(projections, all_info_list, project_meta)

    bokeh.clear()
    r = 0.05
    curr_id = 0
    for s, color in zip(series, pre_colors):
        data_source = {}
        data_source["x"] = [i["x"] for i in s["data"]]
        data_source["y"] = [i["y"] for i in s["data"]]
        data_source["radius"] = [r] * len(s["data"])
        data_source["colors"] = [color] * len(s["data"])
        data_source["ids"] = global_idxs_mapping[s["name"]]
        data_source["names"] = [s["name"]] * len(s["data"])
        curr_id += len(s["data"])
        bokeh.add_data(**data_source)
        bokeh.add_plot(Bokeh.Circle(name=s["name"]))

    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
    card_embeddings_chart.show()
    chart_settings.show()
    update_table()
    info_run.description += "Done!<br>"


def get_or_create_tag_meta(project_id, tag_meta):
    # params: project_id
    # updates: global project_meta, tag_meta
    project_meta_json = api.project.get_meta(id=project_id)
    project_meta = sly.ProjectMeta.from_json(data=project_meta_json)
    tag_names = [tag_meta.name for tag_meta in project_meta.tag_metas]
    if tag_meta.name not in tag_names:
        project_meta = project_meta.add_tag_meta(new_tag_meta=tag_meta)
        api.project.update_meta(id=project_id, meta=project_meta)
    tag_meta = get_tag_meta(project_id, name=tag_meta.name)  # we need to re-assign tag_meta
    return project_meta, tag_meta


def get_tag_meta(project_id, name) -> sly.TagMeta:
    project_meta = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta)
    return project_meta.get_tag_meta(name)


def read_img_tags(ds_id, image_ids, tag_meta):
    tags = []
    if len(image_ids) > 0:
        filters = [{"field": "id", "operator": "in", "value": image_ids}]
        image_infos = api.image.get_list(ds_id, filters=filters, force_metadata_for_links=False)
        id_to_info = {img_info.id: img_info for img_info in image_infos}
        for img_id in image_ids:
            curr_tags = [tag for tag in id_to_info[img_id].tags if tag["tagId"] == tag_meta.sly_id]
            tags.append(curr_tags[0] if len(curr_tags) == 1 else None)
    return tags


def read_labels_tags(object_ids, tag_meta):
    if len(object_ids) == 0:
        return []
    return [read_label_tag(obj_id, tag_meta) for obj_id in object_ids]


def read_img_tag(image_id, tag_meta):
    image_info = api.image.get_info_by_id(image_id)
    tags = [tag for tag in image_info.tags if tag["tagId"] == tag_meta.sly_id]
    if len(tags) == 1:
        return tags[0]


def read_label_tag(object_id, tag_meta):
    tags = api.advanced.get_object_tags(object_id)
    tags_filtered = [tag for tag in tags if tag["tagId"] == tag_meta.sly_id]
    if len(tags_filtered) == 1:
        return tags_filtered[0]


def read_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return read_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return read_label_tag(object_id, tag_meta)


def add_img_tags(image_ids, tag_meta, value=None):
    return api.image.add_tag_batch(image_ids=image_ids, tag_id=tag_meta.sly_id, value=value)


def add_img_tag(image_id, tag_meta, value=None):
    return api.image.add_tag(image_id=image_id, tag_id=tag_meta.sly_id, value=value)


def add_label_tag(object_id, tag_meta, value=None):
    return api.advanced.add_tag_to_object(tag_meta_id=tag_meta.sly_id, figure_id=object_id, value=value)


def add_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return add_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return add_label_tag(object_id, tag_meta)


def remove_img_tag(image_id, tag_meta):
    tag = read_img_tag(image_id, tag_meta)
    if tag:
        tag_id = tag["id"]
        return api.advanced.remove_tag_from_image(tag_meta_id=tag_meta.sly_id, image_id=image_id, tag_id=tag_id)
    else:
        return False


def remove_label_tag(object_id, tag_meta):
    tag = read_label_tag(object_id, tag_meta)
    if tag:
        tag_id = tag["id"]
        return api.advanced.remove_tag_from_object(tag_meta_id=tag_meta.sly_id, figure_id=object_id, tag_id=tag_id)
    else:
        return False


def remove_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return remove_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return remove_label_tag(object_id, tag_meta)
