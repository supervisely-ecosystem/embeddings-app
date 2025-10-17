<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/211832844-522b444d-4534-4b5e-bf12-14c7e7d3aeec.png"/>  

# Explore data with embeddings

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/embeddings-app)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/embeddings-app)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/embeddings-app)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/embeddings-app)](https://supervisely.com)


</div>

# Overview

**With this app you can explore your data: find out clusters of similar objects, outliers, or wrong labels of objects just by looking once at the chart.**


## Features
- **Any recent pre-trained model** can be used from `HuggingFace` or [timm](https://huggingface.co/docs/timm/index) (OpenAI CLIP, ConvNeXt, BEiT and others).
- **3 instance modes**: `images`, `objects` (cropping will be made automatically to every individual object), `both`.
- **Visualize** your embeddings with `UMAP`, `PCA`, `t-SNE` or their combinations.
- **Minimum recalculations**: The app can detect changes in data and only recalculates the outdated embeddings (new images, updated annotations, etc).


# How to Run

1. Run the App either from context menu of a project or dataset, or from the Supervisely ecosystem. From the context menu select `"Run app" -> "Metric Learning" -> "Embeddings App"`.
2. App will be deployed in a minute, then click `"OPEN"`.
3. There you can select another project and dataset for which you want to calculate the embeddings.
4. Select a `model` from the table or type any `model_name` from [HuggingFace hub](https://huggingface.co/models?sort=downloads&search=timm%2F) (only models from timm package are supported now).
5. Select a `device` - one of your GPUs or a CPU, and a `batch size`. The larger batch size is, the faster calculations will be, but more GPU memory will be used. For GPU device you can try bigger values like 64, 128, 256 and more.
6. (optional) Set the additional parameters:
    - **Instance mode**: whether to infer the model on entire images or on cropped objects in the images, or both the images and the cropped objects. Default: `objects`.
    - **Expand crops (px)**: if the object cropping is occurs, you can define an extent to the rectangle of the crop by a few pixels on both XY sides. It is used to give the model a little context on the boundary of the objects.
    - **Projection method**: it is a decomposition method to project the high-dimensional embeddings onto 2D space. You can select one of `UMAP`, `t-SNE`, `PCA` or its combinations. Default: `UMAP`.
    - **Metric**: a parameter used in the projection method. Default: `euclidean`.
7. `Force recalculate` checkbox: by default the app is trying to detect updates in your selected dataset and only recalculates the outdated embeddings (for images that have been updated or added). The checkbox is used when you want to pass this feature.
8. Click `Run`. Calculations may take minutes or hours depending on dataset size, your device and the batch size.
9. After finishing, the calculated embeddings will be saved to `team files` in `/embeddings/{project_name}_{id}/{model_name}`.
    The directory contains:
    - `embeddings.pt`: your saved embedding vectors. You can load it with `torch.load('embeddings.pt')`
    - `cfg.json`: saved options and parameters.
    - `info.json`: some low-level info about all calculated objects. It is used to detect the updates in a dataset.
    - `projections_{projection_method}_{metric}`: saved 2D projections after decomposition.
10. Now you can explore your data in a chart!
    - You can `click` on a point to `preview` an image and its annotation.
    - You can switch an `annotation mode`, showing only a selected object's annotation or the full image annotation.
    - Also try clicking on legends (bottom labels on the graph) to show/hide objects of a class.
11. After you finish using the app, you should stop the app manually in the workspace tasks from “three dots button” context menu, or by clicking `settings button -> stop`  right in application UI on the top of the page near the application name


**Note:**
The embeddings are calculated with large pre-trained models such as OpenAI CLIP, Facebook ConvNeXt.
These models can retrieve very complex relationships in the data, so the data samples were arranged in space by some semantic meaning, not just by color and shape of the object.

## Screenshot

<img src="https://user-images.githubusercontent.com/115161827/217641449-62b677ff-2345-4bbb-b5be-3369cd131a9c.png"/> 
