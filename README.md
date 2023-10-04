# Leveraging Self-Supervised Learning for Fetal Cardiac Planes Classification using Ultrasound Scan Videos

## Abstract | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_7)

Self-supervised learning (SSL) methods are popular since they can address situations with limited annotated data by directly utilising the underlying data distribution. However, adoption of such methods is not explored enough in ultrasound (US) imaging, especially for fetal assessment. We investigate the potential of dual-encoder SSL in utilizing unlabelled US video data to improve the performance of challenging downstream Standard Fetal Cardiac Planes (SFCP) classification using limited labelled 2D US images. We study 7 SSL approaches based on reconstruction, contrastive loss, distillation and information theory, and evaluate them extensively on a large private US dataset. Our observations and finding are consolidated from more than 500 downstream training experiments under different settings.
Our primary observation shows that for SSL training, the variance of the dataset is more crucial than its size because it allows the model to learn generalisable representations which improve the performance of downstream tasks. Overall, the BarlowTwins method shows robust performance irrespective of the training settings and data variations, when used as an initialisation for downstream tasks. Notably, full fine-tuning with 1% of labelled data outperforms ImageNet initialisation by 12% in F1-score and outperforms other SSL initialisations by at least 4% in F1-score, thus making it a promising candidate for transfer learning from US video to image data.

In this work, we aim to clarify the following two questions regarding the dual-encoder SSL methods.
 - *How does SSL pretraining on US video data impact downstream SFCP classification with limited labelled data?*
 - *Which SSL method is effective in utilizing US video data?*


## Code

- `tasks` folder containes the individual python scripts whihc is the main entry point to run each SSL experiments and downstream tasks. Please track the imports in these scripts for easier understanding.
- `configs` parameters can be set if needed with json. Ones used for experiments are updated in releases.
- `algorithms` contains core SSL codes, backbones and other integral components.

### Downloads

NoteBooks for processing/visualization and config jsons for experiments can be found [here](https://github.com/BioMedIA-MBZUAI/Ultrasound-SSL-FetalCardiacPlanes/releases/tag/ASMUS2023-state)


## Cite Us
If you find our code or paper findings useful for your research, consider citing us :smiley:

```
@InProceedings{JGeoB2023_SSLfetalUS,
title="Leveraging Self-supervised Learning for Fetal Cardiac Planes Classification Using Ultrasound Scan Videos",
author="Benjamin, Joseph Geo and Asokan, Mothilal and Alhosani, Amna and
Alasmawi, Hussain and Diehl, Werner Gerhard and Bricker, Leanne and
Nandakumar, Karthik and Yaqub, Mohammad",
editor="Kainz, Bernhard and Noble, Alison and Schnabel, Julia and Khanal, Bishesh and M{\"u}ller, Johanna Paula and Day, Thomas",

booktitle="Simplifying Medical Ultrasound",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="68--78",

isbn="978-3-031-44521-7"
doi={10.1007/978-3-031-44521-7\_7}
}

```

For Code updates: [Mothilal](https://github.com/Mothilal-Asokan/ssl-us) and [JGeoB](https://github.com/JosephGeoBenjamin/Ultrasound-SSL-FetalCardiacPlanes)