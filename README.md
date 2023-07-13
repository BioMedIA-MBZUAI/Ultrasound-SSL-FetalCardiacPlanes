# Leveraging Self-Supervised Learning for Fetal Cardiac Planes Classification using Ultrasound Scan Videos

## Abstract | [Paper]()

Self-supervised learning (SSL) methods are gaining popularity since they can address situations with limited annotated data by directly utilising the underlying data distribution. However, adoption of such methods is not explored enough in ultrasound (US) imaging, especially for fetal assessment.
We investigate the potential of dual-encoder SSL in utilizing unlabelled US video data to improve the performance of challenging downstream Standard Fetal Cardiac Planes~(SFCP) classification using limited labelled 2D US images.
We study $7$ SSL approaches based on reconstruction, contrastive loss, distillation and information theory, and evaluate them extensively on a large private US dataset. Our observations and finding are consolidated from more than $500$ downstream training experiments under different settings.
Our primary observation shows that for SSL training, the variance of dataset is more crucial than the size of dataset because it allows the model to learn generalisable representations which improve the performance of downstream tasks.
Overall, the BarlowTwins method shows robust performance irrespective of the training settings and data variations, when used as an initialisation for downstream tasks. Notably on full fine-tuning with $1\%$ of labelled data, it outperforms ImageNet initialisation by $12\%$ in F1-score and outperforms other SSL initialisations by at least $4\%$ in F1-score. Thus making it a promising candidate for transfer learning from US video to image data.

In this work, we aim to clarify the following two questions regarding the dual-encoder SSL methods.
 - *How does SSL pretraining on US video data impact downstream SFCP classification with limited labelled data?*
 - *Which SSL method is effective in utilizing US video data?*



## Cite Us
```

```

**Related Places**

Also refer code base of [Mothilal Asokan](https://github.com/Mothilal-Asokan/ssl-us)

NoteBooks for processing and visualization can be found [here](https://github.com/JosephGeoBenjamin/ssl-us/releases/tag/ASMUS2023-state)