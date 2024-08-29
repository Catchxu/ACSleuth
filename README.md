# [IJCAI 2024] Domain Adaptive and Fine-grained Anomaly Detection for Single-cell Sequencing Data and Beyond

<br/>
<div align=center>
<img src='/docs/ACSleuth.png' width='60%'>
</div>
<br/>

[Paper link](https://arxiv.org/abs/2404.17454)


## About
Fined-grained anomalous cell detection from affected tissues is critical for clinical diagnosis and pathological research. Single-cell sequencing data provide unprecedented opportunities for this task. However, current anomaly detection methods struggle to handle domain shifts prevalent in multi-sample and multi-domain single-cell sequencing data, leading to suboptimal performance. Moreover, these methods fall short of distinguishing anomalous cells into pathologically distinct subtypes. In response, we propose ACSleuth, a novel, reconstruction deviation-guided generative framework that integrates the detection, domain adaptation, and fine-grained annotating of anomalous cells into a methodologically cohesive workflow. Notably, we present the first theoretical analysis of using reconstruction deviations output by generative models for anomaly detection in lieu of domain shifts. This analysis informs us to develop a novel and superior maximum mean discrepancy-based anomaly scorer in ACSleuth. Extensive benchmarks over various single-cell data and other types of tabular data demonstrate ACSleuth's superiority over the state-of-the-art methods in identifying and subtyping anomalies in multi-sample and multi-domain contexts.


## Get Started
### Prepare Data
ACSleuth accepts single-cell omics data in AnnData format as input. You need to prepare a reference dataset consisting of normal cells, a target dataset consisting of both normal and anomaly cells (to be detected), and a label/ground truth file for validating the results. To better organize the data, you can use `sleuth.save_pkl` to package these datasets into a pkl file for storage. Here, we have provided a preprocessed and well organized demo dataset, `TME_3000.pkl`, located in the `./data/` directory. The subsequent tutorials will be based on this dataset.

### Anomalous Cells Detection
```
python experiment/run_anomaly.py
```


## Citation
If you find our code useful for your research, please cite our paper.
```
@inproceedings{ijcai2024p677,
  title     = {Domain Adaptive and Fine-grained Anomaly Detection for Single-cell Sequencing Data and Beyond},
  author    = {Xu, Kaichen and Ding, Yueyang and Hou, Suyang and Zhan, Weiqiang and Chen, Nisang and Wang, Jun and Sun, Xiaobo},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {6125--6133},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/677},
  url       = {https://doi.org/10.24963/ijcai.2024/677},
}
```


## Contact
If you have any questions, please contact Kaichenxu358@gmail.com.