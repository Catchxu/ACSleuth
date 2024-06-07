# ACSleuth (IJCAI 2024)

<br/>
<div align=center>
<img src='/docs/ACSleuth.png' width='60%'>
</div>
<br/>

<b>Domain Adaptive and Fine-grained Anomaly Detection for Single-cell Sequencing Data and Beyond</b>

<sup>\*</sup>: equal contribution.

<i>Kaichen Xu<sup>\*</sup>, Yueyang Ding<sup>\*</sup>, Suyang Hou, Weiqiang Zhan, Nisang Chen, Jun Wang, Xiaobo Sun</i>

[ArXiv](https://arxiv.org/abs/2404.17454)


## About
Fined-grained anomalous cell detection from affected tissues is critical for clinical diagnosis and pathological research. Single-cell sequencing data provide unprecedented opportunities for this task. However, current anomaly detection methods struggle to handle domain shifts prevalent in multi-sample and multi-domain single-cell sequencing data, leading to suboptimal performance. Moreover, these methods fall short of distinguishing anomalous cells into pathologically distinct subtypes. In response, we propose ACSleuth, a novel, reconstruction deviation-guided generative framework that integrates the detection, domain adaptation, and fine-grained annotating of anomalous cells into a methodologically cohesive workflow. Notably, we present the first theoretical analysis of using reconstruction deviations output by generative models for anomaly detection in lieu of domain shifts. This analysis informs us to develop a novel and superior maximum mean discrepancy-based anomaly scorer in ACSleuth. Extensive benchmarks over various single-cell data and other types of tabular data demonstrate ACSleuth's superiority over the state-of-the-art methods in identifying and subtyping anomalies in multi-sample and multi-domain contexts.


## Citation
```
@article{xu2024domain,
  title={Domain Adaptive and Fine-grained Anomaly Detection for Single-cell Sequencing Data and Beyond},
  author={Xu, Kaichen and Ding, Yueyang and Hou, Suyang and Zhan, Weiqiang and Chen, Nisang and Wang, Jun and Sun, Xiaobo},
  journal={arXiv preprint arXiv:2404.17454},
  year={2024}
}
```
