# 🌍 FUSAR-GPT  
### A SAR-Centric Vision-Language Foundation Model  
**Accepted by CVPR 2026** [![arXiv](https://img.shields.io/badge/arXiv-2602.19190-b31b1b.svg)](http://arxiv.org/abs/2602.19190)

> FUSAR-GPT is a large-scale SAR-centric multimodal model designed to bridge the gap between Synthetic Aperture Radar (SAR) imagery and modern Vision-Language Models (VLMs).

---

## 🔥 News

- 🎉 **[2026.02]** FUSAR-GPT has been accepted to **CVPR 2026**.

---

## 🧠 Motivation

Although Vision-Language Models (VLMs) have achieved remarkable success in optical imagery, their performance on SAR imagery remains significantly limited due to:

- Modality mismatch between optical pretraining and SAR backscatter characteristics  
- Lack of large-scale SAR-language alignment data  
- Weak spatial reasoning and target-level perception in SAR  

FUSAR-GPT is designed to address these challenges through a **SAR-centric training paradigm** and a **two-stage semantic alignment strategy**, enabling domain-generalizable geospatial reasoning.

---

## 🏗️ Architecture Overview
<p align="center">
  <img src="figures/FUSAR-GPT-total.png" width="95%">
</p>

FUSAR-GPT consists of two training stages:

### Stage 1: Large-Scene SAR Captioning Alignment
- 10k high-level descriptive texts  
- Terrain distribution  
- Spatial layout  
- Target concentration patterns  
- Generated using GPT-based structured prompting  

### Stage 2: Target-Level Reasoning
- 2k fine-grained annotations  
- Object counting  
- Localization  
- Classification  
- Detection-oriented supervision  

The framework integrates:

- SAR-adapted visual encoder  
- Multimodal alignment module  
- Structured spatial reasoning training  

---

## 📊 Experimental Highlights

FUSAR-GPT demonstrates:

- Clear improvements over optical-pretrained VLM baselines on SAR understanding tasks  
- Competitive detection performance without task-specific architectural modifications  
- Strong generalization within the evaluated SAR benchmarks
---

## 🔗 Connection to FUSAR-KLIP

This work extends our prior research, **FUSAR-KLIP**[![arXiv](https://img.shields.io/badge/arXiv-2509.23927-b31b1b.svg)](https://arxiv.org/abs/2509.23927), which is a knowledge-guided self-supervised multimodal learning framework for SAR (Synthetic Aperture Radar) image interpretation. It introduces hierarchical cognitive reasoning chains to guide large language models in structured SAR image captioning and semantic understanding.



For more information, please see:

[![GitHub](https://img.shields.io/badge/GitHub-FUSAR--KLIP-1f77b4.svg?logo=github)](https://github.com/yangyifremad/FUSAR-KLIP)


## 📚 Citation

If you find FUSAR-GPT useful in your research, please consider citing:


```bibtex
@misc{zhang2026fusargptspatiotemporalfeatureembedded,
  title={FUSAR-GPT: A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery},
  author={Xiaokun Zhang and Yi Yang and Ziqi Ye and Baiyun and Xiaorong Guo and Qingchen Fang and Ruyi Zhang and Xinpeng Zhou and Haipeng Wang},
  year={2026},
  eprint={2602.19190},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.19190}
}
@misc{yang2025fusarklipmultimodalfoundationmodels,
      title={FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing}, 
      author={Yi Yang and Xiaokun Zhang and Qingchen Fang and Jing Liu and Ziqi Ye and Rui Li and Li Liu and Haipeng Wang},
      year={2025},
      eprint={2509.23927},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.23927}, 
}



