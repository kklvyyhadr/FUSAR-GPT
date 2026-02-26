# 🌍 FUSAR-GPT  
### A SAR-Centric Vision-Language Foundation Model  
**Accepted by CVPR 2026**

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




