# Addressing Inconsistent Labeling with Cross Image Matching for Scribble-Based Medical Image Segmentation

This repository contains the code and resources for the paper "Addressing Inconsistent Labeling with Cross Image Matching for Scribble-Based Medical Image Segmentation". This work focuses on improving the consistency of labeling in medical image segmentation by leveraging cross-image matching techniques.

## Overview

In recent years, there has been a notable surge in the adoption of weakly-supervised learning for medical image segmentation, utilizing scribble annotation as a means to potentially reduce annotation costs. However, the inherent characteristics of scribble labeling, marked by incompleteness, subjectivity, and a lack of standardization, introduce inconsistencies into the annotations. These inconsistencies become significant challenges for the network's learning process, ultimately affecting the performance of segmentation.

To overcome this challenge, we propose a solution involving the creation of a reference set to guide pixel-level feature matching. This reference set is constructed by extracting class-specific tokens and individual pixel-level features from different images annotated with various types of scribbles. Serving as a repository showcasing diverse pixel styles and classes, the reference set becomes the cornerstone for a pixel-level feature matching strategy. This strategy enables the effective comparison of unlabeled pixels, offering guidance, particularly in learning scenarios characterized by inconsistent and incomplete scribbles.

The proposed strategy incorporates smoothing and regression techniques to align pixel-level features across different images. By leveraging the diversity of pixel sources, our matching approach enhances the network's ability to learn consistent patterns from the reference set. This, in turn, mitigates the impact of inconsistent and incomplete labeling, resulting in improved segmentation outcomes.

Extensive experiments conducted on three publicly available datasets demonstrate the superiority of our approach over state-of-the-art methods in terms of segmentation accuracy and stability.

## Features

- **Cross Image Matching**: Utilizes cross-image matching to ensure consistent labeling across different images.
- **Scribble-Based Segmentation**: Focuses on improving segmentation results from minimal scribble annotations.
- **Robust and Efficient**: Designed to work efficiently with limited annotated data while maintaining high accuracy.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/scribble-medical-segmentation.git
cd scribble-medical-segmentation
pip install -r requirements.txt
