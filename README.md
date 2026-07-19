# Face-Aging-Model

**Name**: Tathagata Roy
**Roll No.**: 25B3954

---

This is the official repository for the Face Aging Model project, conducted under the prestigious Seasons of Code (SoC) 2026 initiative by the Web and Coding Club (WnCC), IIT Bombay.

This project leverages advanced Machine Learning and Computer Vision techniques to simulate the complex biological process of human facial aging. By exploring generative models and deep learning architectures, the project focuses on accurately rendering age-progressive and age-regressive facial transformations while preserving identity-specific features. This repository serves as a centralized hub for all codebase development, experimental pipelines, datasets, and trained models.

---

## Project Timeline and Phases

### 1. Learning Phase (Weeks 1 to 8)
The initial stage of the project was dedicated to building a foundational knowledge base in computer science vision techniques and algorithmic image manipulation. 
* **Weeks 1 and 2:** Focused on mathematics for imaging, fundamental pixel operations, and color space dynamics using fundamental data arrays.
* **Weeks 3 and 4:** Studied traditional feature extraction techniques and bounding box region tracking, mapping how geometric face landmarks change across human growth patterns.
* **Weeks 5 and 6:** Explored classical machine learning structures, neural networks, latent space transformations, and the mathematical properties of convolutional filters.
* **Weeks 7 and 8:** Researched specialized aging datasets, examined lighting normalization challenges, and studied structural loss metrics required to protect identity consistency during image modification.

### 2. Project Phase (Pipeline Implementation)
The execution stage focused on constructing a complete, automated programming pipeline within an engine framework to apply extreme aging effects onto uploaded portraits.
* **Image Ingestion and Preprocessing:** Developed an interface to receive user image uploads, passing them to a cascade classifier that dynamically crops the primary face region and normalizes the target matrix to a crisp square grid resolution.
* **Structural Matrix Shifting:** Implemented high-frequency texture layer blending that targets micro-creases, crow's feet, and deep age lines without relying on external server environments.
* **Pigment and Contrast Decay:** Applied custom color space adjustments to drain color saturation and introduce randomized luminance scaling, effectively producing realistic thinning skin and age spot distributions.
* **Diagnostic Visualization:** Built a data visualization module that maps the source identity side by side with the hyper-aged rendering, automatically saving the output comparison matrix.
