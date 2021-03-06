# Automated Cell Recognition Using Single-cell RNA sequencing with Machine Learning
<p align="center">
  <a href="https://github.com/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning?style=plastic"/></a>
  <a href="https://github.com/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning"><img alt="Stars" src="https://img.shields.io/github/stars/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning?style=plastic"/></a>
  <a href="https://github.com/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning"><img alt="Language Use" src="https://img.shields.io/github/languages/count/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning?style=plastic"/></a>
</p>
 This project investigates and summarizes the superiority and limitations of different dimensionality reduction schemes as well as classification methods in specific single-cell RNA sequencing (scRNA-seq) data sets. 

## Table of Contents

<details open="open">
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#nav-1)
- [Introductions](#nav-2)
  - [Background](#nav-2-1)
  - [Pipeline](#nav-2-2)
  - [Dataset](#nav-2-3)
- [Graphics](#nav-3)
- [Features](#nav-4)
- [Contributors](#nav-5)
- [Special Thanks](#nav-6)
- [License](#nav-7)

</details>

<span id="nav-1"></span>

## Introductions

<span id="nav-2"></span>

### Background
Although scRNA-seq technology has gained further capability to capture differential information at the cellular level compared to earlier transcriptome analysis methods including bulk RNA-seq, the cross-cellular technical errors arising from its data acquisition phase and other limitations provide challenges for researchers to maintain a balance between data pre-processing and information retention. Based on this, several relatively mature schemes including t-SNE, PCA, and multiple algorithm combinations on data dimension reduction was explored and tested in this report, and evaluated the accuracy obtained by machine-learning-based classifiers for cell classification tasks as a base metric for comprehensive comparison and evaluation.
<span id="nav-2-1"></span>


### Pipeline
![pipeline](https://user-images.githubusercontent.com/31975605/142730036-ff44b376-2a56-4d84-b76e-c5f3371fff47.png)<br>

This is the pipeline for large-scale, cell identification task from the beginning of raw data to the final classification.
**a.** Labels + Reads Per Kilobase per Million mapped reads. **b.** Multiple dimension reduction methods with multiple dimensions applied. **c.** The specific implementation principle of the PCA + t-SNE combination algorithm. **d.** Visualization in both 2 & 3 dimensions and both with & without labels. **e.** Multiple classifiers with multiple parameters applied
<span id="nav-2-2"></span>

### Dataset
The reprocessed dataset that supports the conclusion of this paper are publicly available online at https://scquery.cs.cmu.edu/processed_data/.
<span id="nav-2-3"></span>

## Graphics
![avatar](https://raw.githubusercontent.com/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning/main/visualization/AE2DVIS.JPG)
![avatar](https://raw.githubusercontent.com/Stry233/Automated-Cell-Recognition-Using-Single-cell-RNA-sequencing-with-Machine-Learning/main/visualization/PCATSNE2D_LABEL.JPG)
<span id="nav-3"></span>


## Contributors
<details>
  <summary>Click me to Open/Close the contributors listing</summary>
  
- [Yuetian Chen](https://github.com/stry233) - Rensselaer Polytechnic Institute, 110 8th Street, Troy, NY, United States, 12180 (email: cheny63@rpi.edu)
- [Chenqi Xu](https://github.com/lightingtom) - Southern University of Science and Technology, Shenzhen, China, 518055
- Yiyang Cao - The University of British Columbia, Vancouver, BC, Canada, V6T 1Z4

</details>
<span id="nav-4"></span>

## Special Thanks
This research was undertaken as part of the CIS - Introduction to Machine Learning "Our Body" Project. Thanks to Prof. Ziv Bar-Joseph for his guidance and instruction in dataset pre-processing and paper refinement.
<span id="nav-5"></span>

## License
[License MIT](LICENSE)
