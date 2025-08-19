# Dissolution Stability Machine Learning Project Using a Gaussian Mixture Model

## Project Overview

This project aims to analyze dissolution test data using GMMs.

In pharmaceutical development and manufacturing, dissolution testing data allows one to predict how quickly and completely a drug will dissolve in a patient's body.

Dissolution methods, however, may vary greatly (e.g., different media, apparatus, agitation speeds, sampling times), and real test results are prone to showing heterogeneity (e.g., some tablets dissolve quickly, others slowly). This project therefore aims to address this problem by building a computational workflow which is able to complete the following:
1) Translate Food and Drug Administration (FDA) method metadata into clearly structured features;
2) Use kinetic models to generate realistic synthetic dissolution profiles;
3) Apply a selection of Gaussian Mixture Models (GMMs) to cluster these profiles into meaningful subgroups (e.g., “fast,” “medium,” “slow”); and
4) Provide a framework that can be applied to real experimental data to detect unexpected subpopulations or anomalies.

This project hopes to add value within the pharmaceutical and biotechnology sector as it could demonstrate how data analysis, machine learning, and statistical modeling can benefit method development, risk assessment, and QC analysis.

The program will intake as inputs the FDA's Dissolution methods database metadata (which includes a drug's dosage form (tablet, capsule, etc.), its apparatus type (basket, paddle, cylinder, etc.), its agitation speed (rpm), its medium composition (e.g., water, HCl, buffer with surfactant), its medium volume (e.g., 500 mL, 900 mL), and its sampling times (e.g., 5, 10, 20, 30, 60 min)) and a selection of synthetic dissolution profiles (simulated from the metadata). For instance, a first-order or Weibull kinetic model may be used to generate curves plotting the percentage of the drug dissolved over time. Parameters can parhaps be linked to method settings (e.g., higher RPM values could lead to faster rate constants).

The program aims to yield several outputs. Firstly, it will cluster dissolution profiles illustrating subgroups of methods/curves labeled as “fast,” “medium,” or “slow” dissolvers. It will also produce informative visualizations such as plots of dissolution curves by cluster and 2D/3D projections of feature space color coded by cluster group. The program will provide cluster descriptors––statistical summaries of what differentiates each subgroup (e.g., Cluster 0 uses low RPM, short times result in fast release). Finally, it will generate probabilistic insights drawn from the data. By using GMMs, the likelihood that a new profile belongs to each subgroup will be known which is crucial for uncertainty handling and model evaluation.

It is hoped that this project would become useful in real-world pharmaceutical and/or biotechnology contexts during several stages, including in Research and Development (R&D), in Process and Method Development, in Manufacturing and Quality Assurance, and in Regulatory and Reporting Operations.

This program may be useful during formulation development as it could be capable of identifying whether a drug batch behaves consistently or splits into sub-populations (e.g., two different release rates). This could help scientists detect potential formulation robustness issues early. It may also be potentially used to compare FDA-recommended methods across drugs and to find natural clusters of dissolution conditions (e.g., methods suited for immediate vs. extended-release). During operation and quality checking, the most performant model might be applied to real batch data. If some tablets cluster into a “slow dissolving” subgroup unexpectedly, this could flag a potential manufacturing deviation which should be addressed. Furthermore, it might be able to provide a systematic way to justify why certain dissolution conditions (RPM, media, times) are chosen. Hence, this program could add a data-driven layer on top of traditional method validation.