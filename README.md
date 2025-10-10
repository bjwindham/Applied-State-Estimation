# Applied State Estimation

## About
This repository contains code generated while studying applied state estimation from RJ Labbe's *"Kalman and Bayesian Filters in Python"* and *"Estimation with Applications to Tracking and Navigation"* by Bar-Shalom, Li, and Kirubarajan. As I progress through the material, I will upload code used to generate plots for my reports and for learning purposes.

## File Descriptions
As I upload new scripts, I provide brief descriptions of what each script does and the Python dependencies required to run it.

- **Applied_State_Estimation.pdf**  
    This is a running document of my notes from the textbooks. **Please note** that some sections are not entirely original or perfectly paraphrased. This document is primarily intended to help me extract and remember key concepts.

- **FilterLib.py**  
    A library of filters and tools I created while studying state estimation. Currently, it includes Python classes for:
    - Creating and simulating state-space systems
    - PID controllers
    - ARX system identification  

    I may eventually split these into separate files so that system identification, controllers, estimators, and system simulation code are organized individually.

- **First_order_noise_filtering.py**  
    This script compares the $\alpha - \beta$ (GH) and $\alpha - \beta - \gamma$ (GHK) filters with a 1st-order low-pass filter. It demonstrates that during portions of constant velocity and/or constant acceleration, the GH and GHK filters respectively outperform the low-pass filter, while performing similarly at constant values. Note that GH and GHK filters have some "inertia" and may not handle sharp transients as effectively.

- **Kalman_1D.py**  
    This script generates a 1st-order system subject to random process disturbances and measurement noise. From the noisy measurements, an ARX model is fit, which is then used in the Kalman filter for the prediction step. The following are plotted on the same figure:
    - Step input
    - The "ideal" system (no process or measurement noise)
    - The measured signal
    - The identified model signal
    - The measured signal filtered with a single-pole low-pass filter
    - The measured signal filtered with the Kalman filter

## Disclaimer
ChatGPT was used to help accelerate the development of various "accessory" classes, such as the ARX class and the State Space class. However, all code has been thoroughly vetted by myself, and **all** code for the various filters has been fully written by me. ChatGPT was also used for formatting and correcting typos in this `.md` file.
