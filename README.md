# Applied State Estimation

## About
Repository of code generated while studying applied state estimation from RJ Labbe's "Kalman and Bayesian Filters in Python" and "Estimation with Applications to Tracking and Navigation" by Bar-Shalom, Li, and Kirubarajan. As I progress through, I'll upload code that's been used to generate plots fo my reports, and used for learning purposes.

## File descriptions
In this section, as I upload new scripts, I'll provide a very brief description of what each script is/what it's used for, and the python dependencies to run it.
- Applied_State_Estimation.pdf
  
    This is my running document of notes from my textbooks. **Please note** that some sections of this document aren't entirely original or perfectly paraphrased. This document is largely just to help me extract and remember important sections.
  
- FilterLib.py

    This is the library of filters and tools that I've created in my study of state estimation. Currently, it includes python classes for creating/simluating state space systems, PID controllers, and performing ARX system Identification. I may eventually split these out to live in different files so I can have SysID, controller, estimations, and systems code all separated.
  
- First_order_noise_filtering.py

    This script compares $\alpha-\beta$ (GH) and  $\alpha-\beta-\gamma$ (GHK) filters with a 1st order low pass filter. It shows that during portions of constatnt velocity and/or constant acceleration, the GH and GHK filters respectivly outperform the low pass filter with similar performance at constant values. However, it hsould be noted, that the GH and GHK filters have some "inertia" and thus don't handle sharp transients as well.
  
- Kalman_1D.py
