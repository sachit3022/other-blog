+++
author = "Sachit gaudi"
title = "Misconceptions: Flow models"
date = "2025-01-01"
description = "Flow"
math = true
pygmentsUseClasses=true
pygmentsCodeFences=true
tableOfContents = true
draft = true
+++
### Brief introduction to flow models
The objective of the generative models is to generate samples from the 

### What are some of the misconceptions?  
 - Flow models always produce straigth paths






Scalable way to train flow models is proposed in <a href='1'>[1]</a>. <a href='2'>[2]</a> provides a very good understanding of the flow models and extends the proofs in great detail along with connections to diffusion models. However, authors claim that the flow models learn 'straight' paths. Although, <a href='3'>[3]</a> has emperically shown that the flow based models does not learn 'straight' paths. In this blog, we will derive the closed form for the path, for the gaussian data distribution. Let's define what are 'straight' paths?
Simple case of $x_0$ of gaussian $\mathcal{N}(0,1)$ to gaussian $\mathcal{N}(\mu,\sigma)$




 

<p><a id="1">[1]</a> Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le.  <i>Flow Matching for Generative Modeling</i>. ICLR, 2023.</p>
<p><a id="2">[2]</a> Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, Itai Gat. <i>Flow Matching Guide and Code</i>. arXiv, 2023.</p>
<p><a id="3">[3]</a> Gao, Ruiqi and Hoogeboom, Emiel and Heek, Jonathan and Bortoli, Valentin De and Murphy, Kevin P. and Salimans, Tim. <i>Diffusion Meets Flow Matching: Two Sides of the Same Coin</i>. https://diffusionflow.github.io, 2024.</p>
