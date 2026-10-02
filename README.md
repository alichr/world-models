<div align="center">

<img src="assets/banner.svg" alt="World Models — learning to perceive, predict, and plan in latent space" width="100%"/>

<br/>

<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-818cf8?style=for-the-badge&labelColor=0d1433"/></a>
<img alt="Status" src="https://img.shields.io/badge/status-active%20research-22d3ee?style=for-the-badge&labelColor=0d1433"/>
<a href="https://github.com/alichr/world-models/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/alichr/world-models?style=for-the-badge&color=c084fc&labelColor=0d1433"/></a>
<a href="https://github.com/alichr/world-models/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/alichr/world-models?style=for-the-badge&color=60a5fa&labelColor=0d1433"/></a>

<br/>

<sub><i>“A world model is an agent's internal simulator — it learns how the world works so it can imagine before it acts.”</i></sub>

</div>

---

## Implemented methods

Each method lives in its own self-contained folder with code, training scripts and a detailed README.

| # | Method | Paper | Environment | Result (this repo) | Paper |
|:-:|---|---|---|:-:|:-:|
| 01 | [**World Models**](01-world-models-2018/) | Ha & Schmidhuber, 2018 · [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) | CarRacing | **803 ± 102** | 906 ± 21 |

<sub>Scores are average reward on unseen tracks (CarRacing is considered solved at 900). The 01 result is from 90 CMA-ES generations on a single laptop (Apple M4, no CUDA GPU), evaluated on 32 tracks.</sub>
