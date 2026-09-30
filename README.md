<h3 align="center">Reinforcement learning via conservative agent for environments with random delays</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-blue?logo=python&style=flat-square" alt="Python Badge"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0.0-%23EE4C2C?logo=pytorch&style=flat-square" alt="PyTorch Badge"></a>
</p>

---

> Pytorch implementation of conservative reinforcement learning for random-delay environments.  
> Paper link: https://www.sciencedirect.com/science/article/abs/pii/S0893608026001073

---

#### Test environments

    python == 3.8.0  
    pytorch == 2.0.0  
    mujoco == 2.2.0  
    mujoco_py == 2.1.2.14  
    gym == 0.26.2  

---

#### Runs

    python main.py --env-name HalfCheetah-v3 --min-obs-delayed-steps 0  --max-obs-delayed-steps 10 --init-obs-delayed-steps 10 --delay-type uniform --random-seed 2026 max-step 1000000


---

#### Citation

    @article{lee2026reinforcement,
      title={Reinforcement Learning via Conservative Agent for Environments with Random Delays},
      author={Lee, Jongsoo and Kim, Jangwon and Jeong, Jiseok and Han, Soohee},
      journal={Neural Networks},
      pages={108645},
      year={2026},
      publisher={Elsevier}
    }


---

#### Acknowledgement

> [Belief Projection-based Q-learning, NeurIPS 2023](https://github.com/jangwonkim-cocel/BPQL)

