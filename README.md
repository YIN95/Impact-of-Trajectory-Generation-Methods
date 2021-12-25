# Group Behavior Recognition Using Attention- and Graph-Based Neural Networks

[[paper](https://ieeexplore.ieee.org/abstract/document/9223584?casa_token=aY1tENkQZ78AAAAA:7wbTad2eCTkzcD_6_X33GbTCU4kb0Ij6C6rY70XXso1aOdCu1mqyNf6bkbF3qV1qboDaxAF-lw)] [[data](https://sites.google.com/view/congreg8/home#h.p_FqKt0M9-taTN)]

If this code helps with your work, please cite:

```bibtex
@inproceedings{yang2020impact,
  title={Impact of trajectory generation methods on viewer perception of robot approaching group behaviors},
  author={Yang, Fangkai and Yin, Wenjie and Bj{\"o}rkman, M{\aa}rten and Peters, Christopher},
  booktitle={2020 29th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
  pages={509--516},
  year={2020},
  organization={IEEE}
}
```

## Dataset

To explore our research questions, we designed an exploratory scenario, Who’s the Spy, for human-robot interactions.

This game involves three players in a group. In every game round, each player is given a card with a word on it. The Spy has a different word card. The players take turns to describe the word and the robot approaches to join the group to identify the spy.

## Experimental Conditions

1. Group type: a) static b) quasi-dynamic [1]
2. Approaching Direction
3. Camera viewpoints: a) egocentric view b) perspective view

## Methods
1. WoZ
In the WoZ (Wizard-of-Oz) approach, the robot is teleoperated by a human operator (a researcher who is a trained operator).


2. Procedural Model
We use the social-aware navigation method as a procedural model to generate approaching group trajectories


3. Imitation Learning Model
We generate approaching group trajectories use a Generative Adversarial Imitation Learning (GAIL) framework with a Group Behavior Recognition framework.














