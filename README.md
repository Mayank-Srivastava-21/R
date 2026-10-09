# Provable Decremental Reinforcement Unlearning Against Environment Inference Attacks

Official implementation for our NeurIPS submission.

## What this repo does
1. Trains a continuous-control SAC agent on a partitioned continuous MDP
   (`SectorNav`: 4 sectors = 4 sub-environments with distinct drift dynamics).
2. Revokes sector 0 and performs **certified reinforcement unlearning**:
   - Decremental RL (Q-suppression hinge) on the revoked sector,
   - Environment poisoning (action inversion + reward flip),
   - KL-anchoring to the original policy on retained sectors (utility preservation),
   - Binary-tree TV-stable prefix-sum mechanism with maximal coupling
     (certifies the rho*sqrt(ln T) retraining-cost bound, Theorem 3).
3. Mounts a trained **environment-inference adversary** and verifies that the
   unlearned policy is statistically indistinguishable (~50% accuracy) from an
   oracle that never saw the revoked environment.
4. Compares against baselines: Oracle retrain (LFS), Non-transferable LFS,
   Catastrophic-forgetting fine-tune, and ablations (w/o poisoning).

## Run
```bash
pip install -r requirements.txt
python main.py --seeds 0 1 2          # full paper run
python main.py --seeds 0 --steps 20000  # quick smoke test
