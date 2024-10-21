# Source Code for *Label-Aware Aggregation for Improved Federated Learning*

This repository contains the source code and supplementary files for the research paper **"Label-Aware Aggregation for Improved Federated Learning"**, authored by **Ahmad Khalil et al.**, published in the **2023 Eighth International Conference on Fog and Mobile Edge Computing (FMEC)**.

The paper is available on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10306055). This code is provided to facilitate reproducibility, further research, and real-world application of the proposed methods.

---

## Abstract

Federated Averaging (FedAvg) is the most common aggregation method used in federated learning, which performs a weighted average of the updates based on the dataset sizes of individual clients. Recent research suggests that FedAvg might not be optimal, as it does not fully account for the diversity in client data distributions. In this paper, we propose **FedLA** (Label-Aware Aggregation), a method that addresses the issue of biased models by considering label distribution alongside data size in the weighted averaging process.

FedLA is particularly effective in scenarios with heterogeneous data distributions and limited client participation in federated learning. Through extensive experiments, we show that FedLA outperforms FedAvg, especially when only a small group of clients participates. Additionally, we explore various properties of data distributions that can guide the selection of an appropriate aggregation method for federated learning tasks.

---

## Repository Structure

- **src Folder:** Contains all source code files, including the main script `federated_main.py` which serves as the entry point for running the experiments.
- **save Folder:** Stores experimental results, model checkpoints, and logs.
- **options.py:** Provides various configuration options for running the experiments.

---

## Running the Experiments

You can run the experiments as described in the paper using the following command format:

```bash
python3 federated_main.py --avg_type=avg_n_classes --number_of_classes_group1_user=2 --noniidness_end_id=1 --frac=0.1
```

### Key Parameters:
- `--avg_type=avg_n_classes`: Aggregation type, set to label-aware averaging (`avg_n_classes`).
- `--number_of_classes_group1_user=2`: Number of classes assigned to clients in group 1.
- `--noniidness_end_id=1`: Controls the level of non-IID (non-identical and independent distribution) among clients.
- `--frac=0.1`: Fraction of clients participating in the training process.

For a full list of parameters and options, refer to `options.py`.

---

## Troubleshooting

If you encounter any issues while running the code, please don't hesitate to reach out to the authors for assistance:

- **Ahmad Khalil**: ahmad.khalil@kom.tu-darmstadt.de or dr.ing.ahmad.khalil@gmail.com

---

## Results

The results of our experiments can be reproduced by following the instructions above. You can explore the saved results and visualizations stored in the `save` folder after the experiment runs.

---

## Citation

If you find this work useful in your research or utilize the code in any form, please cite the paper as follows:

```
@INPROCEEDINGS{10306055,
  author={Khalil, Ahmad and Wainakh, Aidmar and Zimmer, Ephraim and Parra-Arnau, Javier and Anta, Antonio Fernandez and Meuser, Tobias and Steinmetz, Ralf},
  booktitle={2023 Eighth International Conference on Fog and Mobile Edge Computing (FMEC)}, 
  title={Label-Aware Aggregation for Improved Federated Learning}, 
  year={2023},
  volume={},
  number={},
  pages={216-223},
  keywords={Multi-access edge computing;Federated learning;Heterogeneous data distribution;non-IID},
  doi={10.1109/FMEC59375.2023.10306055}}
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### Acknowledgements

We would like to thank the contributors and the community for their feedback and support in developing this method. Special thanks to [TU Darmstadt](https://www.tu-darmstadt.de) and [the KOM research group](https://www.kom.tu-darmstadt.de).

---
