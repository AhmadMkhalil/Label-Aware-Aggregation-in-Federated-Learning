# Source Code and Results of Label-Aware Aggregation for Improved Federated Learning

- **src Folder:** Contains all classes, with `federated_main.py` as the starting point.
- **save Folder:** Contains the full set of experimental results.

# Running the Experiments:
- Please refer to `options.py` to explore the available running options.
- For example, you can run the following command: 
  ```
  python3 federated_main.py --avg_type=avg_n_classes --number_of_classes_group1_user=2 --noniidness_end_id=1 --frac=0.1
  ```
  This command sets `unique_c` to 2, `noniid_s` to 0.1, and `k_r` to 0.1.

# Troubleshooting:
- If you encounter any issues while running the code, please feel free to contact us at ahmad.khalil@kom.tu-darmstadt.de or alternatively at msc.ahmadmkhalil@gmail.com.
