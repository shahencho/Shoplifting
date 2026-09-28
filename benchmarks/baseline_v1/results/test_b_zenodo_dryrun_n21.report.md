## Results: test_b_zenodo_dryrun_n21
Clips scored: 21

|                | pred theft | pred normal |
|----------------|-----------:|------------:|
| **true theft** |         16 |           5 |
| **true normal**|          0 |           0 |

- Precision:   100.0%
- Recall:      76.2%
- Specificity: 0.0%
- F1:          0.865
- Accuracy:    76.2%

### vs. Paza (reported)

| metric | ours | Paza |
|---|---:|---:|
| precision | 1.000 | 0.895 |
| recall | 0.762 | 0.593 |
| specificity | 0.000 | 0.928 |
| f1 | 0.865 | 0.713 |

### Alternative rule: only CONFIRMED counts as an alarm

TP 0, FN 21, FP 0, TN 0 -> precision 0.0%, recall 0.0%, specificity 0.0%, F1 0.000

### Verdict breakdown

label  verdict  
1      NORMAL        5
       TRIGGERED    16

Total API cost: $0.0000
Mean latency: 0.0s

### False alarms (normal clip flagged): 0


### Misses (theft clip passed as normal): 5

- `main95_Trim1` NORMAL (0): nan
- `main307_Trim1` NORMAL (0): nan
- `main120_Trim1` NORMAL (0): nan
- `main182_Trim2` NORMAL (0): nan
- `main163_Trim1` NORMAL (0): nan