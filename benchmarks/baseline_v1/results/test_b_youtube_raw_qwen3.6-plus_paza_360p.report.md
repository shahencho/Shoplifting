## Results: test_b_youtube_raw_qwen3.6-plus_paza_360p
Clips scored: 4

|                | pred theft | pred normal |
|----------------|-----------:|------------:|
| **true theft** |          3 |           1 |
| **true normal**|          0 |           0 |

- Precision:   100.0%
- Recall:      75.0%
- Specificity: 0.0%
- F1:          0.857
- Accuracy:    75.0%

### vs. Paza (reported)

| metric | ours | Paza |
|---|---:|---:|
| precision | 1.000 | 0.895 |
| recall | 0.750 | 0.593 |
| specificity | 0.000 | 0.928 |
| f1 | 0.857 | 0.713 |

### Alternative rule: only CONFIRMED counts as an alarm

TP 2, FN 2, FP 0, TN 0 -> precision 100.0%, recall 50.0%, specificity 0.0%, F1 0.667

### Verdict breakdown

label  verdict  
1      CONFIRMED    2
       NORMAL       1
       UNCERTAIN    1

Total API cost: $0.0571
Mean latency: 120.1s

### False alarms (normal clip flagged): 0


### Misses (theft clip passed as normal): 1

- `7aMUGLzBQFw` NORMAL (90): The person enters the store empty-handed (Frame 1-2). In Frames 4 and 5, the person extends their left arm to reach for an item on the display shelf. There are no observable actions of placing items into pockets, tucking items under clothing, or hiding items behind the body. The interaction is limited to reaching towards the shelf, consistent with normal browsing behavior.