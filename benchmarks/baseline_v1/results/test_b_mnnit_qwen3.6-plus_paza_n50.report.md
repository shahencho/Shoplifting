## Results: test_b_mnnit_qwen3.6-plus_paza_n50
Clips scored: 50

|                | pred theft | pred normal |
|----------------|-----------:|------------:|
| **true theft** |         18 |           7 |
| **true normal**|          1 |          24 |

- Precision:   94.7%
- Recall:      72.0%
- Specificity: 96.0%
- F1:          0.818
- Accuracy:    84.0%

### vs. Paza (reported)

| metric | ours | Paza |
|---|---:|---:|
| precision | 0.947 | 0.895 |
| recall | 0.720 | 0.593 |
| specificity | 0.960 | 0.928 |
| f1 | 0.818 | 0.713 |

### Alternative rule: only CONFIRMED counts as an alarm

TP 17, FN 8, FP 1, TN 24 -> precision 94.4%, recall 68.0%, specificity 96.0%, F1 0.791

### Verdict breakdown

label  verdict  
0      CONFIRMED     1
       NORMAL       24
1      CONFIRMED    17
       NORMAL        7
       UNCERTAIN     1

Total API cost: $0.2309
Mean latency: 38.4s

### False alarms (normal clip flagged): 1

- `normal/normal-59` CONFIRMED (90): In Frame 4, the individual is observed holding a white rectangular box in their left hand while holding a black device (likely a phone) in their right hand. In the subsequent Frame 5, the white box is no longer visible in their hands or on the immediate table surface, and the person is holding the black device with both hands. The disappearance of the item without it being placed back on the table suggests it was concealed, likely placed into a pocket.

### Misses (theft clip passed as normal): 7

- `shoplifting/shoplifting-50` NORMAL (95): The footage depicts an office environment with cubicles and computers, rather than a retail store. The person is seated at a desk working on a computer in frames 1 through 4, and stands up in frame 5. There are no observable actions of concealing merchandise; the individual interacts only with the computer peripherals (keyboard/mouse), and a blue backpack remains stationary on the adjacent chair throughout the sequence.
- `shoplifting/shoplifting-2` NORMAL (95): The person remains seated at a desk throughout all five frames, facing left with their hand near their face. There is no interaction with merchandise, shelves, or the backpack in the foreground. No concealment actions such as placing items in pockets, tucking items under clothing, or hiding items are observed. The behavior appears to be resting or waiting.
- `shoplifting/shoplifting-80` NORMAL (90): The person is observed interacting exclusively with his personal blue backpack. In Frame 1, he stands with the bag on his shoulder. In Frames 2-5, he removes the bag, holds it in front of him, opens it, and reaches inside, appearing to organize or retrieve personal items. He does not touch or move any of the items on the adjacent table (the box, bottle, or packages), which remain stationary throughout the sequence. There is no evidence of him picking up merchandise or concealing it on his person or in the bag.
- `shoplifting/shoplifting-47` NORMAL (15): The person is observed handling a black backpack near a table with various items (boxes, water bottle, papers). In Frame 1 and Frame 2, he is seen holding and adjusting the backpack, appearing to open it or check inside. However, he does not interact with the merchandise on the table; all items (water bottle, boxes) remain in the exact same position in Frame 5 as in Frame 1. In Frames 3 and 4, he turns and walks away carrying the backpack openly at his side. There is no observable action of placing items into the bag, palming items, or hiding items behind the body. The behavior is consistent with a person preparing to leave with their personal bag.
- `shoplifting/shoplifting-3` NORMAL (90): The person is observed browsing items on a table. In Frame 3, he reaches out and touches an orange item, but in Frame 4, the item remains visible on the table and his hands return to his side, indicating he did not take it. In Frame 5, he picks up a white package from the top of a cardboard box and holds it openly in his hand. There are no observable actions of placing items into pockets, tucking items under clothing, or hiding items behind the body. The hand positions in Frames 2 and 4 appear to be neutral resting positions or adjusting clothing, unrelated to concealing merchandise.
- `shoplifting/shoplifting-30` NORMAL (95): The person is seated at a desk throughout all five frames, resting their chin on their hand. There is a backpack visible on a nearby chair, but the person does not interact with it or any other objects. There are no hand-object interactions suggesting the concealment of merchandise, such as placing items in pockets or bags. The behavior is consistent with someone sitting and waiting or thinking, with no suspicious activity observed.
- `shoplifting/shoplifting-84` NORMAL (95): The person is seen opening and adjusting a blue backpack in the first two frames. In Frame 3, with the backpack slung over his shoulder, he reaches for a small box on a table. In Frames 4 and 5, he holds the box openly in his hand and examines it. At no point is the merchandise placed into the bag, tucked into clothing, or hidden from view; the item remains clearly visible in his hand throughout the end of the sequence.