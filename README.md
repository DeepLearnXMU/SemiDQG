# SemiDQG

Code for "Response Enhanced Semi-Supervised Dialogue Query Generation" (AAAI-24).

We share the fundamental tree structure of our repo below. You can reproduce other configuration files based on examples.

```
├─chatgpt               // gpu-3.5-turbo ICL
├─data                  // (empty) generated pseudo data
├─dev
│  ├─distillation       // Stage 2
│  │  └─config
│  │      ├─woi_1k
│  │      └─wow
│  ├─reinforce          // Stage 3
│  │  └─config
│  │      └─wow
│  └─seq2seq            // Stage 1, evaluation, etc.
│      ├─prepare        // preprocessing
│      └─scripts
│          ├─woi_1k
│          └─wow
└─saved_data            // (empty) preprocessed data
```
