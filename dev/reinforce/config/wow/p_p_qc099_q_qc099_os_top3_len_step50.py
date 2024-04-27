data_dir = "../../saved_data/data_wow"
new_data_dir = "../../data/data_wow"
saved_dir = "../../ckpt/t5-base-wow-p-p-qc099-q-qc099-os-top3-len-step50"


class Arguments:
    def __init__(self):
        self.train_file = f"{new_data_dir}/train_with_wow_p_qc099_q_qc099_scored.json"
        self.valid_file = f"{data_dir}/valid.json"
        self.test_file = f"{data_dir}/test.json"
        self.do_train = True
        self.do_predict = True
        self.model_name_or_path = (
            "../../ckpt/t5-base-wow-p-qc099-step300/checkpoint-100"
        )
        self.max_source_length = 256
        self.max_target_length = 64
        self.learning_rate = 3e-5
        self.epoches = 4
        self.batch_size = 1
        self.gradient_accumulation_steps = 64
        self.max_length = 64
        self.num_beams = 4
        self.topk = 3
        self.clip = 1.0
        self.lang = "en"
        self.threshold = 0.3
        self.save_model = True
        self.posterior = False
        self.warmup_steps = 64
        self.weight_decay = 0.01
        self.output_dir = saved_dir
        self.predictions = f"{saved_dir}/generated_predictions.txt"
        self.seed = 42
        self.max_steps = 50
        self.save_steps = 10
        self.use_smooth_len_norm = False
        self.score_normalization = False
