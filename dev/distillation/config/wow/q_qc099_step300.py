data_dir = "../../saved_data/data_wow"
new_data_dir = "../../data/data_wow"
saved_dir = "../../ckpt/t5-base-wow-q-qc099-step300"


class Arguments:
    def __init__(self):
        self.train_file = f"{new_data_dir}/train_with_wow_qc099.json"
        self.valid_file = f"{data_dir}/valid.json"
        self.test_file = f"{data_dir}/test.json"
        self.do_train = True
        self.do_predict = True
        self.model_name_or_path = "../../ckpt/t5-base-wow-q"
        self.posterior = True
        self.max_source_length = 256
        self.max_target_length = 64
        self.learning_rate = 3e-5
        self.epoches = 4
        self.batch_size = 8
        self.gradient_accumulation_steps = 8
        self.report_steps = 50
        self.saved_steps = 200
        self.max_length = 64
        self.num_beams = 4
        self.topk = 1
        self.alpha = 0.001
        self.beta = 1.0
        self.gamma = 0.9
        self.clip = 1.0
        self.lang = "en"
        self.warmup_steps = 64
        self.weight_decay = 0.01
        self.mode = "vanilla-kd"  # vanilla-kd or weighted-kd
        self.output_dir = saved_dir
        self.predictions = f"{saved_dir}/generated_predictions.txt"
        self.seed = 42
        self.max_steps = 300
        self.save_steps = 50
