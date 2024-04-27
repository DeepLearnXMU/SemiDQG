data_dir = "../../saved_data/data_woi"
new_data_dir = "../../data/data_woi/1k"
saved_dir = "../../ckpt/t5-base-woi-p-qc099-1k-pt-step5k"


class Arguments:
    def __init__(self):
        self.train_file = f"{new_data_dir}/train_ood_with_woi_qc099_1k.json"
        self.valid_file = f"{data_dir}/valid.json"
        self.test_file = f"{data_dir}/test.json"
        self.do_train = True
        self.do_predict = True
        self.model_name_or_path = "t5-base"
        self.posterior = False
        # self.teacher_model_name_or_path = '../saved_data/t5-base-woi-q-1k'
        self.max_source_length = 256
        self.max_target_length = 64
        self.learning_rate = 3e-5
        self.epoches = 4
        self.batch_size = 8
        self.gradient_accumulation_steps = 2
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
        self.max_steps = 5000
        self.save_steps = 1000
